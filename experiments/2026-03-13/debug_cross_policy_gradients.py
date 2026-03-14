"""
Debug: Are scorer gradients actually different across target policies?

Takes 50 (state, action) pairs from unguided trajectories and computes
grad_log_prob from all 6 target policy scorers. If gradients are similar
across policies, guidance CAN'T differentiate them regardless of scale.
"""
import sys, json, time
import numpy as np
import torch
import h5py
from pathlib import Path

PROJECT_ROOT = Path("/home1/reishuen/latent_sope")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "sope"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "robomimic"))

from opelab.core.baselines.diffusion.temporal import TemporalUnet
from opelab.core.baselines.diffusion.diffusion import GaussianDiffusion
from opelab.core.baselines.diffusion.helpers import EMA, apply_conditioning
from latent_sope.robomimic_interface.checkpoints import load_checkpoint, build_algo_from_checkpoint
from latent_sope.robomimic_interface.guidance import RobomimicDiffusionScorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ──
CKPT_BASE = PROJECT_ROOT / "third_party/robomimic/diffusion_policy_trained_models"
DEMO_HDF5 = PROJECT_ROOT / "third_party/robomimic/datasets/lift/ph/low_dim_v15.hdf5"
TARGET_ROLLOUT_DIR = PROJECT_ROOT / "rollouts" / "target_policy_50"
DIFFUSION_SAVE_DIR = PROJECT_ROOT / "diffusion_ckpts" / "mvp_v0252_traj_mse"
OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

STATE_DIM = 19
ACTION_DIM = 7
TRANSITION_DIM = 26
CHUNK_SIZE = 4
N_DIFFUSION_STEPS = 256
BASE_DIM = 32
DIM_MULTS = (1, 4, 8)
ACTION_WEIGHT = 5.0
SCORE_TIMESTEP = 5
NUM_SYNTHETIC = 50
T_GEN = 60

TARGET_POLICIES = [
    {"name": "10demos_epoch10",  "dir": "lift_diffusion_10demos/20260311115828",  "ckpt": "models/model_epoch_10.pth"},
    {"name": "100demos_epoch20", "dir": "lift_diffusion_100demos/20260311135551", "ckpt": "models/model_epoch_20.pth"},
    {"name": "test_checkpoint",  "dir": "test/20260309132349",                   "ckpt": "last.pth"},
    {"name": "10demos_epoch30",  "dir": "lift_diffusion_10demos/20260311115828",  "ckpt": "models/model_epoch_30.pth"},
    {"name": "50demos_epoch30",  "dir": "lift_diffusion_50demos/20260311134204",  "ckpt": "models/model_epoch_30.pth"},
    {"name": "200demos_epoch40", "dir": "lift_diffusion_200demos/20260311141036", "ckpt": "models/model_epoch_40.pth"},
]

# ── Load normalization ──
target_data = []
all_states_list, all_actions_list = [], []
for path in sorted(TARGET_ROLLOUT_DIR.glob("rollout_*.h5"))[:50]:
    with h5py.File(path, "r") as f:
        latents = f["latents"][:]
        actions = f["actions"][:]
    states = (latents[:, -1, :] if latents.ndim == 3 else latents).astype(np.float32)
    actions = actions.astype(np.float32)
    target_data.append({"states": states, "actions": actions})
    all_states_list.append(states)
    all_actions_list.append(actions)

with h5py.File(DEMO_HDF5, "r") as f:
    for dk in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1])):
        demo = f[f"data/{dk}"]
        s = np.concatenate([demo["obs"][k][:].astype(np.float32) for k in OBS_KEYS], axis=-1)
        a = demo["actions"][:].astype(np.float32)
        all_states_list.append(s)
        all_actions_list.append(a)

all_states = np.concatenate(all_states_list, axis=0)
all_actions = np.concatenate(all_actions_list, axis=0)
norm_mean = np.concatenate([all_states.mean(0), all_actions.mean(0)]).astype(np.float32)
norm_std = np.maximum(np.concatenate([all_states.std(0), all_actions.std(0)]), 1e-6).astype(np.float32)
norm_mean_t = torch.tensor(norm_mean, device=device)
norm_std_t = torch.tensor(norm_std, device=device)
normalize_fn = lambda x: (x - norm_mean_t) / norm_std_t
unnormalize_fn = lambda x: x * norm_std_t + norm_mean_t

# ── Load diffuser and generate unguided trajectories ──
temporal_model = TemporalUnet(
    horizon=CHUNK_SIZE, transition_dim=TRANSITION_DIM,
    dim=BASE_DIM, dim_mults=DIM_MULTS, attention=False,
).to(device)
diffusion_model = GaussianDiffusion(
    model=temporal_model, horizon=CHUNK_SIZE,
    observation_dim=STATE_DIM, action_dim=ACTION_DIM,
    n_timesteps=N_DIFFUSION_STEPS,
    normalizer=normalize_fn, unnormalizer=unnormalize_fn,
    predict_epsilon=False, loss_type="l2",
    clip_denoised=False, action_weight=ACTION_WEIGHT,
).to(device)
ema = EMA(diffusion_model)
ema.ema_model.load_state_dict(
    torch.load(DIFFUSION_SAVE_DIR / "diffusion_model_ema.pt", map_location=device)
)
print("Loaded diffuser")

# Generate unguided trajectories for test points
np.random.seed(42)
torch.manual_seed(42)

initial_states_t = torch.tensor(
    np.array([ep["states"][0] for ep in target_data[:NUM_SYNTHETIC]]),
    dtype=torch.float32, device=device
)

# Quick generation to get test (state, action) pairs
def generate_unguided(dm, init_states, nfn, ufn, sd, ad, cs, tg, dev):
    B = init_states.shape[0]
    td = sd + ad
    pad = torch.cat([init_states, torch.zeros(B, ad, device=dev)], 1)
    cond_init = nfn(pad)[:, :sd]
    all_traj = torch.zeros(B, tg, td, device=dev)
    conditions = {0: cond_init}
    total = 0
    while total < tg:
        x = torch.randn(B, cs, td, device=dev)
        x = apply_conditioning(x, conditions, sd)
        for t_d in reversed(range(dm.n_timesteps)):
            t_t = torch.full((B,), t_d, device=dev, dtype=torch.long)
            with torch.no_grad():
                mm, _, mlv = dm.p_mean_variance(x=x, t=t_t)
                ms = torch.exp(0.5 * mlv)
            noise = torch.randn_like(x)
            x = mm + (1 - (t_d == 0) * 1.0) * ms * noise
            x = apply_conditioning(x, conditions, sd)
        chunk_u = ufn(x)
        n_store = min(cs - 1, tg - total)
        all_traj[:, total:total+n_store] = chunk_u[:, :n_store]
        total += n_store
        if total >= tg:
            break
        conditions = {0: x[:, -1, :sd]}
    return all_traj.detach()

print("Generating unguided trajectories...")
t0 = time.time()
unguided = generate_unguided(ema.ema_model, initial_states_t, normalize_fn, unnormalize_fn,
                              STATE_DIM, ACTION_DIM, CHUNK_SIZE, T_GEN, device)
print(f"Done in {time.time()-t0:.0f}s")

# Extract test chunks: take chunk at t=20 from each trajectory (mid-trajectory)
# Use chunks of size CHUNK_SIZE for scorer evaluation
TEST_T = 20
test_states = unguided[:, TEST_T:TEST_T+CHUNK_SIZE, :STATE_DIM]  # (50, 4, 19)
test_actions = unguided[:, TEST_T:TEST_T+CHUNK_SIZE, STATE_DIM:]  # (50, 4, 7)
print(f"Test chunks: states {test_states.shape}, actions {test_actions.shape}")

# Also test at different trajectory points
TEST_TIMES = [5, 10, 20, 30, 40, 50]
test_chunks = {}
for t in TEST_TIMES:
    if t + CHUNK_SIZE <= T_GEN:
        test_chunks[t] = {
            "states": unguided[:, t:t+CHUNK_SIZE, :STATE_DIM],
            "actions": unguided[:, t:t+CHUNK_SIZE, STATE_DIM:],
        }
print(f"Test points: {list(test_chunks.keys())}")

# ── Load oracle SRs ──
ORACLE_JSON = PROJECT_ROOT / "results/2026-03-12/oracle_eval_all_checkpoints.json"
with open(ORACLE_JSON, "r") as f:
    oracle_all = json.load(f)

oracle_sr_map = {}
for pol in TARGET_POLICIES:
    name = pol["name"]
    if name == "test_checkpoint":
        with open(CKPT_BASE / "test/20260309132349/oracle_50.json", "r") as f:
            test_oracle = json.load(f)
        oracle_sr_map[name] = float(test_oracle["mean_return"])
    else:
        oracle_sr_map[name] = float(oracle_all[name]["mean_return"])

# ── Compute gradients for each target policy ──
print("\n" + "="*80)
print("CROSS-POLICY GRADIENT DIAGNOSTIC")
print("="*80)

all_grads = {}  # name -> (50, 4, 7) gradient tensor

for pol in TARGET_POLICIES:
    name = pol["name"]
    run_dir = CKPT_BASE / pol["dir"]
    ckpt_file = pol["ckpt"]
    osr = oracle_sr_map[name]

    print(f"\n  Loading {name} (oracle={osr*100:.0f}%)...", end=" ", flush=True)
    t0 = time.time()
    ckpt = load_checkpoint(run_dir, ckpt_path=Path(ckpt_file))
    algo = build_algo_from_checkpoint(ckpt, device=str(device))
    scorer = RobomimicDiffusionScorer(algo, device=str(device),
                                       score_timestep=SCORE_TIMESTEP, obs_keys=OBS_KEYS)

    # Compute gradient at t=20 test point
    grad = scorer.grad_log_prob_chunk(test_states, test_actions)  # (50, 4, 7)
    all_grads[name] = grad.cpu()
    print(f"{time.time()-t0:.0f}s, |grad|={grad.norm(dim=-1).mean():.4f}")

    del algo, scorer, ckpt
    torch.cuda.empty_cache()

# ── Analysis ──
print("\n" + "="*80)
print("GRADIENT COMPARISON")
print("="*80)

names = [p["name"] for p in TARGET_POLICIES]
oracle_srs = [oracle_sr_map[n] for n in names]

# 1. Gradient magnitude per policy
print("\n1. Gradient magnitude (mean |grad| over 50 trajs × 4 steps × 7 dims):")
print(f"   {'Policy':<22} {'Oracle':>7} {'|grad| mean':>12} {'|grad| std':>11}")
print("   " + "-"*55)
for name in names:
    g = all_grads[name]
    gnorm = g.norm(dim=-1)  # (50, 4)
    print(f"   {name:<22} {oracle_sr_map[name]*100:>6.0f}% {gnorm.mean():>12.4f} {gnorm.std():>11.4f}")

# 2. Pairwise cosine similarity between policies
print("\n2. Pairwise cosine similarity (mean over 50 trajs × 4 steps):")
# Flatten to (50*4, 7) for each policy
flat_grads = {n: all_grads[n].reshape(-1, 7) for n in names}

print(f"   {'':>22}", end="")
for n in names:
    print(f" {n[:8]:>9}", end="")
print()

for n1 in names:
    print(f"   {n1:<22}", end="")
    for n2 in names:
        g1 = flat_grads[n1]
        g2 = flat_grads[n2]
        # Cosine similarity per sample, then mean
        cos = torch.nn.functional.cosine_similarity(g1, g2, dim=-1)
        print(f" {cos.mean():>9.4f}", end="")
    print()

# 3. Per-trajectory gradient direction variance
print("\n3. Gradient direction spread across policies (per trajectory):")
print("   For each trajectory, how different are the 6 policy gradients?")
# Stack all policy grads: (6, 50, 4, 7)
stacked = torch.stack([all_grads[n] for n in names])  # (6, 50, 4, 7)
# Normalize each gradient
stacked_norm = stacked / (stacked.norm(dim=-1, keepdim=True) + 1e-8)
# For each (traj, step), compute mean pairwise cosine sim across policies
# Shape: (6, 50, 4, 7) -> for each (traj,step) pair, compute all 6 pairwise cosines
n_pol = len(names)
pairwise_cos = []
for i in range(n_pol):
    for j in range(i+1, n_pol):
        cos = (stacked_norm[i] * stacked_norm[j]).sum(dim=-1)  # (50, 4)
        pairwise_cos.append(cos)
pairwise_cos = torch.stack(pairwise_cos)  # (15, 50, 4)
mean_cos = pairwise_cos.mean()
std_cos = pairwise_cos.std()
print(f"   Mean pairwise cosine similarity: {mean_cos:.4f} ± {std_cos:.4f}")
if mean_cos > 0.9:
    print("   >> PROBLEM A CONFIRMED: All policies give nearly identical gradient directions!")
    print("   >> Guidance CAN'T differentiate them regardless of scale.")
elif mean_cos > 0.7:
    print("   >> Gradients are somewhat similar but have some variation.")
    print("   >> Might work with stronger guidance or different application.")
else:
    print("   >> Gradients are reasonably different. Problem is likely B (too weak).")

# 4. Correlation between gradient properties and oracle SR
print("\n4. Does gradient magnitude correlate with oracle SR?")
mag_per_policy = [all_grads[n].norm(dim=-1).mean().item() for n in names]
from scipy import stats
rho_mag, p_mag = stats.spearmanr(oracle_srs, mag_per_policy)
print(f"   Spearman(oracle_sr, |grad|) = {rho_mag:+.4f} (p={p_mag:.4f})")

# 5. Normalized gradient difference: for each pair of policies, how big is the
#    gradient difference relative to the gradient magnitude?
print("\n5. Relative gradient difference between most/least successful policies:")
worst = names[0]  # 10demos_epoch10, 8% SR
best = names[-1]  # 200demos_epoch40, 90% SR
diff = (all_grads[best] - all_grads[worst]).norm(dim=-1)  # (50, 4)
avg_mag = 0.5 * (all_grads[best].norm(dim=-1) + all_grads[worst].norm(dim=-1))
rel_diff = diff / (avg_mag + 1e-8)
print(f"   Policies: {worst} (8%) vs {best} (90%)")
print(f"   |grad_best - grad_worst| = {diff.mean():.4f}")
print(f"   avg |grad| = {avg_mag.mean():.4f}")
print(f"   Relative difference = {rel_diff.mean():.4f} ({rel_diff.mean()*100:.1f}%)")

# 6. Test at multiple trajectory timepoints
print("\n6. Gradient similarity at different trajectory timepoints:")
print(f"   {'Time':>6} {'Mean pairwise cos':>18}")
print("   " + "-"*30)

for t_test, chunks in test_chunks.items():
    s_t = chunks["states"]
    a_t = chunks["actions"]

    # Just reload test_checkpoint scorer quickly for this
    ckpt = load_checkpoint(CKPT_BASE / "test/20260309132349", ckpt_path=Path("last.pth"))
    algo_test = build_algo_from_checkpoint(ckpt, device=str(device))
    scorer_test = RobomimicDiffusionScorer(algo_test, device=str(device),
                                            score_timestep=SCORE_TIMESTEP, obs_keys=OBS_KEYS)
    grad_test = scorer_test.grad_log_prob_chunk(s_t, a_t).cpu()
    del algo_test, scorer_test, ckpt

    # Compare with worst policy
    ckpt = load_checkpoint(CKPT_BASE / "lift_diffusion_10demos/20260311115828",
                            ckpt_path=Path("models/model_epoch_10.pth"))
    algo_worst = build_algo_from_checkpoint(ckpt, device=str(device))
    scorer_worst = RobomimicDiffusionScorer(algo_worst, device=str(device),
                                             score_timestep=SCORE_TIMESTEP, obs_keys=OBS_KEYS)
    grad_worst = scorer_worst.grad_log_prob_chunk(s_t, a_t).cpu()
    del algo_worst, scorer_worst, ckpt
    torch.cuda.empty_cache()

    cos = torch.nn.functional.cosine_similarity(
        grad_test.reshape(-1, 7), grad_worst.reshape(-1, 7), dim=-1
    )
    print(f"   t={t_test:>3}  {cos.mean():>18.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Mean pairwise cosine similarity across all 6 policies: {mean_cos:.4f}")
print(f"Spearman(oracle_sr, |grad|): {rho_mag:+.4f}")
if mean_cos > 0.9:
    print("\nDIAGNOSIS: Problem A — scorer gradients are nearly identical across policies.")
    print("The diffusion score function at t=5 doesn't distinguish these policies.")
    print("Possible fixes:")
    print("  1. Use a different score_timestep (higher t = noisier, might differentiate more)")
    print("  2. Use a non-diffusion scorer (e.g., train separate BC-Gaussian per policy)")
    print("  3. The policies may genuinely be too similar in action space")
elif mean_cos > 0.7:
    print("\nDIAGNOSIS: Partial A — gradients are similar but not identical.")
    print("The gradient signal exists but is weak relative to the diffuser's prior.")
else:
    print("\nDIAGNOSIS: Problem B — gradients differ but guidance is too weak.")
    print("Try stronger action_scale or apply guidance differently.")
