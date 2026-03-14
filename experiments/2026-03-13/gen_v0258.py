#!/usr/bin/env python3
"""Generate MVP v0.2.5.8 notebook: Small diffusion MLP scorers for cross-policy guidance."""
import json

cells = []

def md(source):
    lines = source.strip().split("\n")
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]})

def code(source):
    lines = source.strip().split("\n")
    cells.append({"cell_type": "code", "metadata": {},
                  "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
                  "outputs": [], "execution_count": None})

# ── Cell 0: Header ──
md("""\
# MVP v0.2.5.8: Small Diffusion MLP Scorers for Cross-Policy Guidance

**Date:** 2026-03-13
**Builds on:** v0.2.5.6 (cross-policy failed, rho=-0.10), gradient debug (cos=0.86-0.95)

## Goal

Test whether replacing the robomimic UNet scorer with small, independently-trained
diffusion MLPs (matching SOPE's actual approach) enables cross-policy guidance.

**Root cause of v0.2.5.6 failure:** All 6 robomimic policies share the same 65M-param
UNet architecture and produce nearly identical score functions at t=5 (cosine sim
0.86-0.95). SOPE uses small MLPs (256 hidden, 32 diffusion steps) trained independently
per target policy on that policy's rollout data.

**Approach:**
1. Collect 10 rollouts from each of 6 target policies (using simulator)
2. Train a small diffusion MLP per policy on its (state, action) data
3. Verify gradient directions differ across policies (cosine sim diagnostic)
4. Run cross-policy guided generation using small MLP scorers

**Success criterion:** Spearman rho > 0.5 between guided OPE and oracle SR""")

# ── Cell 1: Imports + config ──
code("""\
%matplotlib inline
import sys, os, importlib
import numpy as np
import torch
import torch.nn as nn
import h5py, json, math, time
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path("/home1/reishuen/latent_sope")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "sope"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "robomimic"))

from opelab.core.baselines.diffusion.temporal import TemporalUnet
from opelab.core.baselines.diffusion.diffusion import GaussianDiffusion
from opelab.core.baselines.diffusion.helpers import EMA, apply_conditioning
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from latent_sope.robomimic_interface.checkpoints import (
    load_checkpoint, build_algo_from_checkpoint,
    build_rollout_policy_from_checkpoint, build_env_from_checkpoint,
)
from latent_sope.robomimic_interface.rollout import rollout as do_rollout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Paths
CKPT_BASE = PROJECT_ROOT / "third_party/robomimic/diffusion_policy_trained_models"
DEMO_HDF5 = PROJECT_ROOT / "third_party/robomimic/datasets/lift/ph/low_dim_v15.hdf5"
TARGET_ROLLOUT_DIR = PROJECT_ROOT / "rollouts" / "target_policy_50"
DIFFUSION_SAVE_DIR = PROJECT_ROOT / "diffusion_ckpts" / "mvp_v0252_traj_mse"
ORACLE_JSON = PROJECT_ROOT / "results/2026-03-12/oracle_eval_all_checkpoints.json"
OBS_KEYS = sorted(["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"])

# Dims
STATE_DIM = 19
ACTION_DIM = 7
TRANSITION_DIM = 26
CUBE_Z_INDEX = 2
LIFT_THRESHOLD = 0.84

# Chunk diffuser config
CHUNK_SIZE = 4
N_DIFFUSION_STEPS = 256
BASE_DIM = 32
DIM_MULTS = (1, 4, 8)
ACTION_WEIGHT = 5.0

# Generation config
NUM_SYNTHETIC = 50
T_GEN = 60

# Small MLP config (matching SOPE)
MLP_HIDDEN = 256
MLP_EMB = 64
MLP_DIFFUSION_STEPS = 32
MLP_TRAIN_STEPS = 5000
MLP_BATCH_SIZE = 256
MLP_LR = 3e-4
N_ROLLOUTS_PER_POLICY = 10

# Guidance config
ACTION_SCALE = 0.01  # middle ground between our 0.001 and SOPE's 0.05

TARGET_POLICIES = [
    {"name": "10demos_epoch10",  "dir": "lift_diffusion_10demos/20260311115828",  "ckpt": "models/model_epoch_10.pth"},
    {"name": "100demos_epoch20", "dir": "lift_diffusion_100demos/20260311135551", "ckpt": "models/model_epoch_20.pth"},
    {"name": "test_checkpoint",  "dir": "test/20260309132349",                   "ckpt": "last.pth"},
    {"name": "10demos_epoch30",  "dir": "lift_diffusion_10demos/20260311115828",  "ckpt": "models/model_epoch_30.pth"},
    {"name": "50demos_epoch30",  "dir": "lift_diffusion_50demos/20260311134204",  "ckpt": "models/model_epoch_30.pth"},
    {"name": "200demos_epoch40", "dir": "lift_diffusion_200demos/20260311141036", "ckpt": "models/model_epoch_40.pth"},
]

print(f"action_scale={ACTION_SCALE}, {N_ROLLOUTS_PER_POLICY} rollouts/policy")
print(f"{NUM_SYNTHETIC} trajs, T_GEN={T_GEN}")
print(f"MLP: {MLP_HIDDEN} hidden, {MLP_EMB} emb, {MLP_DIFFUSION_STEPS} steps, {MLP_TRAIN_STEPS} train steps")""")

# ── Cell 2: SmallDiffusionScorer class ──
code("""\
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SmallDiffusionScorer(nn.Module):
    \\"\\"\\"Small diffusion MLP scorer matching SOPE's PearceMlp architecture.

    Single-step noise prediction (no temporal UNet), 32 diffusion steps,
    linear beta schedule. Trained via standard diffusion loss on (state, action) pairs.
    \\"\\"\\"

    def __init__(self, state_dim, action_dim, hidden_dim=256, emb_dim=64,
                 diffusion_steps=32):
        super().__init__()
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.Mish(),
        )
        self.state_emb = nn.Sequential(
            nn.Linear(state_dim, emb_dim),
            nn.Mish(),
        )
        self.net = nn.Sequential(
            nn.Linear(action_dim + 2 * emb_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Linear schedule (matching SOPE)
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_ac', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_1mac', (1 - alphas_cumprod).sqrt())

    def forward(self, action, timestep, state):
        t_emb = self.time_emb(timestep)
        s_emb = self.state_emb(state)
        x = torch.cat([action, t_emb, s_emb], dim=-1)
        return self.net(x)

    def train_on_data(self, states, actions, n_steps=5000, batch_size=256, lr=3e-4):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        dev = next(self.parameters()).device
        states_t = torch.tensor(states, dtype=torch.float32, device=dev)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=dev)
        N = len(states_t)

        self.train()
        for step in range(n_steps):
            idx = torch.randint(0, N, (min(batch_size, N),))
            s, a = states_t[idx], actions_t[idx]
            t = torch.randint(0, self.diffusion_steps, (len(idx),), device=dev)
            eps = torch.randn_like(a)
            a_t = self.sqrt_ac[t, None] * a + self.sqrt_1mac[t, None] * eps
            eps_pred = self(a_t, t, s)
            loss = ((eps - eps_pred) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.eval()
        return loss.item()

    @property
    def sigma_1(self):
        return self.sqrt_1mac[1].item()

    @torch.no_grad()
    def grad_log_prob(self, state, action):
        B = state.shape[0]
        t = torch.ones(B, device=state.device, dtype=torch.long)
        eps_pred = self(action, t, state)
        return -eps_pred / self.sigma_1

    @torch.no_grad()
    def grad_log_prob_chunk(self, states, actions):
        B, T, _ = states.shape
        return self.grad_log_prob(
            states.reshape(B * T, -1), actions.reshape(B * T, -1)
        ).reshape(B, T, -1)

print(f"SmallDiffusionScorer defined. sigma[1] = {SmallDiffusionScorer(19, 7).sigma_1:.4f}")""")

# ── Cell 3: Load infrastructure ──
code("""\
# ── Oracle values ──
with open(ORACLE_JSON, "r") as f:
    oracle_all = json.load(f)

oracle_sr_map = {}
for pol in TARGET_POLICIES:
    name = pol["name"]
    if name == "test_checkpoint":
        with open(CKPT_BASE / "test/20260309132349/oracle_50.json", "r") as f:
            oracle_sr_map[name] = float(json.load(f)["mean_return"])
    else:
        oracle_sr_map[name] = float(oracle_all[name]["mean_return"])

print("Oracle SR values:")
for name, sr in oracle_sr_map.items():
    print(f"  {name:<22} {sr*100:.0f}%")

# ── Normalization (from target rollouts + expert demos, same as training) ──
all_states_list, all_actions_list = [], []
for path in sorted(TARGET_ROLLOUT_DIR.glob("rollout_*.h5"))[:50]:
    with h5py.File(path, "r") as f:
        latents = f["latents"][:]
        actions = f["actions"][:]
    states = (latents[:, -1, :] if latents.ndim == 3 else latents).astype(np.float32)
    all_states_list.append(states)
    all_actions_list.append(actions.astype(np.float32))

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

# Initial states for generation (from test checkpoint rollouts)
target_data = []
for path in sorted(TARGET_ROLLOUT_DIR.glob("rollout_*.h5"))[:50]:
    with h5py.File(path, "r") as f:
        latents = f["latents"][:]
    states = (latents[:, -1, :] if latents.ndim == 3 else latents).astype(np.float32)
    target_data.append(states)

initial_states_t = torch.tensor(
    np.array([ep[0] for ep in target_data[:NUM_SYNTHETIC]]),
    dtype=torch.float32, device=device
)
print(f"\\nInitial states: {initial_states_t.shape}")
print("Normalization computed from 50 target rollouts + 200 expert demos")""")

# ── Cell 4: Collect rollouts from each target policy ──
code("""\
# ── Collect rollouts from each target policy ──
print(f"Collecting {N_ROLLOUTS_PER_POLICY} rollouts per policy...")
print(f"Estimated time: {N_ROLLOUTS_PER_POLICY * len(TARGET_POLICIES) * 15 / 60:.0f} min\\n")

policy_train_data = {}
t0_all = time.time()

for i, pol in enumerate(TARGET_POLICIES):
    name = pol["name"]
    run_dir = CKPT_BASE / pol["dir"]
    ckpt_file = pol["ckpt"]
    osr = oracle_sr_map[name]

    print(f"[{i+1}/{len(TARGET_POLICIES)}] {name} (oracle={osr*100:.0f}%)", end=" ", flush=True)
    t0 = time.time()

    ckpt = load_checkpoint(run_dir, ckpt_path=Path(ckpt_file))
    rollout_policy = build_rollout_policy_from_checkpoint(ckpt, device="cpu", verbose=False)
    env = build_env_from_checkpoint(ckpt, render=False, render_offscreen=False, verbose=False)

    all_states, all_actions = [], []
    for ep in range(N_ROLLOUTS_PER_POLICY):
        obs = env.reset()
        rollout_policy.start_episode()
        for t in range(T_GEN):
            act = rollout_policy(obs)
            state = np.concatenate([obs[k].flatten() for k in OBS_KEYS])
            all_states.append(state.astype(np.float32))
            all_actions.append(act.astype(np.float32))
            obs, reward, done, info = env.step(act)
            if done:
                break

    states_arr = np.array(all_states, dtype=np.float32)
    actions_arr = np.array(all_actions, dtype=np.float32)
    policy_train_data[name] = {"states": states_arr, "actions": actions_arr}

    elapsed = time.time() - t0
    print(f"— {elapsed:.0f}s, {len(states_arr)} transitions, "
          f"actions shape {actions_arr.shape}")

    del rollout_policy, env, ckpt
    torch.cuda.empty_cache()

total_collect = time.time() - t0_all
print(f"\\nTotal collection: {total_collect:.0f}s ({total_collect/60:.1f} min)")""")

# ── Cell 5: Train small MLP per policy + gradient comparison ──
code("""\
# ── Train small diffusion MLP per policy ──
print("Training small diffusion MLPs...\\n")

mlp_scorers = {}
t0_all = time.time()

for name, data in policy_train_data.items():
    print(f"  {name:<22}", end=" ", flush=True)
    t0 = time.time()

    scorer = SmallDiffusionScorer(
        STATE_DIM, ACTION_DIM,
        hidden_dim=MLP_HIDDEN, emb_dim=MLP_EMB,
        diffusion_steps=MLP_DIFFUSION_STEPS,
    ).to(device)

    final_loss = scorer.train_on_data(
        data["states"], data["actions"],
        n_steps=MLP_TRAIN_STEPS, batch_size=MLP_BATCH_SIZE, lr=MLP_LR,
    )
    mlp_scorers[name] = scorer
    print(f"loss={final_loss:.6f}, {time.time()-t0:.0f}s")

total_train = time.time() - t0_all
print(f"\\nTotal training: {total_train:.0f}s")

# ── Gradient comparison: are small MLP gradients different across policies? ──
print(f"\\n{'='*80}")
print("GRADIENT COMPARISON: Small MLP vs RobomimicUNet")
print(f"{'='*80}")

# Generate test data (same unguided trajectories as before)
# Load chunk diffuser first for test data generation
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
print(f"\\nLoaded chunk diffuser from {DIFFUSION_SAVE_DIR}")

# Quick unguided generation for test chunks
np.random.seed(42)
torch.manual_seed(42)

def generate_unguided_quick(dm, init, nfn, ufn, sd, ad, cs, tg, dev):
    B = init.shape[0]
    td = sd + ad
    pad = torch.cat([init, torch.zeros(B, ad, device=dev)], 1)
    cond = {0: nfn(pad)[:, :sd]}
    traj = torch.zeros(B, tg, td, device=dev)
    total = 0
    while total < tg:
        x = torch.randn(B, cs, td, device=dev)
        x = apply_conditioning(x, cond, sd)
        for t_d in reversed(range(dm.n_timesteps)):
            t_t = torch.full((B,), t_d, device=dev, dtype=torch.long)
            with torch.no_grad():
                mm, _, mlv = dm.p_mean_variance(x=x, t=t_t)
                ms = torch.exp(0.5 * mlv)
            noise = torch.randn_like(x)
            x = mm + (1 - (t_d == 0) * 1.0) * ms * noise
            x = apply_conditioning(x, cond, sd)
        chunk_u = ufn(x)
        ns = min(cs - 1, tg - total)
        traj[:, total:total+ns] = chunk_u[:, :ns]
        total += ns
        if total >= tg:
            break
        cond = {0: x[:, -1, :sd]}
    return traj.detach()

print("Generating unguided test trajectories...")
unguided = generate_unguided_quick(
    ema.ema_model, initial_states_t, normalize_fn, unnormalize_fn,
    STATE_DIM, ACTION_DIM, CHUNK_SIZE, T_GEN, device
)
test_s = unguided[:, 20:24, :STATE_DIM]  # (50, 4, 19)
test_a = unguided[:, 20:24, STATE_DIM:]  # (50, 4, 7)

# Compute gradients from each small MLP
names = [p["name"] for p in TARGET_POLICIES]
mlp_grads = {}
for name in names:
    g = mlp_scorers[name].grad_log_prob_chunk(test_s, test_a)
    mlp_grads[name] = g.cpu()

# Pairwise cosine similarity
print(f"\\nPairwise cosine similarity (Small MLPs):")
flat_grads = {n: mlp_grads[n].reshape(-1, ACTION_DIM) for n in names}

print(f"   {'':>22}", end="")
for n in names:
    print(f" {n[:8]:>9}", end="")
print()

for n1 in names:
    print(f"   {n1:<22}", end="")
    for n2 in names:
        cos = torch.nn.functional.cosine_similarity(flat_grads[n1], flat_grads[n2], dim=-1)
        print(f" {cos.mean():>9.4f}", end="")
    print()

# Overall mean pairwise cosine
n_pol = len(names)
pairwise = []
for i in range(n_pol):
    for j in range(i+1, n_pol):
        cos = torch.nn.functional.cosine_similarity(flat_grads[names[i]], flat_grads[names[j]], dim=-1)
        pairwise.append(cos.mean().item())
mean_cos = np.mean(pairwise)
print(f"\\nMean pairwise cosine: {mean_cos:.4f}")
print(f"(v0.2.5.6 UNet scorers: 0.7260)")
if mean_cos < 0.5:
    print(">> Small MLPs produce substantially different gradients!")
elif mean_cos < 0.7:
    print(">> Improvement over UNet scorers, moderate differentiation")
else:
    print(">> Still too similar — may not differentiate policies")""")

# ── Cell 6: Generate guided trajectories ──
code("""\
def generate_trajectories(
    diffusion_model, initial_states,
    normalize_fn, unnormalize_fn,
    state_dim, action_dim, chunk_size, t_gen, device,
    target_scorer=None, action_scale=0.0, normalize_grad=True,
):
    guided = (target_scorer is not None and action_scale > 0)
    B = initial_states.shape[0]
    td = state_dim + action_dim
    n_ts = diffusion_model.n_timesteps

    pad = torch.cat([initial_states, torch.zeros(B, action_dim, device=device)], 1)
    cond_init = normalize_fn(pad)[:, :state_dim]
    all_traj = torch.zeros(B, t_gen, td, device=device)
    conditions = {0: cond_init}
    total = 0

    while total < t_gen:
        x = torch.randn(B, chunk_size, td, device=device)
        x = apply_conditioning(x, conditions, state_dim)
        for t_d in reversed(range(n_ts)):
            t_t = torch.full((B,), t_d, device=device, dtype=torch.long)
            with torch.no_grad():
                mm, _, mlv = diffusion_model.p_mean_variance(x=x, t=t_t)
                ms = torch.exp(0.5 * mlv)
            if guided:
                mm = unnormalize_fn(mm)
                sc = mm[:, :, :state_dim]
                ac = mm[:, :, state_dim:]
                tg = target_scorer.grad_log_prob_chunk(sc, ac)
                if normalize_grad:
                    tg = tg / (tg.norm(dim=-1, keepdim=True) + 1e-6)
                guide = torch.zeros_like(mm)
                guide[:, :, state_dim:] = tg
                mm = mm + action_scale * guide
                mm = normalize_fn(mm)
                mm = apply_conditioning(mm, conditions, state_dim)
                mm = unnormalize_fn(mm)
                mm = normalize_fn(mm)
            noise = torch.randn_like(x)
            x = mm + (1 - (t_d == 0) * 1.0) * ms * noise
            x = apply_conditioning(x, conditions, state_dim)
        chunk_u = unnormalize_fn(x)
        n_store = min(chunk_size - 1, t_gen - total)
        all_traj[:, total:total+n_store] = chunk_u[:, :n_store]
        total += n_store
        if total >= t_gen:
            break
        conditions = {0: x[:, -1, :state_dim]}
    return all_traj.detach().cpu().numpy()

# ── Generate unguided (once) ──
print("Generating unguided trajectories...")
np.random.seed(42)
torch.manual_seed(42)
t0 = time.time()
unguided_trajs = generate_trajectories(
    ema.ema_model, initial_states_t, normalize_fn, unnormalize_fn,
    STATE_DIM, ACTION_DIM, CHUNK_SIZE, T_GEN, device,
)
unguided_states = unguided_trajs[:, :, :STATE_DIM]
unguided_sr = np.mean([np.any(unguided_states[j, :, CUBE_Z_INDEX] > LIFT_THRESHOLD)
                       for j in range(NUM_SYNTHETIC)])
print(f"Unguided: SR={unguided_sr*100:.0f}%, {time.time()-t0:.0f}s")

# ── Generate guided per policy ──
results = {}
t0_all = time.time()

for i, pol in enumerate(TARGET_POLICIES):
    name = pol["name"]
    osr = oracle_sr_map[name]
    scorer = mlp_scorers[name]

    print(f"\\n[{i+1}/{len(TARGET_POLICIES)}] {name} (oracle={osr*100:.0f}%)", end=" ", flush=True)

    np.random.seed(42)
    torch.manual_seed(42)

    t0 = time.time()
    guided_trajs = generate_trajectories(
        ema.ema_model, initial_states_t, normalize_fn, unnormalize_fn,
        STATE_DIM, ACTION_DIM, CHUNK_SIZE, T_GEN, device,
        target_scorer=scorer, action_scale=ACTION_SCALE, normalize_grad=True,
    )
    gen_time = time.time() - t0

    gs = guided_trajs[:, :, :STATE_DIM]
    guided_sr = np.mean([np.any(gs[j, :, CUBE_Z_INDEX] > LIFT_THRESHOLD)
                         for j in range(NUM_SYNTHETIC)])

    results[name] = {
        "oracle_sr": osr,
        "guided_sr": guided_sr,
        "guided_states": gs,
        "gen_time": gen_time,
    }
    print(f"— {gen_time:.0f}s, Guided SR={guided_sr*100:.0f}%")

total_gen = time.time() - t0_all
print(f"\\nTotal generation: {total_gen:.0f}s ({total_gen/60:.1f} min)")""")

# ── Cell 7: Summary table + Spearman ──
code("""\
# ── Summary table ──
print(f"{'='*85}")
print(f"v0.2.5.8 SMALL MLP CROSS-POLICY GUIDANCE TEST")
print(f"action_scale={ACTION_SCALE}, MLP: {MLP_HIDDEN}h, {MLP_DIFFUSION_STEPS} steps, "
      f"{MLP_TRAIN_STEPS} train steps")
print(f"{'='*85}")

oracle_srs, guided_opes, unguided_opes = [], [], []

print(f"\\n{'Policy':<22} {'Oracle':>7} {'Unguided':>9} {'Guided':>8} "
      f"{'D(G-U)':>8} {'G err':>7} {'U err':>7}")
print("-" * 75)

for name in [p["name"] for p in TARGET_POLICIES]:
    r = results[name]
    osr = r["oracle_sr"]
    gsr = r["guided_sr"]
    delta = gsr - unguided_sr
    g_err = abs(gsr - osr) / (osr + 1e-8) * 100
    u_err = abs(unguided_sr - osr) / (osr + 1e-8) * 100

    oracle_srs.append(osr)
    guided_opes.append(gsr)
    unguided_opes.append(unguided_sr)

    print(f"{name:<22} {osr*100:>6.0f}% {unguided_sr*100:>8.0f}% {gsr*100:>7.0f}% "
          f"{delta*100:>+7.0f}% {g_err:>6.1f}% {u_err:>6.1f}%")

# Spearman
rho_guided, p_guided = stats.spearmanr(oracle_srs, guided_opes)
print(f"\\n{'='*85}")
print(f"Spearman rho (guided  vs oracle): {rho_guided:+.4f}  (p={p_guided:.4f})")
print(f"Unguided OPE (flat baseline): {unguided_sr:.3f}")
print(f"Guided OPE range: [{min(guided_opes):.3f}, {max(guided_opes):.3f}]")
print(f"Gradient cosine sim (small MLP): {mean_cos:.4f} vs (UNet): 0.7260")

if rho_guided > 0.7:
    print(f"\\nSUCCESS: Small MLP guidance ranks policies (rho={rho_guided:.2f})")
elif rho_guided > 0.3:
    print(f"\\nPARTIAL: Some correlation (rho={rho_guided:.2f}), needs tuning")
else:
    print(f"\\nFAIL: Small MLP guidance does not rank policies (rho={rho_guided:.2f})")""")

# ── Cell 8: Figures ──
code("""\
# ── Figures ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Scatter
ax = axes[0]
ax.scatter(np.array(oracle_srs)*100, np.array(guided_opes)*100,
           s=120, c="coral", edgecolor="black", zorder=5,
           label=f"Guided (rho={rho_guided:.2f})")
ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label="Perfect")
ax.axhline(y=unguided_sr*100, color="steelblue", ls=":", alpha=0.7,
           label=f"Unguided={unguided_sr*100:.0f}%")
for j, name in enumerate([p["name"] for p in TARGET_POLICIES]):
    ax.annotate(name.replace("demos_epoch", "e").replace("test_checkpoint", "test"),
                (oracle_srs[j]*100, guided_opes[j]*100),
                textcoords="offset points", xytext=(5, 5), fontsize=7)
ax.set_xlabel("Oracle SR (%)")
ax.set_ylabel("Guided OPE (%)")
ax.set_title("Oracle vs Guided OPE (Small MLP Scorers)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-5, 100])
ax.set_ylim([-5, 100])

# Panel 2: Bar chart
ax = axes[1]
x = np.arange(len(TARGET_POLICIES))
w = 0.25
short_names = [p["name"].replace("demos_epoch", "e").replace("test_checkpoint", "test")
               for p in TARGET_POLICIES]
ax.bar(x - w, np.array(oracle_srs)*100, w, color="green", edgecolor="black", label="Oracle")
ax.bar(x, [unguided_sr*100]*len(x), w, color="steelblue", edgecolor="black", label="Unguided")
ax.bar(x + w, np.array(guided_opes)*100, w, color="coral", edgecolor="black", label="Guided")
ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("SR / OPE (%)")
ax.set_title("Per-Policy: Oracle vs Unguided vs Guided")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

plt.suptitle(f"v0.2.5.8: Small MLP Scorers (scale={ACTION_SCALE}, rho={rho_guided:.2f})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()""")

# ── Cell 9: Cube z grid ──
code("""\
# ── Cube z trajectory grid ──
n_policies = len(TARGET_POLICIES)
fig, axes = plt.subplots(n_policies, 2, figsize=(14, 4*n_policies))

for row, pol in enumerate(TARGET_POLICIES):
    name = pol["name"]
    r = results[name]
    osr = r["oracle_sr"]
    gs = r["guided_states"]

    ax = axes[row, 0]
    for j in range(min(15, NUM_SYNTHETIC)):
        ax.plot(unguided_states[j, :, CUBE_Z_INDEX], color="steelblue", alpha=0.15)
    ax.plot(unguided_states[:, :, CUBE_Z_INDEX].mean(0), color="darkblue", lw=2)
    ax.axhline(y=LIFT_THRESHOLD, color="red", ls=":", alpha=0.5)
    ax.set_ylim([0.78, 0.95])
    ax.set_title(f"Unguided — SR={unguided_sr*100:.0f}%", fontsize=9)
    ax.set_ylabel(f"{name}\\n(oracle={osr*100:.0f}%)", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[row, 1]
    for j in range(min(15, NUM_SYNTHETIC)):
        ax.plot(gs[j, :, CUBE_Z_INDEX], color="coral", alpha=0.15)
    ax.plot(gs[:, :, CUBE_Z_INDEX].mean(0), color="darkred", lw=2)
    ax.axhline(y=LIFT_THRESHOLD, color="red", ls=":", alpha=0.5)
    ax.set_ylim([0.78, 0.95])
    ax.set_title(f"Guided -> {name} — SR={r['guided_sr']*100:.0f}%", fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("v0.2.5.8: Cube Z — Unguided (left) vs Small MLP Guided (right)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()""")

# ── Cell 10: Summary text ──
code("""\
fig, ax = plt.subplots(1, 1, figsize=(10, 9))
ax.axis("off")

lines = [
    "v0.2.5.8: Small Diffusion MLP Scorers",
    "=" * 55,
    "",
    f"Approach: Train small MLP (SOPE-style) per target policy",
    f"MLP: {MLP_HIDDEN}h, {MLP_EMB}emb, {MLP_DIFFUSION_STEPS} diff steps",
    f"Training: {MLP_TRAIN_STEPS} steps on {N_ROLLOUTS_PER_POLICY} rollouts/policy",
    f"Guidance: action_scale={ACTION_SCALE}, normalize_grad=True",
    "",
    f"Gradient cosine sim (Small MLP): {mean_cos:.4f}",
    f"Gradient cosine sim (UNet):      0.7260",
    "",
]

lines.append(f"{'Policy':<22} {'Oracle':>7} {'Guided':>8}")
lines.append("-" * 40)
for name in [p["name"] for p in TARGET_POLICIES]:
    r = results[name]
    lines.append(f"{name:<22} {r['oracle_sr']*100:>6.0f}% {r['guided_sr']*100:>7.0f}%")

lines += [
    "",
    f"Spearman rho (guided vs oracle): {rho_guided:+.4f} (p={p_guided:.4f})",
    f"Unguided OPE (flat): {unguided_sr:.3f}",
    "",
]

if rho_guided > 0.7:
    lines.append("VERDICT: SUCCESS — Small MLP guidance ranks policies")
elif rho_guided > 0.3:
    lines.append("VERDICT: PARTIAL — some correlation, needs tuning")
else:
    lines.append("VERDICT: FAIL — small MLPs don't help enough")

lines += [
    "",
    f"Rollout collection: {total_collect:.0f}s",
    f"MLP training: {total_train:.0f}s",
    f"Generation: {total_gen:.0f}s",
]

ax.text(0.05, 0.95, "\\n".join(lines), transform=ax.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace")
plt.tight_layout()
plt.show()""")

# ── Assemble notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

outpath = "/home1/reishuen/latent_sope/experiments/2026-03-13/MVP_v0.2.5.8_small_mlp_scorers.ipynb"
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {outpath}")
