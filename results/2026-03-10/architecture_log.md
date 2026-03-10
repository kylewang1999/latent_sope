# Latent SOPE Pipeline — Architecture Log

**Date:** 2026-03-10
**Purpose:** Capture all design choices in the current pipeline for comparison with the SOPE reference implementation (Stitch-OPE). This document covers what we chose, why, and whether it was the right call.

---

## 1. High-Level Overview

The pipeline estimates a policy's value **without running it online** by:
1. Collecting a small offline dataset of rollouts under the target policy
2. Chunking those rollouts into short trajectory segments
3. Training a diffusion model to generate realistic chunks
4. Stitching chunks autoregressively into full synthetic trajectories
5. Scoring synthetic trajectories with a reward function
6. Comparing the mean synthetic return against a ground-truth oracle

The key hypothesis: if the diffusion model learns the chunk distribution well enough, stitched trajectories will have realistic dynamics, and mean returns will approximate the true policy value.

---

## 2. Architecture Summary Table

| Component | Our Pipeline | SOPE Reference |
|-----------|-------------|----------------|
| **State representation** | 19-dim low-dim obs concat | Raw env state (varies by task) |
| **Action dim** | 7 | Varies by task |
| **Transition dim** | 26 (19+7) | state_dim + action_dim |
| **Chunk horizon (T)** | 8 (was 10, now 8) | 8 |
| **Frame stack** | 1 (was 2, now 1) | 1 (condition at index 0) |
| **Total horizon** | 8 (7+1) | 8 |
| **UNet base dim** | 32 | 32 |
| **UNet dim_mults** | (1,4,8) (was (1,2)) | (1,4,8) or (1,2,4,8) |
| **UNet attention** | False | False (optional) |
| **Diffusion steps** | 256 | 1000 |
| **Noise schedule** | Cosine | Cosine (s=0.008) |
| **Predict epsilon** | True | False (predict x₀) |
| **Loss type** | L2 | L2 |
| **Action weight** | 5.0 | 5.0 |
| **Loss discount** | 1.0 | 1.0 |
| **Training steps** | 500k (was 8.5k) | 500k |
| **Steps/epoch** | 5000 (with replacement) | 5000 (with replacement) |
| **Batch size** | 64 | 32 (D4RL default) |
| **LR** | 3e-4 | 1e-3 (D4RL default) |
| **LR scheduler** | CosineAnnealingLR (T_max=2×epochs) | CosineAnnealingLR |
| **Gradient clip** | 1.0 | Not specified |
| **Guidance** | None (skipped) | Policy gradient guidance |
| **Reward model** | Ground-truth analytical | Learned MLP ensemble |
| **Normalization** | Per-feature mean/std | Per-feature mean/std |
| **Stitching overlap** | 0 states (no overlap) | 1 state overlap |
| **Rollout horizon** | 60 | 768 (D4RL) |
| **Num rollouts** | 50 | ~1000 trajectories (D4RL datasets) |
| **Num chunks** | ~1300 | ~125,000 (D4RL) |
| **Oracle rollouts** | 50 | 1000 |
| **Gamma** | 1.0 | 1.0 |

---

## 3. Design Choices — Deep Dive

### 3.1 State Representation: Low-Dim Obs Concatenation (19-dim)

**What:** Instead of using a learned encoder (CNN features, VAE latents, etc.), we concatenate sorted low-dim observation keys into a flat vector:

| Index | Key | Dim | Content |
|-------|-----|-----|---------|
| 0–9 | `object` | 10 | cube_pos(3) + cube_quat(4) + gripper_to_cube(3) |
| 10–12 | `robot0_eef_pos` | 3 | end-effector XYZ |
| 13–16 | `robot0_eef_quat` | 4 | end-effector quaternion |
| 17–18 | `robot0_gripper_qpos` | 2 | gripper joint positions |

**Action dimensions (7-dim, OSC_POSE controller):**

| Index | Content |
|-------|---------|
| 0–2 | End-effector position delta (dx, dy, dz) |
| 3–5 | End-effector orientation delta (axis-angle rotation) |
| 6 | Gripper action (-1 = open, +1 = close) |

All action values are typically in [-1, 1]. The robosuite Lift environment uses the OSC_POSE (Operational Space Control — Pose) controller: 3 dims for translational movement, 3 dims for rotational movement, and 1 dim for the gripper.

**Why:** Simplicity. Lift is a simple task with full low-dim observability. No need to learn an encoder when the observations already contain everything the policy sees.

**Assessment: Mostly good, with caveats.**
- **Pro:** Zero encoder error. The latent space *is* the observation space, so there's no information bottleneck or representation learning failure mode. The reward function can directly read cube_z from index 2 without any decoder.
- **Pro:** Makes debugging transparent — every dimension has a known physical meaning.
- **Con:** Quaternions are a poor representation for diffusion. They live on a 4D unit sphere (‖q‖=1), but the diffusion model generates them in unconstrained ℝ⁴. The 8.5k-step results showed quaternion dims (obj_qy, obj_qz) blowing up 10,000-15,000×. Quaternions have inherent discontinuities (q and -q represent the same rotation) that confuse L2 regression.
- **Con:** The 19-dim space mixes quantities with very different scales and geometries: positions (≈0.0-1.1), quaternions ([-1,1] with unit norm constraint), and gripper positions (≈0.0-0.05). Mean/std normalization helps but doesn't fix the geometric mismatch.
- **Alternative not tried:** 6D rotation representation (continuous, no antipodal issues), or dropping quaternions entirely since Lift reward only depends on cube z-position.
- **SOPE comparison:** SOPE operates on raw MuJoCo states (qpos/qvel), which also mix different units but are typically higher-dimensional. Our space is actually more compact.

#### Experiment Plan: Getting a Positive Signal for Low-Dim Obs

The goal is to get the pipeline working end-to-end on the simplest possible version of the problem, then add complexity back. Each experiment is designed to isolate one variable.

**Phase A: Wait for the 500k-step run (in progress)**

The 500k run (job 7267279) already addresses the two most critical v1 failures: model capacity (dim_mults (1,2)→(1,4,8)) and training budget (8.5k→500k steps). Possible outcomes:

| Outcome | Chunk L2 | OPE Error | Diagnosis | Next Step |
|---------|----------|-----------|-----------|-----------|
| **A1: Full success** | < 1.0 | < 30% | Pipeline works | Scale to multi-policy |
| **A2: Good chunks, bad OPE** | < 1.0 | > 50% | Stitching or reward problem | Phase B |
| **A3: Mediocre chunks** | 1–100 | > 50% | Model learning but slowly | Phase C |
| **A4: Still garbage** | > 100 | > 100% | Fundamental issue | Phase D |

**Phase B: Fix stitching (if A2)**

If individual chunks are good but full trajectories blow up, the stitching loop is compounding errors. Run these experiments in order:

- **B1: Add 1-state overlap.** Modify `generate_full_trajectory()` to advance by `chunk_horizon - 1` instead of `chunk_horizon`. The overlapping state provides a continuity constraint. This is a one-line change and matches SOPE. **Expected: reduces stitch-boundary discontinuities.**

- **B2: Clamp conditioning states.** During stitching, clamp the conditioning state (extracted from the previous chunk's endpoint) to the training data range before feeding it to the next chunk. This prevents out-of-distribution drift. Crude but diagnostic — if it helps dramatically, the issue is conditioning drift.

- **B3: Per-dimension diagnostic.** Track the normalized conditioning state at each stitch point. Plot its z-scores over time. If any dimension exceeds ±5σ by chunk 3-4, that dimension is drifting and dragging others with it. Identify which dimensions drift first — those are the ones the model reconstructs worst.

**Phase C: Close the remaining SOPE gaps (if A3)**

If chunks are improving but not good enough, systematically match SOPE's remaining hyperparameters. Run each as a separate SLURM job — they're independent and can run in parallel on different GPUs.

- **C1: Switch to predict_epsilon=False (predict x₀).** Matches SOPE. Single config change. Retrain 500k steps. **Rationale:** x₀ prediction may produce sharper samples for this low-dim space. If chunk L2 drops significantly, this was the issue.

- **C2: Increase diffusion steps to 1000.** Matches SOPE. Doesn't affect training time, only sampling time (4× slower). Retrain 500k steps. **Rationale:** 256 steps may be too coarse for the noise schedule, causing the model to make imprecise denoising jumps.

- **C3: Match dim_mults exactly — (1,2,4,8) instead of (1,4,8).** Requires total_horizon divisible by 8 → change to chunk_size=7, frame_stack=1, total_horizon=8 (already satisfies this). Adds a 64-channel intermediate level. Retrain 500k steps. **Rationale:** SOPE's 4-level UNet has a deeper bottleneck that may help with temporal coherence.

- **C4: Increase LR to 1e-3** (matching SOPE) and/or reduce batch_size to 32. **Rationale:** Our 3e-4 LR with cosine annealing means the second half of training has negligible learning. Higher peak LR may help the model converge faster on this small dataset.

After running C1-C4, compare chunk L2 and OPE error. Pick the best config as the new baseline.

**Phase D: Simplify the problem (if A4)**

If 500k steps with the larger model still produces garbage, there's something fundamentally wrong. Strip the problem down:

- **D1: Drop quaternions — train on positions only.** Reduce the state space from 19-dim to 9-dim: cube_pos(3), eef_pos(3), gripper_qpos(2), plus maybe gripper_to_cube_z(1). The reward function only needs cube_z (index 2), so quaternions are irrelevant for OPE on Lift. transition_dim drops from 26 to 16. **This is the single most likely fix if the model fundamentally can't learn the distribution.** Quaternions have unit-norm constraints and antipodal symmetry that L2 diffusion handles badly. Removing them makes the problem strictly easier.

  If this works (chunk L2 < 1.0): quaternions were the problem. Consider 6D rotation representation or per-dimension loss weighting before adding them back.
  If this still fails: the problem isn't quaternions — move to D2.

- **D2: Overfit on 1 trajectory.** Take a single rollout (60 steps, ~26 chunks) and train the diffusion model until it memorizes it perfectly (chunk L2 → 0). Use a very high learning rate and no scheduler. **This is a sanity check** — if the model can't memorize 26 chunks, something is wrong with the model architecture, loss function, or conditioning mechanism. Should take <5 minutes on GPU.

  If it can memorize: the model works, problem is generalization/data diversity. Collect 200+ rollouts.
  If it can't memorize: bug in the model, loss, or conditioning. Debug there.

- **D3: Reproduce SOPE on D4RL.** Run the original SOPE code on a simple D4RL task (e.g., Hopper-medium-v2) and verify it works. Then run *our* pipeline on the same D4RL data (adapting the data loader). This isolates whether the problem is our code vs our data. If SOPE works on D4RL but our code doesn't, there's a bug in our diffusion/stitching implementation.

**Phase E: Scale data (after any success)**

Once any configuration produces chunk L2 < 1.0 and OPE error < 50%:

- **E1: 200 rollouts** (~5000 chunks). Each chunk seen ~6400× during 500k steps. Still more oversampled than SOPE (128×), but 4× more diverse than current.
- **E2: Parallel collection.** Use `collect_rollouts(num_workers=4)` to cut collection time from ~50min to ~15min.
- **E3: 100 oracle rollouts** for tighter ground truth (±3.5% std error vs current ±7%).

**Decision tree summary:**

```
500k run completes
├─ Chunk L2 < 1.0, OPE < 30% → SUCCESS → Phase E (scale)
├─ Chunk L2 < 1.0, OPE > 50% → B1 (overlap) → B2 (clamp) → B3 (diagnose)
├─ Chunk L2 1-100 → C1-C4 in parallel (match SOPE hyperparams)
└─ Chunk L2 > 100
   ├─ D1: Drop quaternions (most likely fix)
   ├─ D2: Overfit sanity check
   └─ D3: Reproduce SOPE on D4RL
```

### 3.2 Chunk Size and Frame Stack

**Evolution:**
- **v1 (8.5k steps):** chunk_size=8, frame_stack=2 → total_horizon=10
- **v2 (500k steps):** chunk_size=7, frame_stack=1 → total_horizon=8

**Why the change:** To match SOPE exactly. SOPE conditions on 1 state at index 0 (frame_stack=1), with total horizon T=8. Our original frame_stack=2 meant conditioning on 2 past states, which is subtly different — it gives the model more context but changes the chunk boundary semantics.

**Assessment: The change was correct but the reasoning is worth examining.**
- **Pro (frame_stack=1):** Matches SOPE. Simpler conditioning — only one state pinned. Fewer edge cases at trajectory start (no need to duplicate initial state twice).
- **Con (frame_stack=1):** Less temporal context for denoising. With frame_stack=2, the model sees velocity information implicitly (two consecutive states). With frame_stack=1, it only sees position at a single instant. Whether this matters depends on the dynamics.
- **Pro (total_horizon=8):** Divisible by 4, which is required for dim_mults=(1,4,8) with 3 UNet levels (needs divisibility by 2^(len(dim_mults)-1) = 4). The original total_horizon=10 with dim_mults=(1,2) only needed divisibility by 2.
- **The chunk size itself (7-8 steps)** covers about 12% of a 60-step trajectory. This means ~8 stitching iterations to generate a full trajectory. Each stitch point is a potential error injection site. Larger chunks = fewer stitches but harder to learn; smaller chunks = easier to learn but more compounding error. 7-8 seems reasonable for a 60-step horizon.

### 3.3 TemporalUnet Architecture

**Evolution:**
- **v1:** dim_mults=(1,2) → 2 resolution levels, ~252k parameters
- **v2:** dim_mults=(1,4,8) → 3 resolution levels, ~2-5M parameters

**Why the change:** v1 was catastrophically underparameterized. The model couldn't learn a 26-dimensional temporal distribution with only 252k parameters. The 8.5k-step analysis confirmed this: loss was still decreasing at 0.728 after 500 epochs, and chunk L2 was 7287 (target <1.0).

**Assessment: Necessary fix, but still potentially different from SOPE.**
- **SOPE default is (1,2,4,8)** — 4 resolution levels. We use (1,4,8) — 3 levels. This means SOPE has one more downsampling stage, creating a deeper bottleneck. The difference is likely small (both create multi-million parameter models), but it's a deviation.
- **No attention (attention=False):** Matches SOPE default. Attention would help with long-range temporal dependencies within a chunk, but chunks are only 8 steps — probably not long enough to benefit. Adding attention would also significantly increase compute per step.
- **Base dim = 32:** Creates channels [32, 128, 256] with (1,4,8). SOPE with (1,2,4,8) creates [32, 64, 128, 256]. Our widest layer matches, but we skip the 64-channel level entirely.
- **Kernel size = 5, groups = 8:** Inherited from SOPE's TemporalUnet. The kernel covers 5 timesteps in a single convolution — more than half the 8-step chunk. GroupNorm with 8 groups is standard.
- **Time embedding:** Sinusoidal positional encoding (dim=32) → 2-layer MLP (32 → 128 → 32) with Mish activation. Injected into each ResidualTemporalBlock. Inherited directly from SOPE.

### 3.4 Diffusion Process

**256 diffusion steps (vs SOPE's 1000).**

**Why:** Faster sampling. 256 steps means each chunk sample takes ~256 forward passes through the UNet. With 1000 steps, it would be 4× slower. For a 60-step trajectory needing ~8 chunks × 50 trajectories = 400 chunk samples, this is the difference between ~102k and ~400k forward passes.

**Assessment: Risky tradeoff.**
- **Con:** Fewer diffusion steps = coarser noise schedule = harder denoising. The model has to make larger jumps at each step, which can reduce sample quality. SOPE chose 1000 for a reason — D4RL tasks have more complex dynamics.
- **Pro:** Lift is a simple task. 256 steps may be sufficient for a 26-dim transition space. The cosine schedule is relatively forgiving of step count.
- **Unknown:** We haven't ablated this. If the 500k-step run still produces poor samples, this is a candidate to change. Moving to 1000 steps would 4× slow sampling but might improve quality.
- **Note:** This only affects **sampling** speed. Training always samples a single random timestep per batch element, so diffusion_steps doesn't affect training speed.

**Predict epsilon (True) vs SOPE's predict x₀ (False):**

This is a significant divergence. SOPE predicts x₀ directly (the clean sample), while we predict ε (the noise). Both are mathematically equivalent — you can recover one from the other — but they have different optimization landscapes:
- **ε-prediction:** Uniform-difficulty across timesteps. The model always predicts unit-variance noise regardless of t. Standard in DDPM literature.
- **x₀-prediction:** Harder at high noise (large t), easier at low noise. Can produce sharper samples but training can be less stable.

**Assessment: Neither is clearly better, but using the same mode as SOPE would eliminate one variable.**

### 3.5 Loss Function

**L2 loss with action_weight=5.0, loss_discount=1.0.**

Matches SOPE exactly. The action weight upweights action reconstruction 5× relative to states. This makes sense: actions are what the policy *does*, and OPE ultimately cares about whether the generated behavior matches the policy. States are partially constrained by physics — wrong actions automatically produce wrong states, but the converse isn't as true.

**Assessment: Reasonable, matches reference.**
- loss_discount=1.0 means uniform temporal weighting. No exponential decay across the chunk. This treats all timesteps in a chunk equally, which is fine for short (8-step) chunks. For longer chunks, you might want to discount later steps.

### 3.6 Training Configuration

**500k steps = 100 epochs × 5000 steps/epoch with replacement.**

**Evolution from v1 (8.5k steps):**
- v1 did 500 epochs × 17 batches/epoch = 8,500 steps. The 17 batches were one full pass through ~1,100 chunks.
- v2 samples with replacement: each epoch draws 5,000 random batches from the dataloader, cycling infinitely. This matches SOPE's approach.

**Assessment: The with-replacement sampling was essential.**
- v1's problem was that 17 batches/epoch meant each epoch was a single pass. Even with 500 epochs, the model only saw 8,500 gradient updates. SOPE does 500,000.
- With-replacement means each epoch is 5000 independently sampled batches. The model sees each chunk ~24,600 times over training (500k × 64 / 1300 chunks). This is heavy oversampling, but SOPE does similar (500k × 32 / 125k chunks ≈ 128 times on D4RL — our 24,600× is much higher due to fewer chunks).

**Data diversity concern:**
- 50 rollouts → ~1,300 chunks. SOPE uses D4RL datasets with ~125,000 chunks (100× more diverse).
- Each chunk is seen ~24,600 times during training. This creates serious overfitting risk — the model may memorize chunks perfectly (low chunk L2) but fail to generalize at stitch boundaries where it encounters states slightly different from training.
- **This is probably the biggest risk in the current pipeline.** Even if chunk L2 drops below 1.0, stitching may fail because the conditioning states (from the end of the previous chunk) don't exactly match any training chunk's conditioning states.
- Mitigation: collect more rollouts (200+), or add data augmentation (noise injection, temporal jittering).

**Learning rate (3e-4 vs SOPE's 1e-3):**
- We use a lower LR. Combined with CosineAnnealingLR (T_max=200 = 2×epochs), the LR decays from 3e-4 to near 0 over 200 epochs, meaning the second half of training has very small updates.
- SOPE also uses cosine annealing but with a higher peak LR. The lower LR is more conservative — less risk of overshooting but potentially slower convergence.

**Batch size (64 vs SOPE's 32):**
- Larger batches = more stable gradients but effectively halves the number of unique gradient directions per step. With only 1,300 chunks, batch_size=64 means each batch samples ~5% of the dataset.

### 3.7 Normalization

**Per-feature mean/std computed at super-trajectory level (across all rollout files).**

Formula: `z_norm = (z - mean) / (std + 1e-6)` per feature dimension (26 dims: 19 state + 7 action).

**Assessment: Standard and correct, but has subtleties.**
- **Super-trajectory level** means stats are computed once across all chunks from all 50 rollouts, not per-rollout. This is correct — you want a single normalization for the entire dataset.
- **Applied during training:** Chunks are normalized before being passed to the diffusion model.
- **Applied during stitching:** Conditioning states are normalized; samples are internally normalized; outputs are unnormalized before storage.
- **ε=1e-6 for std:** Prevents division by zero for near-constant features (e.g., gripper_qpos may have very low variance if the gripper barely moves).
- **Potential issue:** The normalization is feature-wise, not sample-wise. Features with multi-modal distributions (e.g., cube_z which is either ~0.82 resting or ~0.88 lifted) will have their modes compressed toward 0 but won't be truly normalized for a Gaussian diffusion model.

### 3.8 Stitching Loop

**Our implementation vs SOPE:**

| Aspect | Ours | SOPE |
|--------|------|------|
| Overlap | 0 states (no overlap) | 1 state overlap |
| Conditioning space | Normalized | Normalized |
| Output space | Unnormalized | Unnormalized |
| Early termination | No | Optional (is_terminated_fn) |
| Tanh action squashing | No | Optional |
| Batch pruning | No | Yes (drop terminated trajs) |

**Key difference — overlap:** SOPE advances by `T-1` steps per chunk (overlapping 1 state), while our implementation advances by `chunk_horizon` steps with no overlap. This means:
- **Ours:** Chunk k generates states [t, t+7]. Chunk k+1 conditions on state t+6 (or t+7) and generates states [t+7, t+14]. No shared states between chunks.
- **SOPE:** Chunk k generates states [t, t+7]. Chunk k+1 conditions on state t+7 and generates states [t+7, t+14]. State t+7 appears in both chunks, providing a smoother transition.

**Assessment: The no-overlap approach may cause discontinuities at stitch boundaries.** The last generated state of chunk k feeds directly as conditioning for chunk k+1, but there's no guarantee the diffusion model generates a continuation that's dynamically consistent. With overlap, at least one state is shared, providing a stronger continuity constraint.

**Conditioning is always in normalized space, extracted from the still-normalized diffusion sample.** This avoids normalize→unnormalize→renormalize round-trip error. Good design — matches SOPE.

### 3.9 Reward Model: Ground-Truth Analytical

**Decision: Use analytical reward (cube_z > 0.84 → reward 1.0) instead of learned MLP.**

```
success = cube_z > table_height + height_threshold
        = cube_z > 0.8 + 0.04
        = cube_z > 0.84
```

**Assessment: Excellent choice for Lift, wouldn't generalize.**
- **Pro:** Zero approximation error. The reward function is simple and fully known. A learned MLP would introduce unnecessary noise.
- **Pro:** The analytical reward directly reads from the latent vector (index 2 = cube z-position), since our latent space IS the observation space (see §3.1). No decoding or simulation needed.
- **Con:** This only works because (a) Lift has a trivially simple reward, and (b) we use low-dim concat encoding so the cube position is directly readable. For tasks with complex rewards (contact forces, velocity penalties, visual success criteria), a learned model would be necessary.
- **SOPE comparison:** SOPE uses learned `RewardEnsembleEstimator` (MLP [64,64,1] with bootstrap sampling, 1000 iterations). This adds reward approximation error but generalizes to any task.

### 3.10 Oracle Estimation

**50 rollouts with gamma=1.0 (undiscounted).**

**Assessment: Potentially noisy.**
- SOPE uses 1000 oracle rollouts. Our 50 gives a noisier estimate. For Lift (binary success), the oracle is basically the success rate × max_return. With 50 rollouts, the standard error of the success rate is ~√(p(1-p)/50) ≈ 7% for p=0.54. This means the oracle itself has ±7% uncertainty.
- For comparing against OPE estimates with potentially 5000% error (as in v1), oracle noise is negligible. But for a converged pipeline where we hope for <10% OPE error, oracle uncertainty matters.

---

## 4. Data Flow: End-to-End Dimension Tracking

```
Step 0: Oracle
  └─ Run policy 50 times (horizon=60) → mean return = 0.54

Step 1: Collect Rollouts
  └─ 50 rollouts × 60 steps
  └─ Per rollout .h5 file:
       latents: (60, frame_stack, 19)   # frame_stack=1 or 2
       actions: (60, 7)
       rewards: (60,)

Step 2: Chunk
  └─ chunk_size=7, stride=2, frame_stack=1
  └─ Per rollout: ~26 chunks
  └─ Total: ~1300 chunks
  └─ Per chunk:
       states_from:  (1, 19)    # conditioning
       actions_from: (1, 7)     # conditioning
       states_to:    (8, 19)    # chunk_size + 1 target states
       actions_to:   (7, 7)     # chunk_size target actions

Step 3: Train
  └─ Batch input to diffusion:
       x: (B=64, T=8, D=26)    # concatenated [state|action]
       cond: {0: (B, 19)}      # pinned initial state
  └─ UNet: (B, 8, 26) → (B, 8, 26)
  └─ 500k gradient steps, CosineAnnealingLR

Step 5: Stitch
  └─ Initial states: (50, 19) from real rollouts
  └─ Generate 8 chunks per trajectory:
       Sample (50, 8, 26) → extract states (50, 7, 19) + actions (50, 7, 7)
       Condition next chunk on last state
  └─ Output: states (50, 60, 19), actions (50, 60, 7)

Step 6: Score
  └─ Decode latents → obs_dict → apply reward_fn
  └─ Per trajectory: return = Σ rewards (gamma=1.0)

Step 7: OPE
  └─ ope_estimate = mean(returns)
  └─ Compare to oracle_value = 0.54
```

---

## 5. What's Missing vs SOPE

### 5.1 Policy Guidance (Step 4 — skipped)

The biggest missing piece. SOPE uses gradient-based guidance during the diffusion sampling process:
- Compute ∇_a log π(a|s) from the target policy
- Optionally compute ∇_a log β(a|s) from the behavior policy (for negative guidance)
- Combined gradient: `guide = ∇ log π - ratio × ∇ log β`
- Added to denoised mean at each diffusion step, scaled by `action_scale`
- Adaptive scaling via cosine schedule multiplier
- k=2 guidance iterations per diffusion step

Without guidance, the diffusion model generates "average" trajectory chunks from the offline data distribution — not chunks specifically consistent with the target policy. For on-policy data (which is what we collect), this distinction is small. But guidance would help the model generate more policy-consistent continuations at stitch boundaries.

**Why skipped:** Robomimic policies don't expose `grad_log_prob()`. Each policy type needs a custom wrapper:
- BC_Gaussian: extract mean/std → analytic Gaussian log-prob
- BC_GMM: log-sum-exp over mixture components
- DiffusionPolicyUNet: use diffusion score

### 5.2 Stitch Overlap

SOPE advances T-1 steps per chunk (1 state overlap). We advance by the full chunk_horizon (0 overlap). The overlap provides a continuity constraint that could reduce stitching artifacts.

### 5.3 Termination Handling

SOPE checks `is_terminated_fn` at each step during stitching and prunes terminated trajectories. We generate fixed-length trajectories with no early stopping. For Lift (where episodes end at horizon regardless of success), this doesn't matter much. For tasks with variable-length episodes, it would.

### 5.4 Action Squashing

SOPE optionally applies tanh to bound actions to [-1, 1]. We don't. Robomimic policies already output bounded actions (the env clips them), but the diffusion model generates unbounded values.

### 5.5 Training Details

- **EMA (Exponential Moving Average):** SOPE mentions EMA model tracking. We don't use EMA, which could help with training stability and sample quality.
- **Diffusion steps (256 vs 1000):** Lower fidelity noise schedule.
- **Predict ε vs x₀:** Different prediction targets with subtly different optimization landscapes.

### 5.6 Data Scale

| Metric | Ours | SOPE (D4RL) |
|--------|------|-------------|
| Trajectories | 50 | ~1000 |
| Chunks | ~1,300 | ~125,000 |
| Chunk presentations during training | ~24,600× each | ~128× each |
| Oracle rollouts | 50 | 1000 |

We have 100× less data diversity. The model sees each chunk 200× more often. This is the biggest concern for generalization.

---

## 6. Latent Extraction — Current State and Future Vision

### 6.1 The Vision: Visual Embeddings from Robomimic

The project is called "Latent SOPE" because the long-term goal is to run diffusion over **learned visual embeddings** from robomimic's policy encoder — not raw observations. The idea is:

1. Robomimic policies (especially DiffusionPolicyUNet) have internal observation encoders that compress high-dimensional inputs (images, point clouds) into compact latent vectors
2. These latents capture task-relevant features the policy has learned
3. Running SOPE-style chunk diffusion in this latent space would let us do OPE even for image-based policies, where the raw observation space is too high-dimensional for diffusion

The `HighDimObsEncoder` class in `encoders.py` is designed for exactly this: it registers a PyTorch forward hook on a specified module inside the policy (e.g., `"policy.nets.policy.obs_encoder"`) and captures the intermediate activations as the latent vector.

### 6.2 Current Reality: Visual Embedding Path Is Stubbed Out

**The `high_dim_encode` path does not work yet.** Specifically:

- `PolicyFeatureHook.update_latent_from_obs()` (rollout.py:142-145) has an early return for non-`low_dim_concat` feat types, with a comment: *"Warning: might actually need to implement the corresponding logic for the high_dim_encode feat_type as well - need to find out"*
- All experiments to date use `feat_type="low_dim_concat"`
- The `HighDimObsEncoder` class exists and can register hooks, but it's never been tested end-to-end in the rollout pipeline

**What's currently used is `LowDimConcatEncoder`**, which does no learned encoding at all. It just concatenates sorted low-dim observation keys into a flat vector:

```
latent[t] = concat(obs["object"][t], obs["robot0_eef_pos"][t], obs["robot0_eef_quat"][t], obs["robot0_gripper_qpos"][t])
         = (19,) vector of raw physical quantities
```

No neural network. No learned features. No compression. The "latent" is a renamed observation vector.

### 6.3 How This Compares to SOPE

**SOPE also uses raw low-dim observations**, not learned latents. It reads `dataset['observations']` from D4RL (joint angles, velocities, positions) and concatenates them with actions. SOPE's `Data` class has an optional `state_enc` parameter but the reference experiments always use identity (no encoding).

So **for the current Lift experiment, our state representation is functionally identical to SOPE's.** The key difference is aspirational: once `high_dim_encode` works, we'll be able to do something SOPE cannot — OPE in learned latent spaces for image-based policies.

### 6.4 What It Would Take to Enable Visual Embeddings

To switch from `low_dim_concat` to `high_dim_encode`:

1. **Fix `update_latent_from_obs()` stub** in rollout.py (lines 142-145) — implement the HighDimObsEncoder forward pass path
2. **Identify the right module path** for the DiffusionPolicyUNet encoder. Candidates listed in `PolicyFeatureHook.__init__()`: `"policy.nets.policy.obs_encoder"`, `"nets.policy"`, `"policy"`
3. **Determine latent dimensionality** — the encoder output dim depends on the policy architecture. Need to inspect the model to find out
4. **Build a reward decoder** — with visual embeddings, cube_z is no longer at a known index. Either:
   - Train a small MLP to predict reward-relevant quantities from latents, or
   - Use the learned `RewardMLP` instead of analytical `LiftRewardFn`
5. **Verify the latents are meaningful** — run sanity checks: are nearby states in latent space actually similar trajectories? Do the latents cluster by success/failure?
6. **Retrain the diffusion model** on new latent dimensions

### 6.5 Shape Flow Through the Current Pipeline

Tracing the exact shape at every stage (with LowDimConcatEncoder):

| Stage | Shape | Notes |
|-------|-------|-------|
| Raw obs from env | `obs["object"]: (10,)`, `obs["robot0_eef_pos"]: (3,)`, etc. | Per-timestep obs dict |
| LowDimConcatEncoder output | `(T, 19)` | Pure concatenation, no transformation |
| PolicyFeatureHook per step | `(19,)` | Cached from encoder forward |
| RolloutLatentRecorder accumulation | List of `(19,)` arrays | One per timestep |
| Saved to H5 | `latents: (T=60, 19)`, `frame_stack` as attribute | frame_stack is metadata, NOT a dimension |
| Loaded from H5 | `latents: (T=60, 19)` | Same as saved |
| `_preprocess_latents()` | `(T, 19)` → `(T, 19)` | Has a `z[:, 0, :]` branch for 3D input, but it's **unreachable** with LowDimConcatEncoder since latents are always 2D |
| `_collect_frame_stack()` | `(T, 19)` → `(frame_stack, 19)` | Looks back in time, creates frame_stack window |
| Dataset output per chunk | `states_from: (1, 19)`, `states_to: (8, 19)`, etc. | frame_stack dim added here by temporal lookback |
| Batch to diffusion | `x: (B, 8, 26)` | states+actions concatenated along feature dim |

**The frame_stack dimension is synthetic** — it's created by `_collect_frame_stack()` looking back in time over the flat `(T, 19)` array, not by reading a pre-existing frame_stack dimension from the latents.

### 6.6 Debugging: Is the Latent Extraction Causing the Blowup?

**Short answer: almost certainly not**, since the current "latents" are just raw observations — the same thing SOPE uses successfully. The blowup must come from downstream: the diffusion model, normalization, or stitching. But here's how to verify and where to look:

#### Diagnostic 1: Verify Latents Are Correct at Source

```python
# Load a saved rollout and check latent values
traj = load_rollout_latents("rollout_latents_50/rollout_0000.h5")
print(f"Latents shape: {traj.latents.shape}")  # Should be (60, 19)
print(f"Latents range: [{traj.latents.min():.4f}, {traj.latents.max():.4f}]")
# Should be in physical range: positions ~[0, 1.1], quaternions [-1, 1], gripper ~[0, 0.05]

# Verify specific dims match known physics
cube_z = traj.latents[:, 2]
print(f"Cube z: [{cube_z.min():.4f}, {cube_z.max():.4f}]")  # Should be ~[0.81, 0.88]
eef_z = traj.latents[:, 12]
print(f"EEF z: [{eef_z.min():.4f}, {eef_z.max():.4f}]")  # Should be ~[0.8, 1.1]
```

If these look correct, the latent extraction is fine. Move to Diagnostic 2.

#### Diagnostic 2: Verify Normalization Round-Trip

Already done in the 8.5k-step analysis: max error was 2.38e-07. **Normalization is not the problem.**

#### Diagnostic 3: Single-Chunk Denoising (No Stitching)

This isolates the diffusion model from the stitching loop:

```python
# Take a real chunk from the dataset
batch = next(iter(dataloader))
batch_dev = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

# Get ground truth
x_gt = diffuser._build_x(batch_dev)  # (B, 8, 26)

# Denoise from the same conditioning
cond = diffuser.make_cond(batch_dev)
sample = diffuser.diffusion.conditional_sample(shape=x_gt.shape, cond=cond, guided=False)

# Compare in UNNORMALIZED space
x_gt_unnorm = diffuser.unnormalizer(x_gt).cpu().numpy()
x_hat_unnorm = diffuser.unnormalizer(sample.trajectories).cpu().numpy()

# Check: are individual denoised chunks already blowing up?
print(f"GT range: [{x_gt_unnorm.min():.2f}, {x_gt_unnorm.max():.2f}]")
print(f"Sample range: [{x_hat_unnorm.min():.2f}, {x_hat_unnorm.max():.2f}]")
```

**If samples blow up here → the diffusion model hasn't learned the distribution.** This was the case at 8.5k steps (chunk L2 = 7287). The model generates noise, and unnormalization amplifies it to extreme values.

**If samples are reasonable here but blow up during stitching → the stitching loop is compounding errors.** Check the conditioning states being passed between chunks — are they drifting out of the normalized training distribution?

#### Diagnostic 4: Stitching Drift Analysis

```python
# Generate a trajectory and track conditioning states at each stitch point
# Are the conditioning states (in normalized space) staying near the training distribution?
# If they drift to values far from [-3, 3] in normalized space, the model is receiving
# out-of-distribution input and generating garbage.
```

#### Diagnostic 5: Per-Dimension Blowup Correlation with Variance

```python
# Check if dimensions with highest blowup correlate with lowest training variance
# If a dimension has tiny std (e.g., gripper_qpos ≈ 0.001 std), then even small
# diffusion errors get amplified 1000× during unnormalization
for d in range(19):
    print(f"Dim {d}: std={norm_stats.std[d]:.6f}, blowup_factor={...}")
```

Small training variance → large unnormalization amplification. This could explain why quaternions blow up: they have small variance in a successful Lift policy (the cube barely rotates), so `std` is tiny, and `x_unnorm = x_norm * std + mean` doesn't amplify much — BUT if the diffusion model generates `x_norm = ±50` instead of ±1, then even multiplying by a small std produces large values. The real question is whether the model generates normalized values far from the [-3, 3] range.

### 6.5 The Real Culprits (Ranked by Likelihood)

1. **Undertrained diffusion model.** At 8.5k steps, the model generates near-random noise in normalized space. Unnormalization converts random noise to large values. The 500k-step run should fix this if the model converges.

2. **Low data diversity causing poor generalization at stitch boundaries.** The model memorizes training chunks but can't handle the slightly-different conditioning states that arise during autoregressive stitching. Each stitch point feeds a conditioning state the model hasn't seen, producing a bad chunk, whose endpoint is even more out-of-distribution, causing cascading failure.

3. **Quaternion dimensions** have correlated components (unit norm constraint) that the independent-dimension normalization doesn't capture. The diffusion model treats each quaternion component independently, which can produce non-unit quaternions that, while not physically meaningful, aren't actually the cause of the blowup (they're a symptom of #1).

4. **No stitching overlap** (0 shared states between chunks) means there's no continuity enforcement at boundaries. Adding 1-state overlap would provide a soft constraint.

---

## 7. Critical Assessment Summary

### What's working well
1. **Pipeline structure is correct.** Steps 0-3, 5-7 are fully implemented and tested end-to-end. The data flows cleanly from rollouts to OPE estimate.
2. **Ground-truth reward** eliminates one error source. Smart for a simple task.
3. **Low-dim concat encoding** is transparent and lossless for Lift.
4. **Normalized-space conditioning** during stitching avoids round-trip error.
5. **Config cross-validation** catches dimension mismatches early.

### What's concerning
1. **Data diversity (50 rollouts, ~1300 chunks)** is 100× less than SOPE's D4RL datasets. High overfitting risk.
2. **Quaternion representation** in the latent space is geometrically problematic for diffusion. The 8.5k-step blowup was worst in quaternion dimensions.
3. **No policy guidance** means stitched trajectories aren't steered toward policy-consistent behavior.
4. **256 diffusion steps** (vs 1000) may reduce sample quality.
5. **predict_epsilon=True** (vs SOPE's False) is a divergence that hasn't been ablated.
6. **No stitching overlap** may cause boundary discontinuities.
7. **No EMA** may reduce sample quality at inference time.

### Priority ranking for fixes (if 500k run still fails)
1. Collect more rollouts (200+) — addresses data diversity
2. Add 1-state stitching overlap — easy fix, may help boundaries
3. Switch to predict_epsilon=False — match SOPE
4. Increase diffusion steps to 1000 — match SOPE
5. Add EMA — standard practice for diffusion
6. Drop quaternion dims or switch to 6D rotation — if quats still blow up
7. Add policy guidance — hardest, most impactful long-term

---

## 7. Hyperparameter History

| Parameter | v1 (8.5k steps) | v2 (500k steps) | Change Reason |
|-----------|-----------------|-----------------|---------------|
| dim_mults | (1, 2) | (1, 4, 8) | Model too small (252k → ~2-5M params) |
| chunk_size | 8 | 7 | Match SOPE total_horizon=8 with frame_stack=1 |
| frame_stack | 2 | 1 | Match SOPE conditioning (1 state at index 0) |
| total_horizon | 10 | 8 | Follows from above two changes |
| epochs | 500 | 100 | Budget redistributed to steps/epoch |
| steps/epoch | 17 (full pass) | 5000 (with replacement) | Match SOPE data throughput |
| total steps | 8,500 | 500,000 | 60× increase to match SOPE budget |
| Chunk L2 (states) | 7,287 | TBD (running) | — |
| OPE relative error | 5,333% | TBD (running) | — |
| % values clamped | 99.7% | TBD (running) | — |
