# SOPE on Robomimic: Setup Comparison

**Date:** 2026-03-10

---

## What SOPE Does (D4RL)

SOPE operates on **large pre-collected offline datasets** from D4RL — typically **~1 million timesteps** across many trajectories. These datasets contain data from various behavior policies (not just the target policy being evaluated).

| D4RL Task | State Dim | Action Dim | Dataset Size | Chunks (T=8, stride=1) |
|-----------|-----------|------------|-------------|----------------------|
| Hopper | 11 | 3 | ~1M steps | ~125,000 |
| HalfCheetah | 17 | 6 | ~1M steps | ~250,000 |

The key idea: SOPE trains a diffusion model on this **large, diverse, off-policy dataset** to learn what trajectory chunks look like. Then it uses **policy guidance** during sampling to steer the generated chunks toward the target policy's behavior. This is how it estimates V^π without running π online.

## How Our Setup Differs

We collect **on-policy data** (rolling out the target policy itself) and train the diffusion model on that small dataset:

| | SOPE (D4RL) | Ours (Robomimic Lift) |
|--|-------------|----------------------|
| Data source | Pre-collected, multi-policy | Single target policy rollouts |
| Trajectories | ~1000+ | 50 |
| Chunks | ~125,000 | ~1,300 |
| Data diversity | High (different policies) | Low (one policy) |
| Policy guidance | Yes (steers toward π) | Skipped |
| Trajectory horizon | 768–1000 steps | 60 steps |

## The Fundamental Tension

Because our data is **on-policy**, we don't strictly need guidance — the chunks already reflect the target policy. But we have **100x fewer chunks**, so the diffusion model has far less to learn from. Each chunk gets seen ~24,600 times during 500k training steps (vs ~128 times in SOPE). That's extreme oversampling.

However, the D2 overfit experiment showed the problem is deeper than data scale — the model can't even memorize 17 chunks. So the immediate blocker is at the model/code level, not the dataset size (see D2.1 for the full diagnosis).

## Summary

SOPE is intended to run on large offline datasets with diverse behavioral coverage. Our adaptation to small on-policy robomimic rollouts is a valid research direction, but the data regime is very different from what SOPE was designed for.

---

## What Needs to Change for an MVP

Given what we've learned today (8.5k baseline, D1 no-quaternions, D2 overfit, D2.1 bug analysis, architecture audit), here's what must change to get a working MVP of Stitch-OPE on Robomimic Lift — ordered from blocking bugs to design improvements.

### Tier 1: Fix Bugs (Blocking)

**1. Fix the training/sampling shape mismatch.**
`sample()` generates sequences of length `chunk_horizon + frame_stack - 1`, but training uses sequences of length `frame_stack + chunk_size` (one timestep longer). The model literally trains on a different sequence length than it samples. This is the most likely cause of the memorisation failure in D2. Fix: change the sample shape to `total_chunk_horizon`. (See D2.1 E1 for details.)

### Tier 2: Match SOPE Where It Matters (High Priority)

**2. Drop quaternions from the state space.**
D1 showed a 14.5x improvement in chunk L2 just from removing quaternions (19→11 state dims). Quaternions have unit-norm constraints and antipodal symmetry that L2 diffusion handles badly, and they're irrelevant for Lift's reward (which only checks cube z-position). Also drop the 3 orientation action dims (indices 3–5) that depend on the removed quaternion state, reducing to a 4-dim action space. This makes state-action space fully consistent and brings transition_dim from 26 down to 15 — close to SOPE's Hopper (14).

**3. Increase model capacity to match SOPE.**
Our dim_mults=(1,2) gives ~250k params. SOPE's Hopper uses (1,2,4,8), Cheetah uses (1,4,8). The model needs enough capacity to learn the denoising function across 256 noise levels, not just enough to store the raw data. Switch to dim_mults=(1,4,8) at minimum.

**4. Verify predict_epsilon setting works on the overfit test.**
Corrected analysis shows SOPE's Hopper uses `predict_epsilon=True` (same as us), so this isn't necessarily wrong. But it's cheap to ablate on the 1-trajectory overfit — if False helps memorisation, use it.

### Tier 3: Design Simplifications (Medium Priority)

**5. Simplify to frame_stack=1.**
SOPE conditions on 1 state at index 0. Our frame_stack=2 means we condition on 2 states, which adds complexity without clear benefit for an 8-step chunk. frame_stack=1 also avoids edge cases at trajectory start. The 500k run already made this change — carry it forward.

**6. Keep the on-policy, no-guidance setup for MVP.**
Our on-policy data collection means the chunks already represent the target policy's behavior. Guidance (Step 4) is the hardest piece to implement (requires per-policy-type `grad_log_prob` wrappers) and provides the least benefit when data is on-policy. Skip it for MVP. If the diffusion model learns good chunks and stitching works, unguided OPE should already give a reasonable estimate.

**7. Keep ground-truth reward.**
The analytical `LiftRewardFn` (cube_z > 0.84) has zero approximation error and works directly on the latent vector. No reason to use a learned reward model for Lift.

### Tier 4: Scale After Validation (After MVP Works)

**8. Scale data to 200 rollouts once the overfit test passes.**
50 rollouts (~1,300 chunks) is still 100x less than SOPE's D4RL data. Each chunk gets seen ~24,600 times — serious overfitting risk. Once the model can memorise a single trajectory, scale to 200 rollouts (~5,000 chunks) to test generalisation. If chunk L2 is good but stitching fails (OPE still bad), this is the most likely culprit.

**9. Add 1-state stitching overlap.**
SOPE advances by T-1 steps per chunk (1 shared state at the boundary). We advance by the full chunk_horizon (0 overlap). The overlap provides a continuity constraint that reduces drift at stitch boundaries. Easy change, defer until after the core diffusion works.

**10. Consider EMA for sample quality.**
SOPE uses EMA during training. Standard practice for diffusion models — the smoothed weights produce better samples at inference time. Low effort to add, but won't fix a broken model.

### What the MVP Looks Like

| Parameter | MVP Value | Rationale |
|-----------|-----------|-----------|
| State dim | 11 (no quaternions) | D1 showed 14.5x improvement |
| Action dim | 4 (no orientation deltas) | Consistent with reduced state |
| Transition dim | 15 | Close to SOPE Hopper (14) |
| dim_mults | (1, 4, 8) | Match SOPE capacity |
| frame_stack | 1 | Match SOPE, simpler |
| chunk_size | 7 | Total horizon 8, divisible by 4 |
| predict_epsilon | True (or False if E3 shows benefit) | Match SOPE Hopper default |
| Diffusion steps | 256 | Same as SOPE Hopper |
| Training steps | 500k | Match SOPE budget |
| Guidance | None | On-policy data, defer for MVP |
| Reward | Ground-truth analytical | Zero error for Lift |
| Sample shape bug | Fixed | Tier 1 blocker |

### Validation Strategy

1. **Overfit test (1 trajectory, 17 chunks):** chunk L2 < 1.0 → model architecture is sound
2. **50-rollout run:** chunk L2 < 10, OPE relative error < 100% → pipeline produces a signal
3. **200-rollout run:** OPE relative error < 30% → MVP success

---

## SOPE Reference Implementation: Full Analysis

Deep dive into `third_party/sope/` to understand every design choice.

### Diffusion Model Architecture (TemporalUnet)

A Conv1d-based U-Net for temporal (trajectory) data:

- **Time Embedding**: Sinusoidal position embeddings → MLP (`dim → dim*4 → dim`)
- **Residual Blocks**: `ResidualTemporalBlock` with two `Conv1dBlock` layers (Conv1d + GroupNorm + Mish), additive time conditioning, residual connection
- **U-Net Structure**: Encoder downsamples via strided Conv1d (stride=2), decoder upsamples via strided TransposeConv1d, skip connections concatenate encoder→decoder
- **Optional Attention**: `LinearAttention` at each level (disabled by default)
- **I/O**: Input `(B, T, state_dim + action_dim)` → rearranged to `(B, transition_dim, T)` for Conv1d → output same shape

Default config:
```python
TemporalUnet(
    horizon=T,
    transition_dim=state_dim + action_dim,
    dim=32,
    dim_mults=(1, 2, 4, 8),
    attention=False
)
```

### GaussianDiffusion: Training and Sampling

**Training (`p_losses`):**
- Cosine beta schedule: `α_cumprod = cos((t/T + 0.008) / 1.008 * π/2)²`, betas clipped to [0, 0.999]
- Forward process: `x_t = √α_t · x_0 + √(1-α_t) · ε`
- Model predicts epsilon (configurable via `predict_epsilon` flag)
- Loss: weighted MSE or L1
  - `loss_weights[t, d] = discount^t * dim_weight[d]`
  - **Action dims get 5x weight** (`action_weight=5`)
  - Discount defaults to 1.0 (uniform per timestep)
- Conditioning applied during training via `apply_conditioning()` (re-pins conditioned states)

**Sampling (`p_sample_loop`):**
- Reverse diffusion: iterate t from `n_timesteps-1 → 0`
- Each step: `x_{t-1} = μ(x_t, t) + σ(t) · ε`
- `μ` (model_mean) from `q_posterior()` given predicted x_0
- `σ` from posterior variance
- `apply_conditioning()` re-pins conditioned positions at each step
- Conditions dict: `{timestep_idx: normalized_state_tensor}` — replaces state dims only

### Policy Guidance

Two guidance functions, applied at each denoising step:

**`gradlog()` — For standard policies (BC_Gaussian, BC_GMM):**
```python
# Extract states (no grad) and actions (with grad)
state = x[:, :, :state_dim].detach()
action = x[:, :, state_dim:].requires_grad_(True)

# Compute log π(a|s) summed over trajectory
log_prob = policy.log_prob(state, action).sum()

# Gradient w.r.t. action only: ∇_a log π(a|s)
grad_action = torch.autograd.grad(log_prob, action)[0]

# Normalize per timestep
grad_action = grad_action / (grad_action.norm(dim=-1, keepdim=True) + 1e-6)
```

**`gradlog_diffusion()` — For diffusion policies:**
- Calls `policy.grad_log_prob(states, actions)` directly (analytic gradient)

**Guidance applied in `default_sample_fn()`:**
```python
model_mean = unnormalize(model_mean)

for _ in range(k_guide):  # typically 1-2
    gradient = gradlog_diffusion(target_policy, model_mean, ...)
    neg_grad = gradlog(behavior_policy, model_mean, ...) if use_neg_grad else 0

    guide = gradient - ratio * neg_grad  # ratio ≈ 0.25
    guide = action_scale * guide         # action_scale ≈ 0.2

    if clamp:
        guide = torch.clamp(guide, -l_inf, l_inf)

    model_mean = model_mean + guide
    model_mean = normalize(model_mean)
    model_mean = apply_conditioning(model_mean, cond, state_dim)
    model_mean = unnormalize(model_mean)
```

**Guidance hyperparams (from cheetah.json):**
```json
{
  "action_scale": 0.2,
  "use_adaptive": false,
  "use_neg_grad": true,
  "normalize_grad": true,
  "k_guide": 1,
  "use_action_grad_only": true,
  "clamp": false,
  "l_inf": 1,
  "ratio": 0.25
}
```

Key insight: guidance operates **only on action dimensions**. Target policy gradient pushes actions toward π; subtracting behavior gradient (scaled by 0.25) pushes away from behavior modes.

### Stitching Loop (`generate_full_trajectory`)

Autoregressive chunk generation with **1-step overlap**:

```python
alive_indices = torch.arange(batch_size)
conditions = {0: normalized_initial_state}
all_trajectories = zeros(batch_size, T_gen, transition_dim)
end_indices = full(batch_size, T_gen)
total_generated = 0

while alive_indices.numel() > 0 and total_generated < T_gen:
    # Generate chunk of length T (e.g., 4)
    samples = diffusion.conditional_sample(
        shape=(len(alive_indices), T, transition_dim),
        conditions=conditions,
        guided=True, ...
    )
    samples = unnormalize(samples)

    # Check termination per-trajectory, store results
    for local_idx, global_idx in enumerate(alive_indices):
        for step in range(T):
            if is_terminated(samples[local_idx, step, :state_dim]):
                end_indices[global_idx] = total_generated + step
                # remove from alive
                break

    # Store T-1 steps (last step becomes next chunk's condition)
    all_trajectories[alive, total_generated:total_generated+T-1] = samples[:, :-1]

    # Condition next chunk on last state (avoid normalize round-trip)
    last_states = samples[alive, -1, :state_dim]
    conditions = {0: normalize(last_states)}

    total_generated += T - 1  # advance by T-1 (1-step overlap)
```

**Design principles:**
- **Chunk size is tiny** — T=4 in Cheetah/Hopper (not 8)
- **1-step overlap is fundamental** — last state of chunk k = first state of chunk k+1, enforcing continuity
- **Conditioning always in normalized space**; termination checks in unnormalized space
- **Per-trajectory early stopping** via `is_terminated_fn` + `end_indices`
- With T=4, advances 3 steps per iteration → 768/3 ≈ 256 iterations to fill a trajectory

### Reward Scoring

**RewardEnsembleEstimator**: ensemble of N bootstrapped MLPs `[64, 64, 1]`, MSE loss, 1000 iterations, batch 32, lr 1e-3. Each model trained on 50% random subsample.

**Scoring loop:**
```python
for i in range(num_trajectories):
    sum_reward, gamma_t = 0, 1.0
    for t in range(end_indices[i]):
        state = samples[i, t, :state_dim]
        action = samples[i, t, state_dim:]
        reward = reward_fn(env, state, action)  # or reward_estimator.predict(...)
        sum_reward += reward * gamma_t
        gamma_t *= gamma
    all_rewards.append(sum_reward)
```

### Evaluation Harness (`evaluate_policies`)

Multi-policy comparison across multiple trials:

1. **Oracle**: Roll out each target policy online → ground truth V^π
2. **Normalize**: Map values to [0,1] using min/max across policies
3. **Per trial** (different seeds): generate trajectories, score, compute normalized estimate
4. **Metrics**:
   - **MSE / Log RMSE**: estimation accuracy on normalized scale
   - **Spearman rank correlation**: policy ranking quality (perfect = 1.0)
   - **Regret@k**: gap between true best and best-estimated top-k policy
5. Results saved to JSON

### Input Data Format

```python
DataType = List[Dict]  # List of episodes
{
    'states': np.ndarray(T, state_dim),
    'actions': np.ndarray(T, action_dim),
    'rewards': np.ndarray(T),
    'next-states': np.ndarray(T, state_dim)
}
```

Normalization: per-dimension mean/std computed over full dataset, applied to concatenated `[state, action]`.

### Key Takeaways

1. **Chunk size is T=4**, much smaller than our T=8. Smaller chunks = more stitching iterations but easier to learn.
2. **1-step overlap** in stitching is not optional — it's how SOPE maintains trajectory continuity.
3. **Guidance is the whole point** — the diffusion model learns general dynamics from diverse off-policy data, and guidance steers it toward the target policy. Without guidance, SOPE just generates behavior-policy-like trajectories.
4. **Action weighting (5x)** in the loss is important — actions are harder to learn and directly affect guidance quality.
5. **Cosine beta schedule** (not linear) for better signal preservation.
6. **Model capacity**: dim_mults=(1,2,4,8) with dim=32 — substantially larger than our (1,2).

---

## New Path Forward: Off-Policy MVP with Guidance

### The Fundamental Problem with the Current Setup

The current on-policy approach is **logically circular**. If we collect rollouts from policy π and train a diffusion model on them, we already have the returns from those rollouts. We can just average them — that's Monte Carlo OPE with zero approximation error. The diffusion model adds nothing.

SOPE's value proposition is: **evaluate a policy you haven't run**, using data from a *different* source. The diffusion model learns plausible dynamics from off-policy data, and guidance steers generation toward the target policy's behavior. Without guidance on off-policy data, there's no reason SOPE exists.

So the current setup — on-policy data, no guidance — can never be a convincing demonstration, no matter how well the diffusion model works.

### What a Convincing MVP Needs

Three things:

1. **Off-policy training data** — a dataset NOT generated by the target policies
2. **Multiple target policies of varying quality** — so we can test ranking
3. **Guidance** — so the diffusion model generates policy-specific trajectories

### Recommended Setup

#### Data Source: Robomimic's Demonstration Datasets

Robomimic ships with human demonstration datasets for Lift (`low_dim.hdf5`). These are large (~200 demos), diverse, and readily available — no collection step needed. This matches SOPE's paradigm exactly: a large pre-collected dataset from a "behavior" distribution that isn't any specific target policy.

Alternatively, collect ~200 rollouts from one trained policy (call it "behavior policy"), then evaluate *different* policies. This is closer to SOPE's D4RL setup and gives us a known behavior policy for the optional negative gradient term.

#### Target Policies: BC_Gaussian, Not Diffusion Policy

BC_Gaussian policies output `(μ(s), σ(s))` — a Gaussian over actions. The guidance gradient is trivial:

```python
# ∇_a log π(a|s) = -(a - μ(s)) / σ(s)²
grad = -(action - policy.mean(state)) / policy.variance(state)
```

That's ~10 lines of code vs. the nightmare of extracting gradients from a diffusion policy. Train 3-5 BC_Gaussian policies of varying quality by varying:
- Amount of training data (10%, 25%, 50%, 100% of demos)
- Or training duration (early vs late checkpoints)
- Or network size

This gives a spread of policy qualities to rank.

#### Diffusion Model: Use SOPE's Code Directly

Rather than debugging our SopeDiffuser wrapper (shape mismatches, memorization failures, etc.), **use SOPE's `GaussianDiffusion` + `TemporalUnet` + `Diffuser` classes directly** with a thin adapter that feeds robomimic data in SOPE's expected format:

```python
# SOPE expects: List[Dict] with keys 'states', 'actions', 'rewards', 'next-states'
# We just convert robomimic rollouts/demos to this format
```

This sidesteps every bug we've found. The reference code is tested and works on D4RL. We only need to write:
- A data adapter (robomimic → SOPE format)
- A policy wrapper (BC_Gaussian → `grad_log_prob` interface)
- A reward function (already done: `cube_z > 0.84`)

#### Architecture: Match SOPE Exactly

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size (T) | 4 | SOPE default, not 8 |
| dim_mults | (1, 2, 4, 8) | SOPE default |
| dim | 32 | SOPE default |
| Diffusion steps | 256 | SOPE default |
| Beta schedule | Cosine | SOPE default |
| Action weight | 5x | SOPE default |
| predict_epsilon | True | SOPE default |
| Stitching overlap | 1 step | SOPE default, non-negotiable |

Drop quaternions (state_dim=11, action_dim=4, transition_dim=15) since they're irrelevant for Lift reward and hard for diffusion.

#### Evaluation

- **Oracle**: 100 online rollouts per target policy → ground truth V^π
- **OPE**: 50 guided synthetic trajectories per policy → estimated V^π
- **Metrics**: Spearman rank correlation across policies, per-policy relative error

#### What We Actually Need to Build

Only three new pieces, everything else is SOPE's existing code:

1. **Data adapter** — convert robomimic demos/rollouts to `List[Dict{'states', 'actions', 'rewards', 'next-states'}]`
2. **BC_Gaussian policy wrapper** — expose `grad_log_prob(states, actions)` using the analytic Gaussian gradient
3. **Glue notebook** — wire it all together: load data → train diffusion → for each target policy, run guided stitching → score → compare to oracle

#### Why This Is Better

- **Intellectually honest**: actually demonstrates SOPE's contribution (off-policy + guidance)
- **Less code**: reuse SOPE's tested implementation instead of debugging our wrapper
- **Simpler guidance**: BC_Gaussian has analytic gradients, ~10 lines
- **Larger dataset**: human demos give us hundreds of trajectories, not 50
- **Falsifiable**: if it doesn't rank 3 policies correctly, we know it doesn't work

The main risk is that Lift may be too simple (all decent policies succeed ~100%), making ranking trivial or impossible. If so, move to a harder task (Can, Square) where policy quality varies more.
