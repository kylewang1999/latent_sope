# Guidance Implementation Notes

**Date:** 2026-03-11
**Purpose:** Document how SOPE guidance works and what's needed to get good LogRMSE

---

## What SOPE Actually Uses for Guidance (Reference Implementation)

**Critical finding: SOPE always uses the actual target policy for guidance — never a proxy.**

### SOPE's Three Experiment Domains

#### 1. D4RL (Hopper, Walker, HalfCheetah)
- **Target policies**: `D4RLSACPolicy` — SAC policies trained to different quality levels
- **Behavior policy**: `D4RLPolicy` — policy embedded in D4RL dataset metadata (extracted from HDF5)
- **Guidance function**: `gradlog()` → calls `policy.log_prob_extended()` (forward pass through MLP to get mean/logstd) → autograd computes `nabla_a log pi(a|s)`
- Both target and behavior are the **actual policies**

#### 2. Gym (Pendulum, Acrobot)
- **Target policies**: `TD3Policy` — TD3 checkpoints at different training stages (e.g. `t5k.pkl`, `t15k.pkl`, `t25k.pkl`)
- **Behavior policy**: `TD3Policy` — a specific TD3 checkpoint
- **Guidance function**: same `gradlog()` mechanism

#### 3. Diffusion Policy Experiments (Hopper, Walker, HalfCheetah)
- **Target policies**: `DiffusionPolicy` — CleanDiffuser-based diffusion policies at different training stages
- **Behavior policy**: `D4RLPolicy` — D4RL dataset policy
- **Guidance function**: `gradlog_diffusion()` → calls `policy.grad_log_prob()` which uses the **diffusion score function** directly (not autograd): `score = diffusion_net(action, t=1, cond(state))`, returns `-score / sigma[1]`

### Policy Types and Their Guidance Interfaces

| Policy Type | Guidance Function | Method Called | How It Works |
|-------------|------------------|--------------|-------------|
| `D4RLPolicy` | `gradlog()` | `log_prob_extended(s, a)` | MLP forward → mean/logstd → Gaussian log-prob → autograd |
| `D4RLSACPolicy` | `gradlog()` | `log_prob_extended(s, a)` | SAC network forward → log-prob → autograd |
| `TD3Policy` | `gradlog()` | `log_prob_extended(s, a)` | TD3 actor forward → Gaussian log-prob → autograd |
| `DiffusionPolicy` | `gradlog_diffusion()` | `grad_log_prob(s, a)` | Score network at t=1 → `-score / sigma[1]` (no autograd) |

### Implication for Our Setup

In v0.2, we used a BC_Gaussian as guidance — but SOPE uses the actual target policy directly. We need to do the same: extract the score function from robomimic's DiffusionPolicyUNet.

**Options to match SOPE's approach:**
1. **Extract the diffusion score** from robomimic's DiffusionPolicyUNet, similar to SOPE's `DiffusionPolicy.grad_log_prob()` using the score network at t=1
2. **Use policy types that natively support `log_prob`** — e.g., train SAC/TD3-style policies as targets (leaves robomimic)

### Key Code Locations (SOPE Reference)
- `gradlog()`: `third_party/sope/opelab/core/baselines/diffusion/diffusion.py:31-89`
- `gradlog_diffusion()`: same file, lines 91-110
- `default_sample_fn()`: same file, lines 152-250
- `D4RLPolicy`: `third_party/sope/opelab/core/policy.py:302-501`
- `D4RLSACPolicy`: same file, lines 567-845
- `DiffusionPolicy`: same file, lines 847-931
- D4RL experiment entry: `third_party/sope/opelab/examples/d4rl/main_full.py`
- Gym experiment entry: `third_party/sope/opelab/examples/gym/main_full.py`
- Diffusion policy experiment entry: `third_party/sope/opelab/examples/diffusion_policy/main_diffusion.py`

---

## MVP v0.2.2: Extracting Diffusion Score for Guidance — Feasibility Analysis

### The Idea

Extract the **score function** from robomimic's DiffusionPolicyUNet directly — matching what SOPE does with its own `DiffusionPolicy.grad_log_prob()`.

**Why**: SOPE uses the actual target policy for guidance, not an approximation. v0.2 showed guidance is the biggest lever (2878% → 26% rel error), so using the real score function should push accuracy much further.

**How SOPE does it**: `DiffusionPolicy.grad_log_prob()` evaluates the noise prediction network at `t=1` (near-clean), returns `-score / sigma[1]`. No `log_prob` needed — just the gradient direction.

### Hurdles

#### 1. Action Dimension Mismatch
- Our chunks use **4-dim actions** (pos + gripper, orientation zeroed out)
- Robomimic's diffusion policy outputs **7-dim actions** (full action space)
- Would need to either map between them or evaluate score in full 7-dim space and project down

#### 2. State Representation Mismatch
- Diffusion policy conditions on the **raw obs dict** (with frame stacking), not our 11-dim reduced state vector
- Would need to reconstruct the obs dict from the latent vector to feed the conditioning network
- `LowDimConcatEncoder.decode_to_obs_dict()` exists but need to verify it produces what the policy expects

#### 3. Chunk-Level vs Timestep-Level Score
- SOPE's `DiffusionPolicy.grad_log_prob()` operates on **single (state, action) pairs**
- Robomimic's score is over an **entire action sequence** (e.g., 16-step action horizon)
- Getting per-timestep gradient means either:
  - Evaluate full sequence score and slice (but how to construct the full sequence from a single timestep?)
  - Run independently per timestep (expensive — one forward pass per timestep per chunk per denoising step)

#### 4. Sigma Schedule Mismatch
- SOPE uses `self.actor.sigma[1]` from CleanDiffuser's schedule
- Robomimic's diffusion policy has its own noise schedule with different parameterization
- Need to find the equivalent sigma at the near-clean timestep

### Assessment

These hurdles are fiddly but not impossible. After inspecting robomimic's internals (see below), the solutions are clear.

---

## MVP v0.2.2: Implementation Plan

### Robomimic's DiffusionPolicyUNet Internals

```
Observations (B, To=2, obs_dict)
    → ObsEncoder → (B, To=2, obs_features)
    → Flatten → (B, To * obs_features)  [global conditioning]

Noisy actions (B, Tp=16, action_dim=7)
    → ConditionalUnet1D(noisy_actions, timestep_t, conditioning)
    → Predicted noise (B, Tp=16, action_dim=7)
```

- **100 DDPM steps**, squaredcos_cap_v2 beta schedule
- **Predicts epsilon** (noise), not x0
- Action prediction horizon Tp = 16 steps, action dim = 7
- Observation horizon To = 2 (frame stacking), action horizon Ta = 8

Key files:
- Diffusion policy algo: `third_party/robomimic/robomimic/algo/diffusion_policy.py`
- ConditionalUnet1D: `third_party/robomimic/robomimic/models/diffusion_policy_nets.py`
- Our checkpoint config: `third_party/robomimic/diffusion_policy_trained_models/test/20260309132349/config.json`

### Why Diffusion Policies Don't Have `log_prob` But Do Have Scores

Diffusion models don't have a tractable likelihood — `log_prob` is explicitly `None` in SOPE's `DiffusionPolicy`. But guidance only needs `nabla_a log pi(a|s)`, not `log pi(a|s)` itself. Diffusion models natively estimate the score `nabla_x log p(x)` — that's literally what the denoising network learns.

SOPE's approach (`policy.py:923-928`):
```python
def grad_log_prob(self, state, action):
    cond_net = self.actor.model['condition']
    diffusion_net = self.actor.model['diffusion']
    score_fn = diffusion_net(action, t=1, cond_net(state))
    return -score_fn / self.actor.sigma[1]
```

Evaluate the noise network at t=1 (near-clean), return `-noise_pred / sigma[1]`. This is confirmed to be the path used in `default_sample_fn()` line 192-193 — it checks `policy.__class__.__name__ == 'DiffusionPolicy'` and routes to `gradlog_diffusion()`.

### How to Get the Score from Robomimic's Policy

```python
def grad_log_prob(obs_dict, actions_sequence):
    # 1. Encode observations
    cond = obs_encoder(obs_dict)          # (B, To * obs_features)

    # 2. Run noise network at t=1
    t = torch.ones(B) * 1                 # near-clean timestep
    noise_pred = unet(actions_sequence, t, cond)  # (B, 16, 7)

    # 3. Score = -noise / sigma
    return -noise_pred / sigma[1]         # (B, 16, 7)
```

### Resolving the 4 Hurdles

#### Hurdle 1 → Solution: Use full 7-dim actions
Stop reducing action dim. Use full 7-dim actions in the chunks. The dimension reduction to 4-dim was a convenience choice in v0.1, not a requirement. With full actions, the score network output directly matches.

#### Hurdle 2 → Solution: Use full 19-dim states
Stop reducing state dim too. Use full 19-dim states (keep quaternions). We already have `LowDimConcatEncoder.decode_to_obs_dict()` to reconstruct the obs dict from the 19-dim latent. For frame stacking (To=2), take the last 2 timesteps of state from the chunk.

#### Hurdle 3 → Solution: Chunk-level guidance (actually good news)
SOPE's `DiffusionPolicy.grad_log_prob()` operates on single (state, action) pairs because their CleanDiffuser policy is single-step. Robomimic's policy operates on **16-step action sequences**. Our chunks are ~8 timesteps.

This is actually a natural fit: pad or truncate our chunk's action sequence to length 16, evaluate the score over the full sequence, then extract the gradient for the timesteps we care about. Chunk-level guidance rather than per-timestep.

#### Hurdle 4 → Solution: One line of code
Compute `sigma[1]` from robomimic's beta schedule: `sigma[t] = sqrt(1 - alpha_bar[t])`. For squaredcos_cap_v2 with 100 steps, `sigma[1]` is a small number.

### Revised Pipeline (v0.1/v0.2 → v0.2.2)

```
v0.1/v0.2 pipeline (reduced dims):
  Chunks: (T, 15) = 11-dim state + 4-dim action
  Guidance: BC_Gaussian → gradlog()

v0.2.2 pipeline (full dims, actual policy score):
  Chunks: (T, 26) = 19-dim state + 7-dim action
  Guidance: robomimic DiffusionPolicyUNet score → gradlog_diffusion()
```

This means **retraining the chunk diffusion model** on full-dim data. But it eliminates all dimension mapping headaches and matches what SOPE actually does.

### What Needs to Change

1. **Chunk diffusion**: Retrain with transition_dim=26 (19+7) instead of 15 (11+4)
2. **New wrapper class**: `RobomimicDiffusionPolicyScorer` that implements `grad_log_prob(states, actions)` by calling the noise network at t=1
3. **Obs reconstruction**: Use `decode_to_obs_dict()` + frame stacking to feed the conditioning network
4. **Action sequence handling**: Pad/truncate chunk actions to match prediction_horizon=16
5. **Sigma computation**: Extract from robomimic's DDPM beta schedule
6. **Plug into existing `gradlog_diffusion()` path** — minimal changes to SOPE's sampling code

The biggest piece of work is the wrapper class (#2). Everything else is config changes or minor plumbing.
