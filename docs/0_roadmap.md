# Project Roadmap

## 1. Current Direction

The current repository centers on a local SOPE-inspired diffusion stack built
around:

- [src/diffusion.py](../src/diffusion.py) for model and wrapper logic
- [src/sampling.py](../src/sampling.py) for local sampling and guided sampling
- [src/train.py](../src/train.py) for training orchestration
- [src/eval.py](../src/eval.py) for chunk, rollout, and OPE evaluation
- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py) for
  robomimic diffusion-policy guidance

The main research direction is to make the local rollout-backed diffuser
numerically consistent enough that chunk RMSE, rollout MSE, guidance behavior,
and OPE estimates can be interpreted together.

## 2. Active Workstreams

### 2.1 Guidance Semantics

- keep the local guidance contract diffusion-only
- validate how much error is introduced by the fixed
  `DiffusionPolicyScoreConfig.score_timestep` approximation
- improve confidence that guided and unguided sampling differ only through the
  intended action-space drift term

### 2.2 Rollout Quality

- track chunk RMSE, rollout MSE, and autoregressive trajectory quality together
- continue checking whether FiLM conditioning and prefix handling are aligned
  with the rollout dataset contract
- use the canonical docs below instead of older meeting-log notes when
  reasoning about current behavior

### 2.3 Reward Modeling

- keep the reward predictor as a transition-level immediate-reward regressor
- stress-test sparse-reward behavior before changing the model family
- treat reward-class imbalance and reward-transform conventions as the next
  likely sources of error if OPE quality stalls

## 3. Canonical Docs

Use these notes as the current owners of each topic:

- [docs/1_robomimic_diffusion_score_guidance.md](./1_robomimic_diffusion_score_guidance.md)
  for robomimic guidance, FiLM conditioning, and visual-backbone behavior
- [docs/2_reward_predictor.md](./2_reward_predictor.md) for reward-model
  semantics
- [docs/3_training_pipeline.md](./3_training_pipeline.md) for training and
  evaluation entrypoints
- [docs/6_sope_diffusion_contract.md](./6_sope_diffusion_contract.md) for DDPM,
  conditioning, and chunk-contract details
- [docs/8_autoregressive_trajectory_generation.md](./8_autoregressive_trajectory_generation.md)
  for rollout generation, OPE returns, and rollout reporting metrics
- [docs/13_predict_epsilon_guidance_compatibility.md](./13_predict_epsilon_guidance_compatibility.md)
  for parameterization and local sampling guidance semantics

## 4. Validation Priorities

When diffusion behavior changes, prefer the smallest checks that still exercise
the affected contract:

1. `py_compile` on the touched modules in the `latent_sope` environment
2. one deterministic guided-versus-unguided chunk-sampling smoke check
3. one small chunk or rollout evaluation if the change affects training,
   sampling, normalization, or reward accounting
