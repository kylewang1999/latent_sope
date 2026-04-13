# Guided Trajectory OPE Script

Relevant code:

- [scripts/test_ope_guided.py](../scripts/test_ope_guided.py)
- [src/robomimic_interface/checkpoints.py](../src/robomimic_interface/checkpoints.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py)

## 1. Summary

[`scripts/test_ope_guided.py`](../scripts/test_ope_guided.py) is the guided
trajectory-level OPE entrypoint. It mirrors the rollout selection, reward
aggregation, and JSON reporting structure of
[`scripts/test_ope.py`](../scripts/test_ope.py), but it attaches robomimic
diffusion policies for target and behavior guidance before autoregressive
trajectory generation.

The script is guided-only. Unguided trajectory OPE remains the responsibility
of [`scripts/test_ope.py`](../scripts/test_ope.py).

## 2. Script Behavior

### 2.1 Inputs

The script preserves the main checkpoint, data, split, device, and batching
arguments from [`scripts/test_ope.py`](../scripts/test_ope.py), and adds:

- `--target-policy-checkpoint`
- `--behavior-policy-checkpoint`
- `--action-score-scale`
- `--num-guidance-iters`
- `--action-score-postprocess`
- `--action-neg-score-weight`
- `--clamp-linf`
- `--target-score-timestep`
- `--behavior-score-timestep`
- `--use-adaptive` / `--no-use-adaptive`
- `--use-neg-grad` / `--no-use-neg-grad`

The default target and behavior checkpoints are:

- `data/policy/rmimic-lift-ph-lowdim_diffusion_260130/models/model_epoch_50_low_dim_v15_success_0.92.pth`
- `data/policy/rmimic-lift-ph-lowdim_diffusion_260130/last.pth`

### 2.2 Policy Checkpoint Resolution

The guided script accepts full checkpoint file paths directly. Each checkpoint
path is resolved by walking upward to the nearest ancestor containing
`config.json`, treating that directory as the robomimic run root, and then
passing the checkpoint path relative to that run root into the existing
checkpoint helpers.

This keeps the CLI short while still using the same robomimic reconstruction
path as the rest of the repository.

### 2.3 Outputs

The JSON report preserves the core OPE fields from
[`scripts/test_ope.py`](../scripts/test_ope.py) and adds:

- `guided: true`
- target and behavior policy checkpoint paths
- target and behavior score timesteps
- nested `guidance_config`

The guided script writes to a separate default report file:

- `<diffusion-checkpoint-stem>_ope_guided_report.json`

so guided runs do not overwrite unguided OPE reports.

## 3. Loader Fix

[`build_algo_from_checkpoint`](../src/robomimic_interface/checkpoints.py) now
initializes both robomimic obs-utils module namespaces before constructing the
policy network:

- `robomimic.utils.obs_utils`
- `third_party.robomimic.robomimic.utils.obs_utils`

This avoids a duplicated-module failure mode where the canonical robomimic
network builder reads modality globals from `robomimic.utils.obs_utils` while
the local helper had only initialized the vendored module namespace.

## 4. Guidance Surface

The script forwards the current local FiLM guidance kwargs expected by
[`FilmGaussianDiffusion.conditional_sample`](../src/diffusion.py):

- `action_score_scale`
- `use_adaptive`
- `use_neg_grad`
- `action_score_postprocess`
- `num_guidance_iters`
- `clamp_linf`
- `action_neg_score_weight`

Target and behavior policy score timesteps are configured separately through
[`DiffusionPolicyScoreConfig`](../src/robomimic_interface/policy.py).

If `--no-use-neg-grad` is set, the script skips behavior-policy loading and
runs target-only guidance.

## 5. Validation

Re-run:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 -m py_compile scripts/test_ope_guided.py src/robomimic_interface/checkpoints.py
```

Smoke test:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && MPLCONFIGDIR=/tmp/matplotlib python3 scripts/test_ope_guided.py --device cpu --max-trajectories 1 --rollout-batch-size 1 --json
```
