# Autoregressive Trajectory Generation, Guided OPE, And Rollout Reporting

Relevant code:

- [scripts/test_ope.py](../scripts/test_ope.py)
- [scripts/test_ope_guided.py](../scripts/test_ope_guided.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/eval.py](../src/eval.py)
- [src/robomimic_interface/checkpoints.py](../src/robomimic_interface/checkpoints.py)
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
- [third_party/sope/opelab/core/baselines/diffuser.py](../third_party/sope/opelab/core/baselines/diffuser.py)

## 1. Summary

The local rollout evaluation stack owns four closely related behaviors:

1. autoregressive full-trajectory generation
2. discounted unguided OPE return estimation with a learned reward model
3. guided trajectory-level OPE using robomimic target and behavior policies
4. rollout-level reporting metrics that compare generated and held-out
   trajectories

These belong in one note because they share the same rollout horizon,
normalization contract, and saved rollout assets.

## 2. Local Autoregressive Generation

`SopeDiffuser.generate_full_trajectory(...)` is the canonical local rollout
implementation. It:

- normalizes the initial condition internally
- samples one chunk at a time
- feeds the trailing generated state prefix into the next chunk
- truncates the last chunk when needed
- returns unnormalized state and action arrays

The chunk-autoregressive recurrence is:

$$\begin{align}
\text{prefix}_{k+1} = \hat{s}_{k, -F:},
\end{align}$$

where $F$ is `frame_stack`.

Relative to upstream SOPE's rollout path, the local wrapper:

- uses FiLM-style context conditioning instead of in-painting
- conditions on a flattened state prefix rather than a single state
- stores each generated chunk directly instead of overlap-by-one stitching
- does not currently implement upstream `tanh_action` handling
- does not currently stop generation early on environment termination

## 3. Rollout-Time Reward And OPE Return

`SopeDiffuser.ope_estimate(...)` applies the learned reward predictor only
after sampling. The path is:

1. generate rollout states and actions
2. score each generated transition with the reward model
3. discount and sum predicted immediate rewards
4. average the returns across the rollout batch

The per-step reward estimate is

$$\begin{align}
\hat{r}_t = \hat{r}_\phi(s_t, a_t),
\end{align}$$

and the per-trajectory OPE return is

$$\begin{align}
\hat{J}_i
=
\sum_{t=0}^{T-1}
\gamma_{\text{ope}}^t \hat{r}_\phi(s_{i,t}, a_{i,t}).
\end{align}$$

`SopeDiffusionConfig.ope_gamma` affects only this discounted-return
computation. It does not change diffusion training or guidance.

## 4. Rollout Reporting Metrics

[`scripts/test_ope.py`](../scripts/test_ope.py) reports three rollout-level
quantities on the selected split:

- `return_ope_estimate`: the diffusion model's mean discounted return estimate
- `return_gt_transformed`: the held-out rollout return after applying the same
  dataset reward transform
- `autoregressive_rollout_mse`: the mean squared error between generated and
  true rollout transitions over the shared rollout horizon

The true rollout return is computed by loading raw per-step rewards, applying
the dataset-configured reward transform, discounting with `ope_gamma`, and then
averaging across the selected split.

The rollout MSE is

$$\begin{align}
\text{MSE}
=
\frac{1}{N T D}
\sum_{i=1}^{N}
\sum_{t=1}^{T}
\left\|
\begin{bmatrix}
\hat{s}_{i,t} \\
\hat{a}_{i,t}
\end{bmatrix}
-
\begin{bmatrix}
s_{i,t} \\
a_{i,t}
\end{bmatrix}
\right\|_2^2,
\end{align}$$

where $T$ is the shared evaluated horizon and $D$ is the full transition
dimension.

## 5. Guided Trajectory OPE Script

[`scripts/test_ope_guided.py`](../scripts/test_ope_guided.py) is the guided
trajectory-level OPE entrypoint. It mirrors the rollout selection, reward
aggregation, and JSON reporting structure of
[`scripts/test_ope.py`](../scripts/test_ope.py), but it attaches robomimic
diffusion policies for target and behavior guidance before autoregressive
trajectory generation.

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

The guided script accepts full checkpoint file paths directly. Each checkpoint
path is resolved by walking upward to the nearest ancestor containing
`config.json`, treating that directory as the robomimic run root, and then
passing the checkpoint path relative to that run root into the existing
checkpoint helpers.

[`build_algo_from_checkpoint`](../src/robomimic_interface/checkpoints.py) also
initializes both robomimic obs-utils module namespaces before constructing the
policy network:

- `robomimic.utils.obs_utils`
- `third_party.robomimic.robomimic.utils.obs_utils`

This avoids a duplicated-module failure mode where the canonical robomimic
network builder reads modality globals from `robomimic.utils.obs_utils` while
the local helper had only initialized the vendored namespace.

The JSON report preserves the core OPE fields from
[`scripts/test_ope.py`](../scripts/test_ope.py) and adds:

- `guided: true`
- target and behavior policy checkpoint paths
- target and behavior score timesteps
- nested `guidance_config`

The guided script writes to a separate default report file
`<diffusion-checkpoint-stem>_ope_guided_report.json`, so guided runs do not
overwrite unguided OPE reports.

The current FiLM guidance surface forwarded by the script is:

- `action_score_scale`
- `use_adaptive`
- `use_neg_grad`
- `action_score_postprocess`
- `num_guidance_iters`
- `clamp_linf`
- `action_neg_score_weight`

If `--no-use-neg-grad` is set, the script skips behavior-policy loading and
runs target-only guidance.

## 6. Normalization Contract

The public rollout interface stays in unnormalized state/action space, but
sampling and autoregressive reconditioning still happen in normalized
transition space internally.

That means:

- the initial condition is normalized with the same transition-level statistics
  used during training
- sampled chunks are denormalized only for the public return value
- the next autoregressive prefix is built from the normalized generated state
  suffix, not from the denormalized output arrays

## 7. Validation

The lightweight validation for this codepath is:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python -m py_compile src/diffusion.py src/eval.py scripts/test_ope.py scripts/test_ope_guided.py src/robomimic_interface/checkpoints.py src/robomimic_interface/dataset.py src/robomimic_interface/rollout.py
```

If the rollout or reward contract changes, rerun the smallest available chunk
or rollout evaluation that exercises:

- autoregressive generation
- transformed-return reporting
- rollout MSE against saved rollout assets
- one guided trajectory-OPE smoke test

Guided smoke test:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && MPLCONFIGDIR=/tmp/matplotlib python3 scripts/test_ope_guided.py --device cpu --max-trajectories 1 --rollout-batch-size 1 --json
```
