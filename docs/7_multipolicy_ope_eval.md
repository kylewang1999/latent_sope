# Multipolicy Guided OPE Eval

Relevant code:

- [scripts/test_ope_guided.py](../scripts/test_ope_guided.py)
- [scripts/test_ope_guided_multipolicy.py](../scripts/test_ope_guided_multipolicy.py)
- [src/robomimic_interface/checkpoints.py](../src/robomimic_interface/checkpoints.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)

## 1. Summary

`test_ope_guided_multipolicy.py` extends the single-policy guided OPE script to
evaluate a whole directory of target-policy checkpoints.

For each target policy, it now:

1. generates guided SOPE trajectories from the selected held-out initial states
2. predicts transformed cumulative reward with the SOPE reward MLP
3. rolls the target policy out online from the corresponding source-demo
   initial conditions
4. re-encodes those target rollouts in the behavior-policy `feat_type="both"`
   feature space
5. compares policy ranking and rollout accuracy across all target policies

The default report is written under `<run_dir>/eval_ope/` as
`*_ope_guided_multipolicy_report_<mmdd_mmss>.json`.

`evaluate_guided_multipolicy_ope(...)` rewrites that JSON after each completed
target-policy evaluation, so an interrupted run still preserves the finished
prefix of `policies`.

The default shared rollout horizon is `80` steps. Both guided SOPE trajectories
and true robomimic simulator rollouts use that same configured horizon unless
the user overrides it on the CLI.

## 2. Initial Conditions And True Rollouts

The selected eval files under `data/robomimic/.../postprocessed_for_ope/both`
are postprocessed robomimic demos, not free-running behavior-policy rollouts.

Each selected file supplies:

- `source_dataset_path`
- `demo_key`

The multipolicy script uses those attrs to reopen the original robomimic HDF5
and reconstruct the initial simulator state for that demo from:

- `states[0]`
- `model_file`
- optional `ep_meta`

The online rollout environment itself is reconstructed from the source
dataset's env metadata via robomimic `FileUtils.get_env_metadata_from_dataset`
and `EnvUtils.create_env_from_metadata(...)`. The script then applies the same
robomimic rollout wrappers that the reference target-policy checkpoint expects,
for example frame stacking. This keeps the base simulator tied to the dataset
that supplied the initial conditions without dropping rollout-time policy
contracts that are stored in checkpoint config.

Target policies are then rolled out from those exact demo initial conditions.
This keeps the target-policy comparison anchored to the same test-trajectory
family used for guided SOPE evaluation.

Unlike the earlier shortest-trajectory truncation behavior, the multipolicy
script now uses one explicit configured rollout horizon for both paths. This
lets the true target-policy rollouts run longer than the shortest held-out
behavior-demo file while still preserving per-step diffusion-versus-true
comparison over the same number of requested steps.

## 3. Re-Encoding And Metrics

### 3.1 Behavior-Feature Re-Encoding

The SOPE diffuser and reward predictor for
`logs/train-sope-feat:both_0417_163242` live in the behavior-policy feature
space. Therefore the true target-policy rollouts are re-encoded with the
behavior checkpoint, not the target checkpoint, before rollout MSE is computed.

This means guided trajectory MSE is measured against:

- target-policy actions executed online in the simulator
- observations re-encoded by the behavior policy with `feat_type="both"`

instead of against the original behavior-demo trajectory.

### 3.2 Return Semantics

The current SOPE run uses `reward_transform="minus_one"`, so the reward model
predicts transformed per-step rewards. The primary multipolicy summary metrics
therefore compare transformed cumulative values:

- Spearman correlation over per-policy mean predicted transformed return versus
  per-policy mean true transformed return
- RMSE over those same per-policy mean values

The report also includes each policy's mean true raw environment return for
reference.

### 3.3 Success Metrics

The report contains two success-rate fields per target policy:

1. true rollout success rate from online robomimic rollouts
2. guided rollout success rate from generated SOPE trajectories

The guided success heuristic uses the preserved low-dimensional Lift prefix in
the `both` feature space and marks success when the generated cube-height
coordinate crosses `0.84`.

## 4. Report Schema And Validation

Top-level report fields include:

- resolved SOPE checkpoints
- split metadata and rollout horizon
- configured rollout horizon (`configured_rollout_horizon`, default `80`)
- rollout env provenance (`rollout_env_source="dataset_metadata"` and
  `rollout_env_dataset_path`)
- rollout wrapper provenance (`rollout_env_wrapper_checkpoint`)
- guidance config
- transformed-return Spearman and RMSE
- one aggregate entry per target checkpoint under `policies`

Each `policies` entry includes at least:

- mean predicted transformed return
- mean true transformed return
- mean true raw return
- true rollout success rate
- guided rollout success rate
- guided-versus-target rollout MSE and RMSE

Validation command:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && \
python scripts/test_ope_guided_multipolicy.py \
    --max-trajectories 2 \
    --max-target-policies 2 \
    --json
```
