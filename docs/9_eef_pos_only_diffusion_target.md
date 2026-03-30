# EEF-Only Diffusion Target

Relevant code:
- [src/sope_diffuser.py](../src/sope_diffuser.py)
- [src/train.py](../src/train.py)
- [scripts/train_sope.py](../scripts/train_sope.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
- [src/robomimic_interface/encoders.py](../src/robomimic_interface/encoders.py)

This note records the training-time debugging option that restricts the SOPE
diffusion loss to the robot end-effector position only.

## Slice Derivation

The rollout latents used by `scripts/train_sope.py` come from the low-dim
concatenation path in [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
and [src/robomimic_interface/encoders.py](../src/robomimic_interface/encoders.py).

For robomimic low-dim Lift observations, the default observation keys are:

- `robot0_eef_pos`
- `robot0_eef_quat`
- `robot0_gripper_qpos`
- `object`

The repo sorts those keys before concatenation, so the effective per-timestep
latent layout is:

- `object`: 10 dims
- `robot0_eef_pos`: 3 dims
- `robot0_eef_quat`: 4 dims
- `robot0_gripper_qpos`: 2 dims

Therefore the per-timestep `robot0_eef_pos` slice is `$[10:13)$`.

## What Changed

`SopeDiffusionConfig` now exposes `diffuser_eef_pos_only`, and
`scripts/train_sope.py` exposes the matching CLI flag
`--diffuser-eef-pos-only`.

When enabled:

- the rollout, dataset, and model tensor shapes remain unchanged
- the diffusion loss weights are masked so only the future
  `robot0_eef_pos` state slice contributes to training loss
- non-EEF state dimensions and action dimensions remain present in the model
  input and output, but they are not supervised by the diffusion loss in this
  debugging mode

This is intentionally a loss-mask ablation, not a dataset-schema change.

## Validation

Validation run after the code change:

- `python3 -m py_compile src/sope_diffuser.py src/train.py scripts/train_sope.py`

Recommended follow-up:

- re-run `scripts/train_sope.py` once with and once without
  `--diffuser-eef-pos-only`
- compare held-out chunk RMSE to see whether the large reconstruction error is
  concentrated in non-EEF state channels or in the end-effector position itself
