# Guided Success State Index In `src/eval.py`

## 1. Summary

[`src/eval.py`](../src/eval.py) no longer hard-codes the Lift guided-success
state index as `2`.

Instead, `evaluate_guided_multipolicy_ope(...)` now infers the index from the
resolved [`PolicyFeatureHook`](../src/robomimic_interface/rollout.py) feature
layout unless the caller explicitly overrides it.

## 2. Why The Old Constant Was Wrong

The old constant assumed the generated state began with the raw Lift
`object[0:3] = (x, y, z)` block, so `cube_z` would live at global index `2`.

That assumption does not hold for every feature mode:

- `low_dim_concat` uses the configured low-dimensional key order from the
  robomimic policy config
- `both` uses the policy's actual observation-encoder concatenation order

## 3. Current Lift Offsets

For the local Lift image checkpoint under
[`data/policy/rmimic-lift-mh-image-v15-diffusion_260123`](../data/policy/rmimic-lift-mh-image-v15-diffusion_260123),
the inferred offsets are:

- `low_dim_concat`: `robot0_eef_pos(3) | robot0_eef_quat(4) | robot0_gripper_qpos(2) | object(10)`
- `both`: `agentview_image(64) | object(10) | robot0_eef_pos(3) | robot0_eef_quat(4) | robot0_eye_in_hand_image(64) | robot0_gripper_qpos(2)`

Since Lift stores `cube_z` as `object[2]`, the correct global indices are:

- `low_dim_concat`: `11`
- `both`: `66`

## 4. Code Paths

The inference logic is implemented in [`src/eval.py`](../src/eval.py):

- `_obs_encoder_feature_width(...)`
- `_resolve_guided_success_object_z_index(...)`
- `evaluate_guided_multipolicy_ope(...)`

## 5. Validation

When validating this change, re-run a small guided multipolicy OPE smoke test
and confirm that:

- the top-level report stores the inferred `guided_success_state_index`
- the current Lift `feat_type="both"` setup reports `guided_success_state_index = 66`
- callers can still override the inferred value by passing
  `guided_success_object_z_index` explicitly
