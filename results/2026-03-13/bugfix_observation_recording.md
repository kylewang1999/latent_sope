# Bugfix: Rollout Recorder Drops Terminal Observation

**Date:** 2026-03-13
**Status:** FIXED — rollout.py updated, rollouts need re-collection

## Bug Summary

The `rollout()` function in `rollout.py` records pre-step `obs` at each timestep but never records the post-step `next_obs` from the final transition when the episode terminates on `done` or `success`. This means the observation where the task succeeds (e.g. `cube_z > 0.84` for Lift) is never stored.

## Impact

This bug was the **root cause** of v0.3.2.3's total failure (0% SR across all configs). Without success states in the training data, the chunk diffuser cannot generate trajectories that cross the success threshold, regardless of guidance.

- **v0.3.2.2**: Masked by 200 expert demos in training data (expert HDF5 uses a different recording pipeline that doesn't have this bug, reaching cube_z = 0.864–0.886)
- **v0.3.2.3**: Fully exposed — medium-only behavior data has max cube_z ≈ 0.835, below the 0.84 success threshold

## Root Cause

In `rollout()` (lines 488–528), the loop does:

```
for step_i in range(horizon):
    act = policy(obs)
    next_obs, reward, done, info = env.step(act)  # next_obs has success state
    ...
    recorder.record_step(obs=obs, ...)             # records pre-step obs
    if done or success:
        break                                       # next_obs is lost
    obs = next_obs                                  # only reached if not done
```

The final `next_obs` (where cube_z > 0.84) is passed to `record_step` but only stored if `store_next_obs=True`, which defaults to `False`. And even then, the `obs` for the terminal step is still the pre-step observation.

## Fix

Two changes to `rollout.py`:

### 1. New method `RolloutLatentRecorder.record_terminal_obs()` (line 262)

Records the terminal observation after done/success. Appends the obs and its latent encoding, plus sentinel zero-action/zero-reward to keep all per-step arrays aligned.

### 2. Call it on episode termination (line 522–528)

```python
if done or success:
    if recorder is not None:
        recorder.record_terminal_obs(next_obs)
    break
```

### Effect on trajectory shape

Before fix: successful episode with 40 steps → latents shape `(40, frame_stack, 19)`, max cube_z ≈ 0.835
After fix: same episode → latents shape `(41, frame_stack, 19)`, final entry has cube_z > 0.84

The extra entry has a zero-action sentinel. Downstream consumers (chunking, diffuser training) already handle variable-length trajectories and will naturally include the success state in their chunks.

### Non-success episodes

For episodes that reach the horizon without success, the loop ends without `break`, so `record_terminal_obs` is not called — these episodes are unchanged. For episodes that terminate on `done=True` but `success=False`, the terminal obs is still recorded (harmless, just one more non-success state).

## Verification Plan

After re-collecting rollouts with the fix:
1. Check that successful rollouts now have `max(cube_z) > 0.84` in the recorded data
2. Confirm trajectory lengths are +1 for successful episodes vs before
3. Re-run v0.3.2.3 — expect nonzero unguided SR and meaningful guidance effects

## Files Changed

- `src/latent_sope/robomimic_interface/rollout.py`: Added `record_terminal_obs()` method and call site
