# Rollout Horizon And Success Metrics

Relevant code:

- [src/eval.py](../src/eval.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
- [third_party/robomimic/robomimic/config/base_config.py](../third_party/robomimic/robomimic/config/base_config.py)
- [third_party/robomimic/robomimic/__init__.py](../third_party/robomimic/robomimic/__init__.py)
- [third_party/robomimic/robomimic/scripts/run_trained_agent.py](../third_party/robomimic/robomimic/scripts/run_trained_agent.py)
- [data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json](../data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json)

## 1. Summary

robomimic rollouts do not use a separate early-failure detector. A rollout ends
when one of the following happens:

- the task succeeds
- the environment reports `done`
- the configured rollout horizon is exhausted

For the local Lift MH diffusion-policy checkpoint, there are three different
horizon numbers that can matter:

- `400`: robomimic base-config rollout default
- `500`: robomimic dataset-registry recommendation for Lift MH
- `80`: local default for `evaluate_guided_multipolicy_ope(...)`

The current multipolicy report also exposes two trajectory-level metrics that
can look redundant:

- `mean_true_raw_return`
- `true_rollout_success_rate`

They are not equivalent by definition. They only coincide in the current sparse
Lift setup because successful rollouts receive total raw return `1.0`,
unsuccessful rollouts receive `0.0`, and the rollout loop stops immediately on
success.

## 2. Rollout Horizon Layers

### 2.1 robomimic Base Default

robomimic's base config sets:

- `config.experiment.rollout.horizon = 400`

This is the generic default maximum number of environment steps per evaluation
rollout in the library.

### 2.2 Dataset-Registry Recommendation

robomimic also stores task- and dataset-specific rollout horizons in its
dataset registry. For multi-human Lift datasets, the registry records:

- `lift / mh / raw -> 500`
- `lift / mh / low_dim -> 500`
- `lift / mh / image -> 500`

This is the recommended rollout horizon for Lift MH experiments when configs
are generated from the registry.

### 2.3 Current Local Checkpoint Config

The local checkpoint config at
[data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json](../data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json)
stores:

- `experiment.rollout.horizon = 400`

robomimic's
[run_trained_agent.py](../third_party/robomimic/robomimic/scripts/run_trained_agent.py)
uses the checkpoint's stored rollout horizon when the user does not pass
`--horizon` explicitly. For this checkpoint, a default rollout therefore caps
at `400` steps, not `500`.

### 2.4 Local Multipolicy OPE Default

The local multipolicy OPE entrypoint in
[src/eval.py](../src/eval.py) sets:

- `rollout_horizon = 80`

by default in `evaluate_guided_multipolicy_ope(...)`.

This `80`-step budget is applied to both:

- guided SOPE-generated trajectories
- true online robomimic rollouts from reconstructed demo initial conditions

As a result, the multipolicy OPE path intentionally does not use the full
robomimic checkpoint rollout horizon unless the caller overrides it.

## 3. What Counts As Failure

There is no separate branch that marks a trajectory as failed before the
horizon budget is consumed. The rollout loop simply steps until:

- `env.is_success()["task"]` becomes true
- `done` becomes true
- the `for step_i in range(horizon)` loop runs out

If the loop reaches the last allowed step without success, the rollout is just
an unsuccessful truncated rollout and the evaluator moves on to the next
initial condition.

## 4. `mean_true_raw_return` Versus `true_rollout_success_rate`

### 4.1 `true_rollout_success_rate`

In `evaluate_guided_multipolicy_ope(...)`,
`true_rollout_success_rate` is computed as the mean of `target_traj.success`
over all selected initial conditions.

The rollout path sets `target_traj.success` from
`bool(env.is_success()["task"])`.

This is therefore a per-trajectory success indicator averaged across rollout
attempts.

### 4.2 `mean_true_raw_return`

In the same report, `mean_true_raw_return` is computed as the mean of
`target_traj.total_reward`.

`target_traj.total_reward` is the sum of the per-step environment rewards
encountered along one rollout trajectory.

This means `mean_true_raw_return` is:

- the average trajectory return

and not:

- the average per-step reward over every visited state-time pair

### 4.3 Why They Match In The Current Lift Setup

The current rollout environment is reconstructed from
[data/robomimic/lift/mh/image_v15.hdf5](../data/robomimic/lift/mh/image_v15.hdf5),
whose stored env metadata uses:

- `reward_shaping = false`

robomimic documents this setting as sparse task-completion reward semantics for
robosuite dataset processing.

Together with early termination on success, this makes the current Lift raw
return effectively binary at the trajectory level:

- success -> total raw return `1.0`
- failure within the horizon budget -> total raw return `0.0`

Under those conditions,
`mean_true_raw_return == true_rollout_success_rate`.

### 4.4 When They Would Diverge

These two metrics would stop matching if any of the following changed:

- the environment used dense rewards
- successful trajectories could collect reward for multiple post-success steps
- rollouts continued after success instead of terminating immediately
- reward magnitude depended on more than final task completion
