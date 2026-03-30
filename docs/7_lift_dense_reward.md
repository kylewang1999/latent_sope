# Dense Reward For Lift

Relevant context:
- Shared discussion: <https://chatgpt.com/share/69c5c6de-fe4c-83e8-bfa4-5a9d255a3435>

This note summarizes the discussion about how one could construct a dense
reward for the robomimic / robosuite Panda lift task, especially if we later
want a shaped reward for reward modeling, OPE, or value-estimation work.

## Baseline View

The standard dense reward design for cube lift is usually **staged** instead of
fully sparse:

- `reach`: reward increases as the gripper gets closer to the cube
- `grasp`: bonus when the gripper has a stable grasp
- `lift`: bonus when the cube has been lifted above the table

This is the practical template that people usually mean when they say the lift
task has a dense reward. The sparse version keeps only the terminal lift
success, while the dense version adds intermediate shaping.

At a high level, the staged reward can be written as

$$\begin{align}
r_t
&=
w_{\text{reach}} \, r_{\text{reach}}(s_t)
+ w_{\text{grasp}} \, r_{\text{grasp}}(s_t)
+ w_{\text{lift}} \, r_{\text{lift}}(s_t).
\end{align}$$

The important point is not the exact constants. The important point is that the
reward is **monotone with task progress** and gives the policy signal before the
final success event.

## What To Use As A Starting Point

If we want a reward that stays close to the usual robosuite semantics, the most
defensible first version is:

1. A smooth distance-to-cube term for reaching.
2. A binary or soft contact term for grasping.
3. A height-above-initial-height term for lifting.

That gives a simple shaping family such as

$$\begin{align}
r_t
&=
\alpha \exp\!\left(-\beta \lVert p_{\text{gripper}} - p_{\text{cube}} \rVert \right)
+ \gamma \, g_t
+ \delta \, \mathrm{clip}(h_t - h_0, 0, h_{\max}),
\end{align}$$

where:

- $p_{\text{gripper}}$ is gripper position
- $p_{\text{cube}}$ is cube position
- $g_t$ is a grasp indicator or soft grasp score
- $h_t$ is the cube height at time $t$
- $h_0$ is the initial cube height

This is not meant to claim an exact robosuite formula. It is a smooth version
of the same staged logic.

## Why This Matters For Our Setting

For our use case, a shaped reward can be useful in at least two ways:

- as supervision for a learned per-step reward model
- as a more informative target when evaluating sampled trajectories from the
  SOPE diffuser

The dense reward should therefore satisfy a few practical constraints:

- it should be computable from the state or observation representation we
  actually store
- it should not depend on privileged simulator internals that disappear in the
  offline pipeline unless we explicitly reconstruct them
- it should increase smoothly enough that near-miss trajectories are ranked
  sensibly

## Recommended Progression

The discussion suggests the following progression instead of jumping straight to
an elaborate learned reward:

1. Start with the standard staged template: `reach + grasp + lift`.
2. Make the reach and lift terms smooth, but keep grasp simple at first.
3. Check whether the resulting signal is aligned with actual task success on
   saved lift trajectories.
4. Only then consider training a reward MLP on top of that supervision or using
   it inside downstream OPE.

## Open Design Choices

There are still a few unresolved choices we should make explicitly before
implementation:

- Whether grasp should stay binary or become a soft contact / enclosure score.
- Whether the lift term should reward raw height, height above initial cube
  height, or height above a success threshold margin.
- Whether the final reward should be clipped to stay non-negative and bounded.
- Whether we want the shaped reward only for evaluation, or also as a target
  for a learned reward model.

## Practical Recommendation

If we need a dense lift reward soon, the best near-term choice is:

- `reach`: smooth exponential of gripper-to-cube distance
- `grasp`: binary contact / grasp bonus
- `lift`: positive clipped height-above-initial-height term

That is simple, interpretable, and close to the task-structure described in the
discussion, while still being smoother than a pure success-only reward.
