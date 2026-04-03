# Dense Reward Design for Lift Task

Date: 2026-04-03

## Motivation

The current `LiftRewardFn` is purely sparse: $r_t = \mathbf{1}[z_{\text{cube},t} > 0.84]$. This gives no signal for near-miss trajectories and makes it hard to rank synthetic trajectories that approach but don't complete the lift. A dense reward would be useful in two ways:

1. As a more informative scoring function for evaluating SOPE-generated trajectories
2. As supervision for training a learned per-step reward model (MLP fallback)

## Available Observations

The dense reward must be computable from the stored latent vector (19-dim, `LowDimConcatEncoder`). The relevant fields are:

| Index | Obs Key | Dim | Content |
|-------|---------|-----|---------|
| 0--2 | `object` | 3 | $p_{\text{cube}} = (x, y, z)$ |
| 3--6 | `object` | 4 | cube quaternion |
| 7--9 | `object` | 3 | $\Delta p = p_{\text{gripper}} - p_{\text{cube}}$ (displacement) |
| 10--12 | `robot0_eef_pos` | 3 | $p_{\text{gripper}} = (x, y, z)$ |
| 13--16 | `robot0_eef_quat` | 4 | end-effector quaternion |
| 17--18 | `robot0_gripper_qpos` | 2 | finger joint positions $(q_1, q_2)$ |

Notably, we do **not** have contact forces or grasp success flags. Grasp must be inferred from proximity + gripper closure.

## Reward Structure

The dense reward follows a standard **staged** design with three components:

$$
r_t = w_{\text{reach}} \, r_{\text{reach}}(s_t) + w_{\text{grasp}} \, r_{\text{grasp}}(s_t) + w_{\text{lift}} \, r_{\text{lift}}(s_t)
$$

where each component is bounded in $[0, 1]$ and the weights satisfy $w_{\text{reach}} + w_{\text{grasp}} + w_{\text{lift}} = 1$, so $r_t \in [0, 1]$.

### Component 1: Reach

Measures how close the gripper is to the cube. We have the displacement vector $\Delta p_t = p_{\text{gripper},t} - p_{\text{cube},t}$ stored directly in `object[7:10]`, so the distance is:

$$
d_t = \lVert \Delta p_t \rVert_2
$$

The reach reward uses an exponential kernel:

$$
r_{\text{reach}}(s_t) = \exp(-\beta_{\text{reach}} \, d_t)
$$

This gives $r_{\text{reach}} = 1$ when the gripper is at the cube and decays smoothly with distance. The scale parameter $\beta_{\text{reach}}$ controls the decay rate. With $\beta_{\text{reach}} = 10$, the reward is $\approx 0.37$ at $d = 0.1\text{m}$ and $\approx 0.05$ at $d = 0.3\text{m}$.

**Alternative**: $r_{\text{reach}} = 1 - \tanh(\beta_{\text{reach}} \, d_t)$, which is also bounded in $[0, 1]$ and has similar shape. The exponential is slightly simpler.

### Component 2: Grasp

Detects whether the gripper has closed around the cube. Since we lack contact/force sensors, we use a **proximity $\times$ gripper-closed** heuristic. Both sub-conditions use soft sigmoids to keep the reward differentiable:

$$
r_{\text{grasp}}(s_t) = \sigma\!\left(\frac{-(q_{1,t} + q_{2,t}) + \tau_{\text{close}}}{\epsilon_q}\right) \cdot \sigma\!\left(\frac{-d_t + \tau_{\text{prox}}}{\epsilon_d}\right)
$$

where:

- $q_{1,t}, q_{2,t}$ are the two finger joint positions (`gripper_qpos`)
- $q_{1,t} + q_{2,t}$ is the total gripper opening (larger = more open)
- $\tau_{\text{close}}$ is the gripper closure threshold
- $\tau_{\text{prox}}$ is the proximity threshold for "close enough to grasp"
- $\epsilon_q, \epsilon_d$ are temperature parameters controlling sigmoid sharpness
- $\sigma(x) = 1 / (1 + e^{-x})$ is the logistic sigmoid

The first sigmoid fires when the gripper is closed ($q_1 + q_2 < \tau_{\text{close}}$). The second fires when the gripper is near the cube ($d_t < \tau_{\text{prox}}$). Both must be satisfied for a high grasp score.

**Design choice -- soft vs binary**: We use soft sigmoids rather than hard indicators. Binary grasp creates a discontinuity that makes near-miss trajectories (gripper almost closed, or almost touching) indistinguishable from complete misses. Since the whole point of dense reward is to rank near-misses sensibly, soft is preferred.

**Threshold calibration needed**: The values of $\tau_{\text{close}}$ and $\tau_{\text{prox}}$ depend on the actual range of `gripper_qpos` in rollout data. This must be checked empirically before implementation. Tentative values: $\tau_{\text{close}} = 0.04$, $\tau_{\text{prox}} = 0.02\text{m}$, $\epsilon_q = 0.01$, $\epsilon_d = 0.005$.

### Component 3: Lift

Rewards upward cube motion above the table surface. Uses linear interpolation with clipping:

$$
r_{\text{lift}}(s_t) = \text{clip}\!\left(\frac{z_{\text{cube},t} - h_0}{h_{\text{scale}}},\; 0,\; 1\right)
$$

where:

- $z_{\text{cube},t}$ is the cube height (index 2 of `object`)
- $h_0 = 0.8\text{m}$ is the table height
- $h_{\text{scale}} = 0.1\text{m}$ is the normalization range (reward saturates at $z = 0.9\text{m}$)

This means:
- Cube on table ($z \approx 0.82$): $r_{\text{lift}} \approx 0.2$
- Cube at success threshold ($z = 0.84$): $r_{\text{lift}} = 0.4$
- Cube well above table ($z = 0.9$): $r_{\text{lift}} = 1.0$

**Design choice -- what baseline height?** Three options were considered:

1. **Height above table** ($z - 0.8$): Simple, consistent across trajectories. Chosen.
2. **Height above initial cube position**: Per-trajectory dependent, adds complexity for little gain since the cube always starts at the same height ($\approx 0.8208\text{m}$).
3. **Height above success threshold** ($z - 0.84$): This is just a shifted sparse reward and defeats the purpose of going dense.

**Alternative form**: An exponential lift term $r_{\text{lift}} = 1 - \exp(-\beta_{\text{lift}} \max(z - h_0, 0))$ would also work. The linear-clip version is simpler and more interpretable.

## Proposed Weights

The lift component should dominate since it represents the actual task objective. Reaching and grasping are intermediate sub-goals that enable lifting.

| Weight | Value | Rationale |
|--------|-------|-----------|
| $w_{\text{reach}}$ | 0.3 | Provides signal early in the trajectory |
| $w_{\text{grasp}}$ | 0.3 | Rewards gripper engagement with cube |
| $w_{\text{lift}}$ | 0.4 | Largest weight for the terminal objective |

These are starting values. The weights should be tuned so that:

- A trajectory that reaches but doesn't grasp scores higher than one that doesn't reach
- A trajectory that grasps but doesn't lift scores higher than one that only reaches
- A trajectory that lifts to any degree scores highest

This monotonicity is guaranteed as long as $w_{\text{lift}} > 0$ and the components activate in the correct order (reach $\to$ grasp $\to$ lift).

## Full Reward Equation

Combining all components:

$$
r_t = 0.3 \exp(-10 \, d_t) + 0.3 \, \sigma\!\left(\frac{-G_t + 0.04}{0.01}\right) \sigma\!\left(\frac{-d_t + 0.02}{0.005}\right) + 0.4 \, \text{clip}\!\left(\frac{z_{\text{cube},t} - 0.8}{0.1}, 0, 1\right)
$$

where $d_t = \lVert \Delta p_t \rVert_2$ and $G_t = q_{1,t} + q_{2,t}$.

## Properties

- **Bounded**: $r_t \in [0, 1]$ by construction.
- **Monotone with task progress**: reaching $\to$ grasping $\to$ lifting produces strictly increasing reward.
- **Smooth**: All components are continuous and differentiable (no hard thresholds).
- **Computable from stored observations**: Uses only `object[0:10]`, `robot0_gripper_qpos[0:2]`. No simulator access needed.
- **Compatible with existing pipeline**: Follows the `GroundTruthRewardFn` protocol (takes `obs_dict`, returns `(T,)` rewards). Drop-in replacement for `LiftRewardFn`.

## Open Items Before Implementation

1. **Calibrate gripper_qpos range**: Inspect actual rollout data to determine what values correspond to open vs closed gripper. This sets $\tau_{\text{close}}$ and $\epsilon_q$.
2. **Validate reward ordering**: On a few real trajectories (successful and failed lifts), verify that the dense reward orders them correctly: lift > grasp-only > reach-only > miss.
3. **Check interaction with the obs recording bug**: The known bug (rollout recorder drops the final success state) means cube_z maxes out at $\approx 0.838$ in recorded data. The dense reward should still give meaningful signal since it rewards intermediate progress, but the lift component will be capped below its theoretical maximum for successful trajectories.
4. **Decide on use case scope**: Is this for evaluation scoring only, or also as MLP training labels? If the latter, we should verify the reward signal is learnable (i.e., the MLP can approximate it from $(z, a)$ pairs).

## References

- Prior discussion: https://chatgpt.com/share/69c5c6de-fe4c-83e8-bfa4-5a9d255a3435
- Current sparse reward: `src/latent_sope/eval/reward_model.py:43-77` (`LiftRewardFn`)
- Latent vector layout: CLAUDE.md, "Latent Vector Layout" table
- Robosuite Lift reward semantics: `cube_z > table_height + 0.04`
