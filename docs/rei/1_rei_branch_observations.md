# Rei Branch Observations

This note summarizes what the `rei/` worktree appears to implement, relative to
the two current goals described in
[`docs/1_robomimic_diffusion_score_guidance.md`](../1_robomimic_diffusion_score_guidance.md):

1. implement guided diffusion where the target policy is a robomimic diffusion policy
2. estimate target-policy value by generating synthetic "rollouts" with SOPE and scoring them

The summary below is based on observed source-code differences between `main`
and `Rei`, plus the experiment filenames and commit messages in `rei/`.

## High-Level Takeaway

`rei/` contains meaningful prototype code toward both goals, but mostly as
adapter, rollout, scoring, and evaluation infrastructure around SOPE and
robomimic. It does **not** appear to finish the end-to-end guided-diffusion
integration described in the main design note.

In particular:

- the branch adds a robomimic diffusion scorer adapter
- the branch adds synthetic full-trajectory generation and trajectory scoring utilities
- the branch adds oracle and OPE evaluation helpers
- the branch does **not** modify `third_party/robomimic` or `third_party/sope`
- the branch still leaves the deepest integration gaps unresolved: timestep plumbing, sequence-shape alignment inside SOPE's guidance path, and a single reusable end-to-end value-estimation entrypoint

## Meaningful Code Toward The Two Goals

### 1. Robomimic Diffusion Guidance Adapter

Relevant code:

- [`rei/src/latent_sope/robomimic_interface/guidance.py`](../../../rei/src/latent_sope/robomimic_interface/guidance.py)

What it does:

- Implements `RobomimicDiffusionScorer`, a wrapper around robomimic's diffusion policy.
- Reconstructs robomimic observation conditioning from flat low-dimensional states.
- Builds a coherent action sequence matching robomimic's temporal UNet interface.
- Converts `noise_pred_net` output into a score-like `grad_log_prob(...)` signal.

Why it matters:

- This is the clearest code in `rei/` aimed directly at goal 1.
- It is the branch's attempt to make a robomimic diffusion policy look like the kind of policy-guidance object SOPE expects.

Observed limitations:

- It uses a fixed `score_timestep=1` instead of plumbing the current SOPE diffusion timestep through the adapter.
- It is a prototype in `src/`, but there is no evidence in `src/` that SOPE's internal guidance path was modified to call it in the exact sequence-level way described in the main note.
- Because `third_party/sope` is unchanged, the shape-alignment issue discussed in the main note still appears unresolved at the library boundary.

### 2. SOPE Chunk Sampling And Full-Trajectory Stitching

Relevant code:

- [`rei/src/latent_sope/diffusion/sope_diffuser.py`](../../../rei/src/latent_sope/diffusion/sope_diffuser.py)

What it does:

- Fixes the SOPE chunk-sampling shape to use the full chunk horizon.
- Adds `generate_full_trajectory(...)`, which autoregressively stitches chunk samples into longer synthetic trajectories from initial states.
- Threads a `guided` flag and guidance kwargs into the SOPE diffusion sampler.

Why it matters:

- This is the main code toward goal 2, because synthetic value estimation needs generated trajectories, not just one-step chunk samples.
- It is also the place where guided sampling would need to be wired into reusable code.

Observed limitations:

- The file still contains explicit TODOs saying guided diffusion is not wired end to end.
- Termination handling and real `end_indices` tracking are still missing.
- There is no single reusable `estimate_value(...)` function built on top of this sampler.

### 2a. Conditioning And Normalization Contract In `rei/src`

Relevant code:

- [`rei/src/latent_sope/robomimic_interface/dataset.py`](../../../rei/src/latent_sope/robomimic_interface/dataset.py)
- [`rei/src/latent_sope/diffusion/sope_diffuser.py`](../../../rei/src/latent_sope/diffusion/sope_diffuser.py)
- [`rei/src/latent_sope/diffusion/train.py`](../../../rei/src/latent_sope/diffusion/train.py)
- [`rei/src/latent_sope/eval/metrics.py`](../../../rei/src/latent_sope/eval/metrics.py)

What the reusable path does:

- `RolloutChunkDataset` slices each rollout into `states_from`,
  `actions_from`, `states_to`, and `actions_to`
- `SopeDiffuser.loss(...)` concatenates prefix and future chunks into one
  training trajectory
- `make_cond(...)` conditions only on prefix states, not on prefix actions

So the reusable `rei/src` path is best described as:

- state-conditioned chunk diffusion
- with prefixed action history carried in the target tensor rather than in the
  hard conditioning dict

Observed normalization issue in the branch:

- `make_rollout_chunk_dataloader(...)` computes one shared normalization over
  `[state, action]` transitions
- `states_from`, `states_to`, and `actions_to` are normalized
- `actions_from` is left in raw units

That means the reusable branch path mixes normalized state prefixes with raw
action prefixes inside the training target, which is a real contract mismatch.

The reusable branch sampling path still relies on normalization functions for:

- initial-state normalization
- sampled-chunk unnormalization
- carrying the next conditioning state forward in normalized space

### 2b. Notebook-Specific Conditioning In `v0.2.5.14`

The `v0.2.5.14` notebook family does not use the reusable `rei/src` path
described above.

Instead, those notebooks:

- instantiate `TemporalUnet` and `GaussianDiffusion` directly
- build notebook-local `normalize_fn` and `unnormalize_fn`
- condition on only one state at one timestep
- update that single-state condition autoregressively after each chunk

The conditioning loop is therefore much simpler than the reusable
frame-stacked path. In pseudocode, it looks like:

```python
# initial state in raw state space
state0 = initial_state

# move into notebook-local normalized diffusion space
cond_init = normalize_fn(concat(state0, zeros_action))[:, :state_dim]
conditions = {0: cond_init}

generated_chunks = []
for chunk_idx in range(num_chunks):
    sample = diffusion_model.conditional_sample(
        shape=(batch_size, horizon, state_dim + action_dim),
        cond=conditions,
        guided=guided,
        ...
    )

    # sample.trajectories is still in normalized diffusion space
    chunk_norm = sample.trajectories
    chunk_raw = unnormalize_fn(chunk_norm)
    generated_chunks.append(chunk_raw)

    # carry forward only the final generated state, not a frame stack
    last_state_norm = chunk_norm[:, -1, :state_dim]
    conditions = {0: last_state_norm}
```

Conceptually, the notebook path is:

- initialize with one normalized state at diffusion timestep index `0`
- generate one chunk
- take the last generated state from that chunk
- reuse that last state as the only condition for the next chunk

So the effective contract is:

```python
cond = {0: current_state}
```

not:

```python
cond = {
    0: prefix_state_0,
    1: prefix_state_1,
    ...,
    S - 1: prefix_state_{S-1},
}
```

and not:

```python
cond = {
    t: (state_t, action_t)
}
```

This matters because the notebook path is really "single-state autoregressive
chunk stitching" rather than "state-conditioned chunk diffusion with a
frame-stacked history."

So conclusions from the `v0.2.5.14` experiments should be interpreted as
evidence about that notebook-local sampling procedure, not about the reusable
branch pipeline as a whole.

### 3. Rollout Collection From Robomimic Policies

Relevant code:

- [`rei/src/latent_sope/robomimic_interface/collect.py`](../../../rei/src/latent_sope/robomimic_interface/collect.py)
- [`rei/src/latent_sope/robomimic_interface/rollout.py`](../../../rei/src/latent_sope/robomimic_interface/rollout.py)
- [`rei/src/latent_sope/robomimic_interface/checkpoints.py`](../../../rei/src/latent_sope/robomimic_interface/checkpoints.py)

What it does:

- Collects offline rollout datasets from robomimic checkpoints into `.h5` files.
- Extracts latent features during rollouts using the existing feature-hook machinery.
- Adds a terminal-observation recording fix so the final success state is not dropped.
- Adds an environment-construction fix needed before env reset and observation handling.

Why it matters:

- These pieces support both goals by making the data path from robomimic policy to SOPE training data more reliable.
- The terminal-state fix is especially important because losing the success state would systematically bias both training data and synthetic-return estimates.

### 4. Oracle Value, Reward Scoring, And OPE Metrics

Relevant code:

- [`rei/src/latent_sope/eval/oracle.py`](../../../rei/src/latent_sope/eval/oracle.py)
- [`rei/src/latent_sope/eval/reward_model.py`](../../../rei/src/latent_sope/eval/reward_model.py)
- [`rei/src/latent_sope/eval/metrics.py`](../../../rei/src/latent_sope/eval/metrics.py)

What it does:

- Adds on-policy oracle evaluation for robomimic checkpoints.
- Adds trajectory-based oracle computation from pre-collected trajectories.
- Adds a simple Lift ground-truth reward function and a learned reward MLP for scoring synthetic trajectories.
- Adds reconstruction metrics and OPE metrics such as per-policy error, rank correlation, and `regret@k`.

Why it matters:

- This is the evaluation layer needed for goal 2.
- Even without a finished guided sampler, these utilities let the branch compare synthetic returns against oracle values and debug whether generated trajectories are useful for OPE.

Observed limitation:

- This is value **scoring** and value **evaluation** infrastructure, not a finished value-estimation pipeline driven by a fully integrated guided sampler.

## What `rei/` Does Not Seem To Implement Yet

Relative to the main design note, the following still appear missing in reusable code:

- a modification to SOPE's internal guidance call so it passes observation windows, action windows, and the current diffusion timestep into the robomimic adapter
- a fully resolved normalized-action-space guidance path
- a single end-to-end entrypoint that performs guided synthetic rollout generation for a robomimic diffusion target policy and returns the resulting OPE estimate

So the right characterization is:

- `rei/` implements important building blocks
- `rei/` does not yet appear to complete the final integration

## What Rei's Experiments Seem To Be Investigating

From the notebook names and commit messages, the experiments appear to be moving
through several stages.

### Stage A: Can latent SOPE reproduce basic synthetic trajectories at all?

Representative experiment names:

- `latent_sope_5_rollouts`
- `latent_sope_50_rollouts`
- `8.5k_steps`
- `500k_steps`
- `8.5k_steps_no_quarternions`

What these seem to investigate:

- whether the chunk diffuser can reconstruct trajectories from collected robomimic rollouts
- how much data is needed
- whether longer training helps
- whether certain state features, such as quaternions, hurt reconstruction quality

### Stage B: Establish an unguided baseline before policy guidance

Representative experiment names:

- `MVP_v0.1_sope_on_robomimic`

What this seems to investigate:

- whether SOPE-style synthetic rollout generation behaves sensibly on robomimic data with no guidance
- what reconstruction or return-estimation baseline should be expected before adding target-policy steering

### Stage C: Can positive or negative guidance actually steer synthetic trajectories?

Representative experiment names:

- `MVP_v0.2_sope_on_robomimic`
- `MVP_v0.2.1_guidance_ablations`
- `MVP_v0.2.2_diffusion_score_guidance`
- `MVP_v0.2.3_negative_guidance`
- `MVP_v0.2.5.4_scorer_timestep_sweep`
- `MVP_v0.2.5.5_corrected_guidance`

What these seem to investigate:

- whether robomimic diffusion-score guidance can push synthetic trajectories toward stronger target policies
- whether negative guidance helps separate target and behavior effects
- which guidance scale and timestep regime are stable and meaningful
- whether observed failures come from incorrect gradient scale, wrong timestep choice, or a deeper mismatch between SOPE and robomimic diffusion policies

### Stage D: Is the diffuser training distribution itself the bottleneck?

Representative experiment names:

- `MVP_v0.2.4_target_data_diffuser`
- `MVP_v0.3.2.3_medium_behavior_data`
- `MVP_v0.3.2.4_pooled_behavior_data`

What these seem to investigate:

- whether training the chunk diffuser only on expert-like data makes guidance ineffective
- whether the synthetic model needs broader-quality behavior data or target-policy rollouts to support meaningful steering
- whether data mixture changes synthetic rollout quality and OPE error more than guidance alone

### Stage E: Can the method rank multiple target policies correctly?

Representative experiment names:

- `oracle_eval_all_checkpoints`
- `oracle_eval_target_policies`
- `MVP_v0.3.1_sope_on_robomimic`
- `MVP_v0.3.2_multi-policy_positive_and_negative_test`
- `MVP_v0.3.2.1_multi-policy-rerun`
- `MVP_v0.3.2.2_multi-policy-guidance-fix`

What these seem to investigate:

- whether synthetic OPE can distinguish a range of target checkpoints instead of only one target policy
- whether the branch can recover the ranking of policies by true oracle success rate or return
- whether guidance helps enough to improve policy selection, not just single-policy return estimation

### Stage F: Why does guidance fail to differentiate policies?

Representative experiment names:

- `MVP_v0.2.5.2_trajectory_mse`
- `MVP_v0.2.5.3_scorer_debug`
- `MVP_v0.2.5.6_cross_policy_guidance`
- `MVP_v0.2.5.7_action_logprob_diagnostic`
- `MVP_v0.2.5.8_small_mlp_scorers`
- `MVP_v0.2.5.9_bc_gaussian_scorers`
- `MVP_v0.2.5.10_action_distribution_diagnostics`
- `addressing_action_diversity`
- `action_diversity_comparison`

What these seem to investigate:

- whether the scorer gradients for different policies are actually distinct
- whether guidance changes the sampled action distribution in a measurable way
- whether simpler scorers behave better than the robomimic diffusion-score proxy
- whether poor OPE performance is caused by low action diversity, weak gradients, or scorer miscalibration rather than by the chunk diffuser alone

### Stage G: Does the same debugging story hold outside Lift / robomimic?

Representative experiment names:

- `MVP_v0.2.5.11_d4rl_sope_calibration`
- `MVP_v0.2.5.12_d4rl_diagnostics_quick`
- `hopper_d4rl_diagnostics`
- `hopper_d4rl_diagnostics_guided`

What these seem to investigate:

- whether the calibration and guidance issues are specific to Lift
- whether a D4RL control setting exposes the same failure modes more clearly
- whether cross-policy ranking and guidance diagnostics transfer to a simpler benchmark

## Overall Read On Rei's Research Direction

At this stage, Rei's branch does **not** look like it is merely tuning scripts.
It is building the infrastructure needed to ask a focused research question:

Can SOPE-style guided diffusion be made to work when the target policy is a
robomimic diffusion policy, and if not, is the failure caused by scorer
construction, sampler integration, training-data support, or policy
indistinguishability?

## Follow-Up Questions

### 1. Why Are Quaternions Involved?

Quaternions are involved because Rei's Lift setup initially uses robomimic's
full low-dimensional observation and action interface, which includes rotation
information:

- the state encoder / scorer path reconstructs robomimic observations from the
  policy's full `obs_shapes`, so any default Lift low-dim observation keys in
  the checkpoint are carried into the flattened state representation
- the Lift observation layout includes `robot0_eef_quat` explicitly, and the
  `object` observation also contains `cube_quat`
- the Lift action space under `OSC_POSE` includes 3 orientation-control
  dimensions in addition to translation and gripper control

Relevant evidence:

- [`rei/src/latent_sope/robomimic_interface/guidance.py`](../../../rei/src/latent_sope/robomimic_interface/guidance.py) reconstructs states from `algo.obs_shapes`, so the scorer inherits robomimic's full low-dim observation layout by default.
- [`rei/src/latent_sope/eval/reward_model.py`](../../../rei/src/latent_sope/eval/reward_model.py) defines the default Lift obs keys as `["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]` and documents that `object[3:7]` is `cube_quat`.

So the short answer is: quaternions are not a SOPE requirement by themselves;
they appear because Rei initially kept the native robomimic Lift state/action
representation.

Why Rei then starts worrying about them:

- Rei's notes explicitly identify quaternions as a poor fit for vanilla
  diffusion over Euclidean coordinates because they live on a unit-norm
  manifold and have the sign symmetry $q$ and $-q$ representing the same
  rotation.
- In Rei's debugging notes, quaternion coordinates are singled out as blowing
  up badly during early chunk-generation experiments, which motivated the
  `8.5k_steps_no_quarternions` ablation.
- Rei later also notes that Lift reward / success is driven primarily by cube
  height, so quaternions may be unnecessary for the OPE target being debugged.

In equation form, the geometric issue is that a valid quaternion must satisfy

$$\begin{align}
\lVert q \rVert_2^2 = q_w^2 + q_x^2 + q_y^2 + q_z^2 = 1,
\end{align}$$

but the diffusion model samples in unconstrained Euclidean space
$\mathbb{R}^4$. That mismatch is exactly the concern described in Rei's
architecture and blow-up notes.

### 2. Is `small_mlp_scoreres` Implemented In Any `src/` Files?

Not in `rei/src/`. The name appears to refer to the experiment
`MVP_v0.2.5.8_small_mlp_scorers`, and the implementation lives in experiment
artifacts, not in reusable source modules.

What I found:

- `rei/src/` contains the reusable robomimic diffusion scorer
  `RobomimicDiffusionScorer` in
  [`rei/src/latent_sope/robomimic_interface/guidance.py`](../../../rei/src/latent_sope/robomimic_interface/guidance.py).
- `rei/src/` also contains `RewardMLP` in
  [`rei/src/latent_sope/eval/reward_model.py`](../../../rei/src/latent_sope/eval/reward_model.py), but that model is for reward prediction, not for policy guidance.
- The actual "small MLP scorer" experiment is documented in
  [`rei/results/2026-03-13/MVP_v0.2.5.8_small_mlp_scorers.md`](../../../rei/results/2026-03-13/MVP_v0.2.5.8_small_mlp_scorers.md) and generated from
  [`rei/experiments/2026-03-13/gen_v0258.py`](../../../rei/experiments/2026-03-13/gen_v0258.py), where Rei defines and trains a temporary small diffusion MLP scorer per policy.

So the precise answer is:

- `small_mlp_scorers` exists as an experiment-specific implementation
- it does **not** appear to be promoted into a reusable `rei/src/...` module
- therefore it should be treated as exploratory code, not branch-level source
  infrastructure

### 3. Important Policy-Eval Acronyms / Terms In Rei's Context

Below are the main policy-eval terms that show up in Rei's branch and notes.

#### Oracle

In Rei's code, the oracle is the ground-truth policy value computed by actually
running the target policy in the environment. This is implemented in
[`rei/src/latent_sope/eval/oracle.py`](../../../rei/src/latent_sope/eval/oracle.py).

If a policy is $\pi$, reward is $r_t$, discount is $\gamma$, and horizon is
$H$, then the oracle value is

$$\begin{align}
V^\pi
= \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{H-1} \gamma^t r_t\right].
\end{align}$$

In practice, Rei estimates this with on-policy Monte Carlo rollouts:

$$\begin{align}
\hat{V}_{\mathrm{oracle}}^\pi
= \frac{1}{K}\sum_{i=1}^{K}
\sum_{t=0}^{H-1} \gamma^t r_t^{(i)}.
\end{align}$$

That is exactly the "ground truth" object that `OracleResult.mean_return`
stores.

How the per-step reward is computed in `rei` depends on which oracle path is
used:

- **Default oracle path:** `oracle_value(...)` calls the robomimic / env
  rollout helper and uses `stats.total_reward` directly. In that path, the
  per-step rewards are whatever the environment emitted during the true on-policy
  rollout, but `rei` only keeps their episode sum unless a custom reward
  function is provided.
- **Re-scored oracle path:** if `oracle_value(...)` is called with a
  `reward_fn` and `encoder`, `rei` records the per-step observations along the
  on-policy rollout, decodes them into an observation dict, and then computes
  per-step rewards as `reward_fn(obs_dict)`. For Lift, the default analytical
  reward is

$$\begin{align}
r_t
=
\mathbf{1}\{z^{\text{cube}}_t > 0.8 + 0.04\},
\end{align}$$

up to the configurable `reward_scale`, because
`LiftRewardFn` checks whether the cube height exceeds the success threshold.

#### OPE

OPE means off-policy evaluation: estimate $V^\pi$ for a target policy $\pi$
without running $\pi$ online at evaluation time.

In Rei's SOPE-style setup, the branch tries to do this by:

1. generating synthetic trajectories intended to look like the target policy
2. scoring those trajectories with either a ground-truth reward function or a
   learned reward model
3. averaging the resulting synthetic returns

If $\tilde{\tau}^{(i)}$ are synthetic trajectories and
$\tilde{r}^{(i)}_t$ are their scored rewards, then Rei's OPE estimate is

$$\begin{align}
\hat{V}_{\mathrm{OPE}}^\pi
= \frac{1}{N}\sum_{i=1}^{N}
\sum_{t=0}^{H-1} \gamma^t \tilde{r}^{(i)}_t.
\end{align}$$

This is the quantity compared against the oracle in
[`rei/src/latent_sope/eval/metrics.py`](../../../rei/src/latent_sope/eval/metrics.py).

How the per-step reward is computed in the OPE regime is more explicit than in
the default oracle path, because the synthetic trajectories must be scored
after generation:

- **Ground-truth OPE scoring:** `score_trajectories_gt(...)` decodes each
  generated latent-state sequence back into an observation dict and applies an
  analytical reward function frame by frame. For Lift, this again means

$$\begin{align}
\tilde{r}_t
=
\mathbf{1}\{\tilde{z}^{\text{cube}}_t > 0.8 + 0.04\},
\end{align}$$

up to `reward_scale`.
- **Learned-reward OPE scoring:** `score_trajectories(...)` applies
  `RewardMLP` pointwise to each synthetic `(s_t, a_t)` pair, so the per-step
  reward is

$$\begin{align}
\tilde{r}_t
=
\hat{r}_\phi(\tilde{s}_t, \tilde{a}_t).
\end{align}$$

Then, in either case, the synthetic return is formed by discounting and
summing those per-step scores:

$$\begin{align}
\hat{G}
=
\sum_{t=0}^{H-1} \gamma^t \tilde{r}_t.
\end{align}$$

#### SOPE

SOPE here refers to the synthetic-rollout OPE setup: train a chunk-level
diffusion model on offline data, stitch sampled chunks into synthetic
trajectories, and score those trajectories to estimate value.

The core intuition is:

$$\begin{align}
\text{offline data} \rightarrow \text{trajectory model}
\rightarrow \text{synthetic trajectories} \rightarrow \hat{V}^\pi.
\end{align}$$

In Rei's branch, this is the broader framework around
`sope_diffuser.py`, oracle helpers, reward scoring, and OPE metrics.

#### Scorer

In Rei's context, a scorer is a policy-guidance object that provides a gradient
with respect to the action, not necessarily a normalized probability density.

For a target policy $\pi(a \mid s)$, the ideal guidance quantity is

$$\begin{align}
\nabla_a \log \pi(a \mid s).
\end{align}$$

For standard policies this might come from a tractable `log_prob`. For
robomimic diffusion policies, Rei approximates this using the diffusion
noise-prediction network near a clean timestep. The scorer file states the
approximation as

$$\begin{align}
\nabla_a \log \pi(a \mid s)
\approx -\frac{\hat{\varepsilon}_\theta(a, s, t)}{\sigma_t},
\end{align}$$

where $\hat{\varepsilon}_\theta$ is the predicted diffusion noise and
$\sigma_t = \sqrt{1 - \bar{\alpha}_t}$.

That is what `RobomimicDiffusionScorer` is trying to produce in
[`rei/src/latent_sope/robomimic_interface/guidance.py`](../../../rei/src/latent_sope/robomimic_interface/guidance.py).

#### Positive Guidance / Negative Guidance

Rei's notes use "positive guidance" to mean pushing sampled actions toward the
target-policy scorer, and "negative guidance" to mean subtracting a
behavior-policy gradient so the synthetic trajectory moves away from behavior
modes.

Schematically, the update looks like

$$\begin{align}
g(s, a)
= \nabla_a \log \pi_{\text{target}}(a \mid s)
- \lambda \nabla_a \log \pi_{\text{behavior}}(a \mid s),
\end{align}$$

followed by a scaled action-space update during diffusion sampling.

#### Reward Model

The reward model is separate from the scorer. It does **not** guide the
diffusion sampler toward a policy. Instead, it assigns rewards to already
generated trajectories so they can be turned into return estimates.

In `rei/src`, this is `RewardMLP` in
[`rei/src/latent_sope/eval/reward_model.py`](../../../rei/src/latent_sope/eval/reward_model.py),
which learns a map

$$\begin{align}
\hat{r}_\phi(s_t, a_t) \approx r_t.
\end{align}$$

The estimated return from a generated trajectory is then

$$\begin{align}
\hat{G}
= \sum_{t=0}^{H-1} \gamma^t \hat{r}_\phi(s_t, a_t).
\end{align}$$

#### Spearman / Regret@k

These are multi-policy OPE diagnostics in
[`rei/src/latent_sope/eval/metrics.py`](../../../rei/src/latent_sope/eval/metrics.py):

- Spearman $\rho$ checks whether the ordering induced by OPE agrees with the
  ordering induced by oracle values.
- `regret@k` measures how much true value is lost if one selects policies using
  OPE rankings instead of oracle rankings.

Rei's code defines

$$\begin{align}
\text{regret@}k
= \frac{1}{k}\sum_{i \in \text{TopK}_{\mathrm{oracle}}} V_i
- \frac{1}{k}\sum_{i \in \text{TopK}_{\mathrm{ope}}} V_i.
\end{align}$$

#### One Useful Distinction

Rei's branch uses two different learned objects that are easy to conflate:

- **policy scorer**: provides $\nabla_a \log \pi(a \mid s)$ for guidance
- **reward model**: provides $\hat{r}(s, a)$ for scoring generated trajectories

Those are different roles, different equations, and different files.

The experiments appear less about polishing a final system and more about
identifying which of those bottlenecks is the dominant one.
