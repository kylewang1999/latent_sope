# SOPE Reward Predictor

This note documents how the vendored SOPE implementation trains its per-step
reward predictor, what target it learns, and where that logic is implemented in
this repository.

The relevant code is in the vendored SOPE tree under
[`third_party/sope/opelab/`](../third_party/sope/opelab/).

## 1. Short Answer

SOPE trains a transition-level regressor for immediate reward, not return-to-go
and not terminal success.

Given an offline dataset of transitions
$\{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$, the reward predictor is trained on

$$\begin{align}
x_i &= [s_i; a_i], \\
y_i &= r_i,
\end{align}$$

using mean-squared error:

$$\begin{align}
\mathcal{L}(\phi)
=
\frac{1}{B}\sum_{i=1}^{B}
\left(\hat{r}_\phi(s_i, a_i) - r_i\right)^2.
\end{align}$$

So the learned model is

$$\begin{align}
\hat{r}_\phi(s, a) \approx r(s, a),
\end{align}$$

where $r(s, a)$ is the immediate per-step reward from the offline dataset.

## 2. Where Training Happens

### 2.1 Training entrypoint

The reward predictor is instantiated and trained from
[`third_party/sope/opelab/examples/helpers.py`](../third_party/sope/opelab/examples/helpers.py).

In `evaluate_policies(...)`, SOPE:

1. loads an offline dataset
2. calls `Data(env).train_reward_estimator(offline_data)`
3. passes the resulting `reward_estimator` into each baseline's `evaluate(...)`

Relevant call site:

- [`third_party/sope/opelab/examples/helpers.py`](../third_party/sope/opelab/examples/helpers.py)

This means reward-model training happens once per evaluation run, before the
baseline estimates are computed.

### 2.2 Dataset-to-supervision conversion

The actual extraction of training examples happens in
[`third_party/sope/opelab/core/data.py`](../third_party/sope/opelab/core/data.py).

`Data.train_reward_estimator(...)` iterates over each trajectory `tau` in the
offline dataset and over each timestep `i` in that trajectory, then builds:

$$\begin{align}
x_i &= [s_i; a_i], \\
y_i &= r_i.
\end{align}$$

Concretely:

- `states = tao["states"]`
- `actions = tao["actions"]`
- `rewards = tao["rewards"]`
- each input is `np.concatenate([states[i], actions[i]])`
- each target is `rewards[i]`

Important implication:

- `next_state` is loaded into the dataset structure, but it is **not** used to
  train the reward predictor
- the predictor is supervised directly from per-transition reward labels
- this matches the STITCH-OPE description of learning an immediate reward model
  from behavior transitions

## 3. Model Architecture And Optimization

The model configuration is split across
[`third_party/sope/opelab/core/data.py`](../third_party/sope/opelab/core/data.py)
and
[`third_party/sope/opelab/core/reward.py`](../third_party/sope/opelab/core/reward.py).

### 3.1 Architecture

`Data.train_reward_estimator(...)` constructs

$$\begin{align}
\texttt{MLP([64, 64, 1])},
\end{align}$$

so the default reward predictor is a 2-hidden-layer MLP with scalar output.

### 3.2 Optimizer and objective

`RewardEstimator` in
[`third_party/sope/opelab/core/reward.py`](../third_party/sope/opelab/core/reward.py)
uses:

- Adam optimizer
- learning rate `1e-3`
- batch size `64` for `RewardEstimator`
- MSE loss on immediate reward

The core loss implementation is:

$$\begin{align}
\mathcal{L}(\phi)
=
\mathbb{E}\left[
\left(\hat{r}_\phi(s, a) - r\right)^2
\right].
\end{align}$$

In the current code, training is stochastic minibatch optimization for a fixed
number of iterations:

- `RewardEnsembleEstimator.fit(..., iters=1000, seeds=[42])` is called from
  `Data.train_reward_estimator(...)`
- each training step samples a minibatch by index with replacement

## 4. Ensemble Structure

SOPE wraps the single reward regressor in `RewardEnsembleEstimator`, defined in
[`third_party/sope/opelab/core/reward.py`](../third_party/sope/opelab/core/reward.py).

Its intended structure is:

- train multiple bootstrap models
- each model sees a random subsample of the offline transitions
- predictions are stacked so downstream code can aggregate across models

The ensemble API is:

$$\begin{align}
\hat{r}(s, a)
=
\left[
\hat{r}_{\phi_1}(s, a), \dots, \hat{r}_{\phi_M}(s, a)
\right].
\end{align}$$

However, in the current repository state, `Data.train_reward_estimator(...)`
calls:

```python
est.fit(rewards_x, rewards_y, 1000, [42])
```

with the default `n_bootstraps=1`.

So despite the ensemble wrapper, the current default behavior is effectively a
single reward regressor trained on a 50% random subsample of the transition
dataset.

## 5. How The Predictor Is Used During OPE

The learned reward predictor is consumed in
[`third_party/sope/opelab/core/baselines/diffuser.py`](../third_party/sope/opelab/core/baselines/diffuser.py).

After the diffusion model generates synthetic trajectories, `Diffuser.evaluate`
loops over each generated timestep:

1. extract `state_t`
2. extract `action_t`
3. compute reward either from an environment-specific `reward_fn`, if one is
   available, or from `reward_estimator.predict([state_t; action_t])`
4. discount and accumulate

In equations, when no analytic `reward_fn` is available, SOPE forms the
synthetic return as

$$\begin{align}
\hat{G}
=
\sum_{t=0}^{T_i-1}
\gamma^t
\hat{r}_\phi(s_t, a_t).
\end{align}$$

This is the exact role of the reward predictor in SOPE:

- it scores already-generated trajectories
- it does **not** guide the diffusion sampler
- it predicts immediate reward, not long-horizon value

## 6. Data Assumptions

The current SOPE reward-predictor path assumes:

- the offline dataset already contains per-step rewards
- those rewards are aligned with the `(state, action)` pairs in each trajectory
- immediate reward is learnable directly from `[s; a]`

That matters for sparse-reward tasks. If positive rewards are rare, the current
MSE regressor is still trained on the same sparse labels:

$$\begin{align}
y_i \in \{0, 1\}
\end{align}$$

or similarly sparse continuous targets, depending on the environment.

The code does not currently add:

- class reweighting
- focal losses
- return-to-go targets
- terminal-only supervision

So for sparse tasks, SOPE is relying on plain supervised regression on sparse
per-transition rewards.

## 7. Repo-Specific Distinction: SOPE Vs `rei`

This repository also contains a separate reward-scoring path in
[`rei/src/latent_sope/eval/reward_model.py`](../rei/src/latent_sope/eval/reward_model.py).

That file is **not** the vendored SOPE implementation. It provides:

- `LiftRewardFn`, an analytical per-step reward for robosuite Lift
- `RewardMLP`, a PyTorch reward regressor used in Rei-specific experiments

So there are two reward-model stories in this repo:

- **SOPE reward predictor:** JAX MLP in `third_party/sope/opelab/core/reward.py`,
  trained through `core/data.py`
- **Rei reward scoring utilities:** PyTorch / analytical code in
  `rei/src/latent_sope/eval/reward_model.py`

If the question is specifically "how does SOPE train the per-step reward
predictor?", the authoritative implementation is the vendored SOPE path in
`third_party/sope/opelab/core/{data,reward}.py`.

## 8. Robomimic Lift Reward Predictor In `main`

The main worktree also now contains a repository-specific reward predictor for
robomimic Lift rollout data. This path is still SOPE-style in the sense that it
learns immediate reward from raw transition inputs:

$$\begin{align}
\hat{r}_\phi(s_t, a_t) \approx \tilde{r}_t.
\end{align}$$

For the rollout corpus under
`data/rollout/rmimic-lift-ph-lowdim_diffusion_260130`, the ground-truth label
is read directly from the saved rollout file field `rewards`.

The important repository-specific change is that the default target is shifted:

$$\begin{align}
\tilde{r}_t = T(s_t, r_t) = r_t - 1.
\end{align}$$

So raw reward `0` becomes `-1` and raw reward `1` becomes `0`.

### 8.1 Label distribution

On the current robomimic Lift rollout corpus:

- there are `300` rollout files
- there are `13,305` transition steps total
- rewards are binary, with values in `{0, 1}`
- `297` steps are positive, so the positive-step fraction is about `2.23%`
- `297 / 300` trajectories contain exactly one positive reward
- every positive reward occurs on the final step of its trajectory

So the stored reward behaves like a terminal-success label rather than a dense
shaped reward. The default shift $\tilde{r}_t = r_t - 1$ preserves immediate
reward regression while making cumulative predicted reward more informative for
later OPE-style scoring.

### 8.2 State alignment

The rollout `.h5` files store low-dim observations with a frame-stack axis of
length `2`, ordered from oldest to newest. For reward prediction, the current
state must therefore come from the newest frame:

$$\begin{align}
s_t = \texttt{latents}[t, -1, :].
\end{align}$$

The dataset path in `main` now uses `latents[:, -1, :]` rather than
`latents[:, 0, :]` when collapsing low-dim frame stacks for reward
supervision.

### 8.3 Code path

The repository-specific reward predictor is implemented in:

- [`src/robomimic_interface/dataset.py`](../src/robomimic_interface/dataset.py):
  newest-frame low-dim collapse plus `rewards_to_raw` and transformed
  `rewards_to`
- [`src/diffusion.py`](../src/diffusion.py):
  `RewardPredictorConfig` and `RewardPredictor`
- [`src/train.py`](../src/train.py):
  `train_rewardpred(...)` as the canonical reward entrypoint, plus
  compatibility aliases `train_reward(...)`,
  `train_rewardpred_with_loaders(...)`, and `train_reward_with_loaders(...)`
- [`scripts/train_sope.py`](../scripts/train_sope.py):
  separate reward hyperparameters, where reward `lr` comes from
  `RewardPredictorConfig.lr` via `--reward-lr`, while reward `epochs` and
  `batch_size` come from the reward-side `TrainingConfig` via
  `--reward-epochs` and `--reward-batch-size`
- [`src/eval.py`](../src/eval.py):
  `load_reward_checkpoint(...)`

## 9. Practical Summary

SOPE's reward predictor in this repo is:

- supervised on offline transition tuples
- trained on concatenated `[state, action]`
- fit to immediate reward labels `r_t`
- implemented as `MLP([64, 64, 1])`
- optimized with Adam and MSE
- wrapped in a bootstrap-style ensemble class, but currently used with one
  bootstrap by default
- consumed only during synthetic trajectory scoring when no hand-coded reward
  function is available

The robomimic Lift reward predictor in `main` keeps the same immediate-reward
regression structure, but it uses rollout-file rewards, a default shifted
target $\tilde{r}_t = r_t - 1$, and the newest frame in the saved low-dim
stack as the current state.

## 9. Dense Lift Reward Follow-Up

For robomimic / robosuite Lift, a useful adjacent question is what shaped reward
to use if we later want denser supervision for reward modeling, OPE, or value
estimation.

The most defensible first version is a staged dense reward rather than a purely
sparse success flag:

- `reach`: reward increases as the gripper gets closer to the cube
- `grasp`: bonus when the gripper has a stable grasp
- `lift`: bonus when the cube has been lifted above the table

At a high level:

$$\begin{align}
r_t
&=
w_{\text{reach}} \, r_{\text{reach}}(s_t)
+ w_{\text{grasp}} \, r_{\text{grasp}}(s_t)
+ w_{\text{lift}} \, r_{\text{lift}}(s_t).
\end{align}$$

A simple smooth starting point is

$$\begin{align}
r_t
&=
\alpha \exp\!\left(-\beta \lVert p_{\text{gripper}} - p_{\text{cube}} \rVert \right)
+ \gamma \, g_t
+ \delta \, \mathrm{clip}(h_t - h_0, 0, h_{\max}),
\end{align}$$

where $p_{\text{gripper}}$ is gripper position, $p_{\text{cube}}$ is cube
position, $g_t$ is a grasp indicator or soft grasp score, and $h_t - h_0$
measures lift progress above the initial cube height.

This matters for the reward-predictor story because SOPE's current regressor
only learns from whatever per-step rewards the offline dataset already carries.
If the dataset reward is too sparse to be useful, a shaped lift reward is the
most direct next target to define before changing the model architecture.

Practical recommendation:

1. start with `reach + grasp + lift`
2. keep the reward computable from the stored state or observation surface
3. check whether the shaped signal aligns with actual task success on saved
   trajectories
4. only then consider training a learned reward model against that shaped target
