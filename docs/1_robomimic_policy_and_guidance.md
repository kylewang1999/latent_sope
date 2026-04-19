# Robomimic Policy, Guidance, Visual Embeddings, And Training-Time Eval Semantics

Relevant code:

- [data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json](../data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json)
- [third_party/sope/opelab/core/baselines/diffusion/helpers.py](../third_party/sope/opelab/core/baselines/diffusion/helpers.py)
- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py)
- [third_party/robomimic/robomimic/config/diffusion_policy_config.py](../third_party/robomimic/robomimic/config/diffusion_policy_config.py)
- [third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)
- [third_party/robomimic/robomimic/models/obs_core.py](../third_party/robomimic/robomimic/models/obs_core.py)
- [third_party/robomimic/robomimic/models/obs_nets.py](../third_party/robomimic/robomimic/models/obs_nets.py)
- [third_party/robomimic/robomimic/models/base_nets.py](../third_party/robomimic/robomimic/models/base_nets.py)
- [third_party/robomimic/robomimic/utils/file_utils.py](../third_party/robomimic/robomimic/utils/file_utils.py)
- [third_party/robomimic/robomimic/utils/obs_utils.py](../third_party/robomimic/robomimic/utils/obs_utils.py)
- [third_party/robomimic/robomimic/utils/tensor_utils.py](../third_party/robomimic/robomimic/utils/tensor_utils.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/eval.py](../src/eval.py)
- [src/sampling.py](../src/sampling.py)
- [src/train.py](../src/train.py)
- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py)
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
- [src/robomimic_interface/post_process_robomimic_h5.py](../src/robomimic_interface/post_process_robomimic_h5.py)
- [scripts/train_sope.py](../scripts/train_sope.py)
- [scripts/test_ope_guided.py](../scripts/test_ope_guided.py)

## 1. Summary

Robomimic's diffusion policy is trained as an epsilon predictor over action
sequences, not as a tractable density model with a native
$\log \pi(a \mid s)$ interface. In the current repository:

- the usable guidance signal is the denoising score implied by that
epsilon-prediction parameterization
- the local sampler expects diffusion-style policies exposing
`grad_log_prob(state, action)`
- the local SOPE wrapper uses FiLM-style conditioning, not upstream
in-painting-style conditioning
- the default Lift MH image policy concatenates `19` low-dimensional features
with `128` visual features into one per-step observation feature with width
`147`
- with `observation_horizon = 2`, robomimic flattens that history into
`obs_cond` with width `294`
- `feat_type="both"` stores the one-step `obs_encoder` output, not the flattened
history vector consumed by the U-Net
- the postprocessed visual-policy latent corpus preserves the low-dimensional
prefix, so `robot0_eef_pos` remains addressable at slice `[10:13]`
- training-time held-out chunk evaluation logs only unguided metrics under
`eval_metrics/*` and `eval_diagnostics/*`; guided checkpoint-backed evaluation
lives in `[scripts/test_ope_guided.py](../scripts/test_ope_guided.py)`

## 2. What Robomimic Actually Trains

Robomimic's diffusion policy:

1. encodes the observation history
2. samples Gaussian noise and a diffusion timestep
3. corrupts clean action chunks with the scheduler
4. predicts the noise with `noise_pred_net`
5. minimizes MSE against the sampled noise

The model is therefore a sequence model over action chunks. The score used for
local guidance is the usual DDPM score approximation:

$$\begin{align}
\nabla_{a_t} \log p_t(a_t \mid s)
\approx

- \frac{\hat{\epsilon}_\theta(a_t, t, s)}{\sqrt{1 - \bar{\alpha}_t}}.
\end{align}$$

## 3. Local Guidance Contract

The local SOPE-style sampler does not ask robomimic for an exact likelihood.
Instead it requires a policy adapter with the contract

$$\begin{align}
\texttt{policy.gradlogprob(states, actions)}
\rightarrow
\frac{\partial \log p(\text{actions} \mid \text{states})}
{\partial \text{actions}}.
\end{align}$$

In the current repo that means:

- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py)
reconstructs the robomimic policy internals and exposes `grad_log_prob(...)`
- [src/sampling.py](../src/sampling.py) assumes both target and behavior
policies follow that same diffusion-policy contract
- [src/diffusion.py](../src/diffusion.py) exposes the local FiLM guidance knobs
`action_score_postprocess`, `action_neg_score_weight`, and `clamp_linf`
- guidance edits only action channels; the local sampler does not treat the
policy as a scorer over state coordinates

The current local chunk contract is:

- `states`: `[B, H, Ds]`
- `actions`: `[B, H, Da]`
- `grad_log_prob(states, actions)`: `[B, H, Da]`

Current caveat:

- the adapter still uses a fixed score timestep rather than the active chunk
sampler timestep, so guidance remains an approximation even though it is now
wired end to end for diffusion-policy adapters

### 3.1 Local Adapter Config Surface

`[DiffusionPolicyScoreConfig](../src/robomimic_interface/policy.py)` now keeps
only the fixed `score_timestep` knob.

The older local `repeat_single_state_to_horizon` branch has been removed. When
the adapter receives a single-step encoded observation with shape
`[B, Dobs_feat]`, it now always repeats that feature vector across the
robomimic observation horizon before flattening to `obs_cond`.

This keeps the local adapter contract smaller and matches the only mode that
was actually being used in the repository.

### 3.2 Parameterization Compatibility

The chunk diffuser and the guidance policy are separate diffusion contracts. In
particular:

- chunk-side and policy-side parameterizations do not need to match
- the chunk diffuser always uses its own beta schedule, posterior mean path, and
posterior-variance schedule
- changing chunk-side `predict_epsilon` changes the denoiser training target and
the interpretation of the chunk reverse mean, but it does not change the
Gaussian noise scale injected by the chunk sampler
- changing policy-side parameterization changes only the score-conversion
formula used inside `grad_log_prob(state, action)`

The current robomimic adapter implements only the epsilon-prediction score
conversion. So chunk-side `predict_epsilon=False` is supported as long as the
chunk model remains internally consistent, but policy-side
`predict_epsilon=False` would require replacing the epsilon-based score
conversion with the corresponding `predict-x0` form.

### 3.3 Action-Score Postprocess Surface

[`guided_sample_step`](../src/sampling.py) treats the policy-score API and
action-score postprocessing as one explicit local contract instead of encoding
those choices through older upstream-derived switches.

The public FiLM sampling API under
[`FilmGaussianDiffusion.conditional_sample`](../src/diffusion.py) now uses:

- `action_score_scale`
- `action_score_postprocess`
- `action_neg_score_weight`
- `clamp_linf`

Inside [`prepare_guidance`](../src/sampling.py), the sampler retrieves raw
action-only scores on chunk tensors and then applies one of three local
heuristics independently at each chunk timestep:

- `"none"`: keep the raw score unchanged
- `"l2"`: L2-normalize each per-timestep action-score vector
- `"clamp"`: clip both target and behavior scores to `[-clamp_linf, clamp_linf]`

The final local guide is

$$\begin{align}
\text{guide}
=
\text{postprocess}(g_{\pi})
-
\text{action\_neg\_score\_weight}\,\text{postprocess}(g_{\beta}),
\end{align}$$

when negative guidance is enabled, and just the postprocessed target score
otherwise.

## 4. Horizon Alignment And Checkpoint Semantics

### 4.1 Observation-Horizon Alignment

The local FiLM chunk diffuser conditions on `states_from`, which contains
exactly `frame_stack` prefix states from the dataset contract in
[`RolloutChunkDataset`](../src/robomimic_interface/dataset.py). The robomimic
guidance adapter interprets action-score queries using its own
`observation_horizon`. Allowing those horizons to differ would silently mix two
incompatible sequence contracts.

`SopeDiffuser` therefore requires:

- `cfg.frame_stack == policy.observation_horizon`

### 4.2 Checkpoint Semantics

It is important not to overstate what is "baked into" a robomimic diffusion
policy checkpoint.

Robomimic restores `prediction_horizon` from the serialized checkpoint config,
not from weight shapes alone. In the standard load path,
[`config_from_checkpoint(...)`](../third_party/robomimic/robomimic/utils/file_utils.py)
parses `ckpt_dict["config"]`, and
[`policy_from_checkpoint(...)`](../third_party/robomimic/robomimic/utils/file_utils.py)
rebuilds the policy from that config before deserializing the saved weights.

`DiffusionPolicyConfig` also ties the training dataset contract to that same
window by setting `train.seq_length` to match `prediction_horizon`. In the
default inference helper
[`DiffusionPolicyUNet._get_action_trajectory(...)`](../third_party/robomimic/robomimic/algo/diffusion_policy.py),
robomimic then allocates the denoised action sample with shape
`[B, T_p, D_a]` using the restored `prediction_horizon`.

This distinction matters because:

- `observation_horizon` controls the conditioning width
  `$[B, T_o, D_{\text{obs}}] \rightarrow [B, T_o D_{\text{obs}}]$`
- `prediction_horizon` controls the denoised action-sequence shape
  `$[B, T_p, D_a]$`
- the robomimic `ConditionalUnet1D` is convolutional over the time axis and
  receives the runtime sample horizon, so raw weight tensors alone **do not**
  reliably identify `$T_p$`

### 4.3 What If Inference Uses $T_p' \neq T_p$?

In the standard checkpoint-restoration path, robomimic does not expose a
different inference horizon automatically: `_get_action_trajectory(...)`
constructs its Gaussian sample using the restored `prediction_horizon = T_p`.
So asking for some different $T_p'$ means either editing the loaded config or
bypassing that helper and calling the denoiser on a custom-length action tensor.

If $T_p' < T_p$, the policy is asked to denoise a shorter action window than it
saw during training. This truncates the temporal support the denoiser was
optimized for. If $T_p' \leqslant T_o$, there is no future-action region after
the observation prefix at all. If $T_p' < T_o - 1 + T_a$, robomimic's rollout
slicer can no longer return the full configured `action_horizon`. In the local
SOPE adapter, exact whole-chunk scoring uses $T_p = T_o + H$, so if
$T_p' < T_o + H$ only the first $T_p' - T_o$ future actions remain inside the
exact scoring contract.

If $T_p' > T_p$, the denoiser is being asked for action positions beyond the
training horizon. Because `ConditionalUnet1D` is convolutional over time and
its conditioning width depends on $T_o$ rather than $T_p$, the forward pass can
sometimes be run at a longer horizon. But those suffix positions were never
directly supervised during training, so this is extrapolation rather than
interpolation. For the local SOPE scorer, any positions beyond $T_o + H$ also
sit outside the intended exact chunk window unless they are explicitly dropped.

There is also a separate architectural issue: the current stride-2 downsample /
upsample stack is not perfectly horizon-agnostic. In a local shape probe of the
shipped `ConditionalUnet1D`, horizons `8`, `12`, `16`, and `20` round-tripped
cleanly; `11`, `15`, and `19` rounded up to `12`, `16`, and `20`; and `9`,
`10`, `13`, `14`, `17`, and `18` failed with skip-connection size mismatches.
So arbitrary $T_p'$ is not supported even before considering the train-test
distribution shift from changing the action-window length.

### 4.4 Scheduler Device Handling

`DiffusionPolicy.grad_log_prob(...)` converts robomimic's predicted epsilon
into a score by reading `noise_scheduler.alphas_cumprod` at the configured
guidance timestep.

That scheduler tensor is not part of the robomimic module tree, so moving the
policy network to CUDA does not guarantee that `alphas_cumprod` moves with it.
The adapter therefore materializes `alphas_cumprod` on the current action or
denoiser device before gathering the per-query $\bar{\alpha}_t$ values.

Without this device move, guided runs can fail on CUDA with a device-mismatch
error when indexing a CPU scheduler tensor using a CUDA timestep tensor.

## 5. FiLM Conditioning Path

### 5.1 Why The Backbone Is FiLM-Style

Robomimic's `ConditionalUnet1D` does not instantiate a class literally called
`FiLMLayer`, but its conditioning path is still FiLM-style.

The timestep embedding and optional observation-conditioning vector are
concatenated into one global conditioning vector:

$$\begin{align}
\text{global\_feature}
\in
\mathbb{R}^{B \times \text{cond\_dim}},
\qquad
\text{cond\_dim}
=
\text{diffusion\_step\_embed\_dim}
+
\text{global\_cond\_dim}.
\end{align}$$

Each conditioned residual block maps that vector into per-channel scale and bias
parameters that modulate intermediate convolution features. This is why the
local diffusion wrapper treats robomimic as the canonical FiLM-conditioned
backbone.

### 5.2 Contrast With Upstream SOPE Conditioning

Upstream SOPE uses in-painting-style conditioning: `apply_conditioning(...)`
overwrites selected trajectory entries inside the sampled tensor, and the
conditioning is re-applied during reverse sampling.

The local wrapper instead passes the conditioning prefix as an external FiLM
context to the denoiser. So the local implementation should be described as
SOPE-inspired rather than identical to upstream SOPE conditioning.

## 6. Visual Policy Observation Embeddings

### 6.1 Configured Observation Streams And Widths

The policy at
`data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json` encodes
both configured camera streams and concatenates their features with the
configured low-dimensional observations into one per-timestep observation
feature.

The configured observation group `obs` includes these low-dimensional keys:

- `robot0_eef_pos`
- `robot0_eef_quat`
- `robot0_gripper_qpos`
- `object`

It also includes these RGB keys:

- `agentview_image`
- `robot0_eye_in_hand_image`

Both RGB keys use the same `VisualCore` encoder configuration with
`feature_dimension = 64`. Therefore the visual-only part of one per-timestep
observation feature is:

$$\begin{align}
D_{\text{visual}}
= D_{\text{agentview}} + D_{\text{eye-in-hand}}
= 64 + 64
= 128.
\end{align}$$

For the local Lift image policy, the raw low-dimensional observation shapes
are:

- `object`: `(10,)`
- `robot0_eef_pos`: `(3,)`
- `robot0_eef_quat`: `(4,)`
- `robot0_gripper_qpos`: `(2,)`

Robomimic computes shape metadata after observation preprocessing. RGB keys are
converted from HWC to CHW, while low-dimensional keys keep their raw shapes.
For this policy, the per-timestep low-dimensional width is therefore:

$$\begin{align}
D_{\text{low-dim}}
&= 10 + 3 + 4 + 2 \\
&= 19.
\end{align}$$

This gives the concrete Lift image policy widths:

$$\begin{align}
D_{\text{obs}}
&= D_{\text{low-dim}} + D_{\text{agentview}} + D_{\text{eye-in-hand}} \\
&= 19 + 64 + 64 \\
&= 147, \\
T_o D_{\text{obs}}
&= 2 \cdot 147 \\
&= 294.
\end{align}$$

Do not assume $D_{\text{obs}} = 2D_z$ unless the low-dimensional keys are
explicitly excluded and $D_z$ refers to one camera stream. Verify these
dimensions from checkpoint shape metadata or a reconstructed encoder before
treating them as a hard contract in another experiment.

### 6.2 Encoder And Conditioning Path

Robomimic constructs `nets["policy"]["obs_encoder"]` as an
`ObservationGroupEncoder`. It computes `obs_dim = obs_encoder.output_shape()[0]`
and creates `ConditionalUnet1D` with `global_cond_dim = obs_dim * observation_horizon`.

At inference time, `DiffusionPolicyUNet._get_action_trajectory(...)` performs
the path below:

1. prepare observation tensors through the rollout wrapper
2. add a time axis when a single observation is supplied
3. run `TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)`
4. receive `obs_features` with shape `[B, T_o, D_{\text{obs}}]`
5. flatten to `obs_cond` with shape `[B, T_o D_{\text{obs}}]`
6. pass `obs_cond` to `nets["policy"]["noise_pred_net"](..., global_cond=obs_cond)`

The key tensor boundary is:

$$\begin{align}
\texttt{obsfeatures} &\in \mathbb{R}^{B \times T_o \times D_{\text{obs}}}, 
\texttt{obscond}
&= \texttt{flatten}(\texttt{obsfeatures}, \texttt{startdim}=1) 
&\in \mathbb{R}^{B \times (T_o D_{\text{obs}})}.
\end{align}$$

The diffusion-policy U-Net therefore does not receive separate camera tensors
or a second multi-modal fusion block. It receives one flattened observation
history vector as global FiLM conditioning.

### 6.3 Shape Metadata, Key Ordering, And Fusion

Robomimic records processed observation shapes in
`FileUtils.get_shape_metadata_from_dataset(...)`. It iterates through
`sorted(all_obs_keys)`, computes the processed shape for each key with
`ObsUtils.get_processed_shape(...)`, and stores the result in
`shape_meta["all_shapes"]`.

For the local Lift image policy, this sorting yields the per-key concatenation
order:

1. `agentview_image`
2. `object`
3. `robot0_eef_pos`
4. `robot0_eef_quat`
5. `robot0_eye_in_hand_image`
6. `robot0_gripper_qpos`

`ObservationGroupEncoder` is built from `OrderedDict(self.obs_shapes)`, so this
order propagates into the final feature concatenation unless a different
`obs_shapes` ordering is injected upstream.

Inside `ObservationEncoder.forward(...)`, each observation key is processed in
configured order. RGB keys pass through their `VisualCore`; low-dimensional
keys are flattened directly; all per-key features are concatenated into the
single per-timestep feature:

$$\begin{align}
\texttt{feat}
= \texttt{torch.cat}([\texttt{feat}_1, \ldots, \texttt{feat}_K], \texttt{dim}=-1).
\end{align}$$

This is the exact place where:

- multiple camera embeddings are concatenated with each other
- low-dimensional state is concatenated with the camera embeddings

There is no extra fusion block between `ObservationEncoder` and
`ConditionalUnet1D`.

### 6.4 Per-Key Encoder Construction And Training Contract

`DiffusionPolicyUNet._create_networks()` constructs one
`ObservationGroupEncoder` for the `"obs"` group. Internally,
`obs_encoder_factory(...)` creates one `ObservationEncoder`, and then
`ObservationEncoder.register_obs_key(...)` registers each observation key with:

- its processed input shape
- a per-key randomizer stack, if configured
- a per-key encoder core class, if configured

For `low_dim`, `core_class = null`, so the key is flattened directly.

For `rgb`, `core_class = "VisualCore"`, so each camera key gets its own
`VisualCore` instance. In the local Lift image checkpoint, the `rgb` encoder
config uses:

- `backbone_class = "ResNet18Conv"`
- `pool_class = "SpatialSoftmax"`
- `feature_dimension = 64`

So one camera path is:

$$\begin{align}
\texttt{image}*{HWC}
\rightarrow \texttt{image}*{CHW}
\rightarrow \texttt{ResNet18Conv}
\rightarrow \texttt{SpatialSoftmax}
\rightarrow \texttt{Flatten}
\rightarrow \texttt{Linear}(64).
\end{align}$$

During standard diffusion-policy BC training:

- the observation encoder lives inside the `policy` module together with
`noise_pred_net`
- the forward pass runs through `obs_encoder` before the denoiser
- the optimizer is built over the enclosing `policy` module

So for the default configuration, the visual encoder is trained jointly with
the diffusion U-Net.

This is not universal. Pretrained backbones such as `R3MConv` and `MVPConv`
expose a `freeze` flag and can remain frozen even though the outer robomimic
policy is still optimized.

## 7. Rollout Feature Extraction And Postprocessed Latents

### 7.1 Implemented Rollout Feature Modes

Rollout feature extraction in
[src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
supports three modes:

- `low_dim_concat`: concatenate the current-frame prepared low-dimensional
observation keys in robomimic config order
- `image_embedding`: concatenate the current-frame embeddings for all RGB keys
in robomimic config order
- `both`: extract the current-frame output of the full policy observation
encoder

For a feature corresponding to the one-step parent of the U-Net conditioning
input, use `feat_type="both"`. This extracts the output of
`nets["policy"]["obs_encoder"]` before robomimic flattens the history into
`[B, T_o D_{\text{obs}}]`.

The visual-latent strategy in the `Celina` branch is appropriate for a narrower
question: extracting a single camera stream's visual feature from
`obs_encoder.obs_nets[rgb_key]`. By default it picks one RGB key, so it does
not produce the full policy conditioning feature unless it is extended to:

- capture both `agentview_image` and `robot0_eye_in_hand_image`
- preserve robomimic's per-key concatenation order
- include the low-dimensional features if the target is the full
$D_{\text{obs}}$ feature

### 7.2 Current-Frame Selection And EMA Caveats

The rollout environment can return frame-stacked observations. In that case,
`_prepare_observation(...)` produces tensors with shape `[B, frame_stack, ...]`.
Robomimic's `TensorUtils.time_distributed(...)` applies the observation encoder
per frame and restores the feature tensor as `[B, frame_stack, D]`.

The rollout recorder stores one current-step feature per environment step. It
therefore selects the newest frame slot:

$$\begin{align}
z_t = \texttt{obsfeatures}_{[:, -1, :]}.
\end{align}$$

This keeps saved rollout latents shaped `[T, D_z]` for all three feature
modes.

The image policy has EMA enabled. During `_get_action_trajectory(...)`,
robomimic uses `self.ema.averaged_model` when it exists. A policy-consistent
feature hook should therefore attach to the same `obs_encoder` instance used by
the rollout path, not necessarily only `policy.nets.policy.obs_encoder`.

Diffusion policies also maintain an action queue. With `action_horizon > 1`,
the policy may not run its encoder at every environment step. A recorder that
needs one feature per environment step should force-run the observation encoder
on the current observation, as `PolicyFeatureHook.update_latent_from_obs(...)`
does in [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py).

### 7.3 Rollout-Free Dataset Conversion

The rollout-free converter in
[src/robomimic_interface/post_process_robomimic_h5.py](../src/robomimic_interface/post_process_robomimic_h5.py)
reads robomimic `image_v15.hdf5` demonstrations directly and writes one
`RolloutLatentTrajectory` file per source demo. It does not construct a
robomimic environment or step the diffusion policy.

The converter still requires a trained policy checkpoint. The dataset contains
raw RGB observations, but the learned `VisualCore` weights that define the
policy image embedding live in the checkpoint. In this setup, "rollout-free"
means no environment rollout, not checkpoint-free.

The converter supports the same feature modes as rollout collection:

- `low_dim_concat`: current-frame low-dimensional observations
- `image_embedding`: current-frame embeddings for all configured RGB cameras
- `both`: current-frame full `obs_encoder` output

For the local Lift MH image checkpoint, running the converter with
`feat_type="both"` and the default output-path inference writes these
one-step observation features under:

`data/robomimic/lift/mh/postprocessed_for_ope/both`

Frame-stack windows are built with robomimic `SequenceDataset` padding
semantics. The saved latent remains one current-frame feature per dataset step:
`[T, D_z]`. For `feat_type="both"`, this means the converter stores the
one-step observation-encoder feature with width $D_{\text{obs}}$, not the
flattened history vector with width $T_o D_{\text{obs}}$ consumed by the
diffusion U-Net.

The converter also validates `--feat-type` against checkpoint `config.json`.
For the local Lift image policy, the checkpoint uses both low-dimensional and
RGB modalities, so the allowed feature mode is `both`.

### 7.4 Visual-Policy EEF Slice In SOPE Chunk Diffusion

The default robomimic MH image policy config keeps low-dimensional state in the
observation stream even though the policy also uses RGB inputs. For the
postprocessed SOPE training corpus under
`data/robomimic/lift/mh/postprocessed_for_ope/both`, the saved latent width is
`147`, matching `19` low-dimensional features plus `128` visual features.

The converted `.h5` files also store the raw low-dimensional observation arrays
in their `obs/` group, and the real file ordering used by
[scripts/train_sope.py](../scripts/train_sope.py) is:

1. `object`: width `10`
2. `robot0_eef_pos`: width `3`
3. `robot0_eef_quat`: width `4`
4. `robot0_gripper_qpos`: width `2`

So the effective `robot0_eef_pos` slice is still `[10:13]` inside the
low-dimensional prefix of the `147`-wide latent. This matches both:

- `[scripts/train_sope.py::_infer_eef_pos_slice](../scripts/train_sope.py)`
- `[SopeDiffuser._resolve_eef_pos_slice](../src/diffusion.py)`

The visual embedding does not remove or reorder the low-dimensional prefix in
the saved current-step feature used by chunk diffusion.

## 8. Training-Time Evaluation Semantics

### 8.1 Logging Namespaces

Training-time evaluation is orchestrated through
[src/eval.py](../src/eval.py). The SOPE training loop calls
`evaluate_sope(...)`, which in turn calls
`evaluate_diffusion_chunk_mse(..., evaluate_guided=False)`.

The training loop therefore logs only unguided held-out chunk metrics under:

- `eval_metrics/`*
- `eval_diagnostics/*`

The older `eval_metrics:unguided/*`, `eval_diagnostics:unguided/*`, and
placeholder guided namespaces overstated what the training loop can actually
evaluate and are no longer part of the current contract.

### 8.2 Separation From Guided Checkpoint-Backed Evaluation

Guided sampling needs an additional target-policy checkpoint that
[scripts/train_sope.py](../scripts/train_sope.py) does not resolve at training
time.

So guided trajectory sampling and checkpoint-backed OPE remain separate offline
evaluation paths owned by
[scripts/test_ope_guided.py](../scripts/test_ope_guided.py), rather than part
of the training loop's held-out evaluation summary.

## 9. Validation

The smallest meaningful checks for this part of the stack are:

1. reconstruct a robomimic `PolicyAlgo` from checkpoint artifacts and confirm
  access to EMA weights, `noise_pred_net`, `obs_encoder`, and scheduler state
2. print `obs_encoder.output_shape()`, the captured one-step feature shape, and
  the final `obs_cond` shape passed to `ConditionalUnet1D`
3. verify that `grad_log_prob(...)` returns the expected action-gradient shape
4. run one guided and one unguided chunk sample and confirm only action
   coordinates receive explicit score edits
5. run `source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 -m py_compile src/diffusion.py src/sampling.py src/robomimic_interface/policy.py src/robomimic_interface/dataset.py src/eval.py scripts/train_sope.py`
6. run `source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 -c 'from scripts.train_sope import _infer_eef_pos_slice; print(_infer_eef_pos_slice("data/robomimic/lift/mh/postprocessed_for_ope/both"))'` and confirm that it prints `(10, 13)`
7. if you plan to test $T_p' \neq T_p$, run a one-batch `ConditionalUnet1D` or `noise_pred_net` shape probe at that horizon first and confirm that the output horizon matches the requested input horizon
