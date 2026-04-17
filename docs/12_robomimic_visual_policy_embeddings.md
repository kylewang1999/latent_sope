# Robomimic Visual Policy Observation Embeddings

This note records how the local robomimic Lift image diffusion policy builds
the per-timestep observation feature that becomes the conditional input to the
diffusion-policy U-Net.

Relevant code and artifacts:

- [data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json](../data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json)
- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py)
- [third_party/robomimic/robomimic/utils/file_utils.py](../third_party/robomimic/robomimic/utils/file_utils.py)
- [third_party/robomimic/robomimic/utils/train_utils.py](../third_party/robomimic/robomimic/utils/train_utils.py)
- [third_party/robomimic/robomimic/utils/tensor_utils.py](../third_party/robomimic/robomimic/utils/tensor_utils.py)
- [third_party/robomimic/robomimic/utils/obs_utils.py](../third_party/robomimic/robomimic/utils/obs_utils.py)
- [third_party/robomimic/robomimic/models/base_nets.py](../third_party/robomimic/robomimic/models/base_nets.py)
- [third_party/robomimic/robomimic/models/obs_core.py](../third_party/robomimic/robomimic/models/obs_core.py)
- [third_party/robomimic/robomimic/models/obs_nets.py](../third_party/robomimic/robomimic/models/obs_nets.py)
- [third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)
- [third_party/robomimic/robomimic/algo/algo.py](../third_party/robomimic/robomimic/algo/algo.py)
- [third_party/robomimic/robomimic/utils/dataset.py](../third_party/robomimic/robomimic/utils/dataset.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
- [src/robomimic_interface/post_process_robomimic_h5.py](../src/robomimic_interface/post_process_robomimic_h5.py)

## 1. Summary

The policy at
`data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json` encodes
both configured camera streams and concatenates their features with the
configured low-dimensional observations into one per-timestep observation
feature.

Each camera key is processed by its own `VisualCore` instance. Each low-dimensional
key is passed through without a learned per-key encoder and is flattened directly.
The fusion mechanism is late concatenation inside
`ObservationEncoder.forward(...)`, not early channel stacking across cameras.

The diffusion-policy U-Net does not receive separate camera tensors or a second
multi-modal encoder stage. It receives one flattened history vector
`obs_cond`, and this vector is used as global FiLM conditioning inside
`ConditionalUnet1D`. The key tensor boundary is:

$$\begin{align}
\texttt{obs\_features} &\in \mathbb{R}^{B \times T_o \times D_{\text{obs}}}, \\
\texttt{obs\_cond} &= \texttt{flatten}(\texttt{obs\_features}, \texttt{start\_dim}=1)
                  \in \mathbb{R}^{B \times (T_o D_{\text{obs}})}.
\end{align}$$

For this policy, `observation_horizon = 2`, so the U-Net conditioning vector is
two per-timestep observation features concatenated along the feature axis.

## 2. Configured Observation Streams

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

The full per-timestep feature also includes the low-dimensional observation
keys:

$$\begin{align}
D_{\text{obs}}
= D_{\text{low-dim}} + D_{\text{agentview}} + D_{\text{eye-in-hand}}.
\end{align}$$

For the local Lift image v15 dataset, the raw observation shapes are:

- `agentview_image`: `(84, 84, 3)`
- `robot0_eye_in_hand_image`: `(84, 84, 3)`
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

## 3. Encoder And Conditioning Path

Robomimic constructs `nets["policy"]["obs_encoder"]` as an
`ObservationGroupEncoder`. It computes `obs_dim = obs_encoder.output_shape()[0]`
and creates `ConditionalUnet1D` with `global_cond_dim = obs_dim *
observation_horizon`.

At inference time, `DiffusionPolicyUNet._get_action_trajectory(...)` performs
the path below:

1. prepare observation tensors through the rollout wrapper
2. add a time axis when a single observation is supplied
3. run `TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)`
4. receive `obs_features` with shape `[B, T_o, Dobs]`
5. flatten to `obs_cond` with shape `[B, T_o * Dobs]`
6. pass `obs_cond` to `nets["policy"]["noise_pred_net"](..., global_cond=obs_cond)`

Inside `ObservationEncoder.forward(...)`, each observation key is processed in
configured order. RGB keys pass through their `VisualCore`; low-dimensional
keys are flattened directly; all per-key features are concatenated into the
single per-timestep feature.

### 3.1. Observation Preprocessing

There are two robomimic paths that prepare observations before they reach the
encoder:

1. Training batches go through
   `Algo.postprocess_batch_for_training(...)`, which applies
   `ObsUtils.process_obs_dict(...)` on `obs`, `next_obs`, and `goal_obs`.
2. Rollout and checkpoint inference go through
   `RolloutPolicy._prepare_observation(...)`, which applies
   `ObsUtils.process_obs(...)` to visual keys.

For RGB observations, `ObsUtils.process_obs(...)` calls
`ImageModality._default_obs_processor(...)`, which:

- converts uint8 images to float
- scales pixels from `[0, 255]` into `[0, 1]`
- converts image layout from HWC to CHW

This is the boundary that turns raw robomimic HDF5 image observations into the
tensor layout expected by `VisualCore`.

### 3.2. Shape Metadata And Key Ordering

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

### 3.3. Per-Key Encoder Construction

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
\texttt{image}_{HWC}
\rightarrow \texttt{image}_{CHW}
\rightarrow \texttt{ResNet18Conv}
\rightarrow \texttt{SpatialSoftmax}
\rightarrow \texttt{Flatten}
\rightarrow \texttt{Linear}(64).
\end{align}$$

If an RGB randomizer such as `CropRandomizer` is enabled, it wraps the visual
core as `forward_in(...) -> visual net -> forward_out(...)`. The local Lift
checkpoint leaves `obs_randomizer_class = null`, so the image path is the
direct `VisualCore` path above.

### 3.4. Concatenation Across Cameras And Low-Dimensional State

`ObservationEncoder.forward(...)` iterates through `self.obs_shapes` in
configured order. For each key it applies:

1. `randomizer.forward_in(...)`, if present
2. the per-key network, if present
3. the encoder activation
4. `randomizer.forward_out(...)`, if present
5. flatten to `[B, D_k]`

It then concatenates all per-key features with:

$$\begin{align}
\texttt{feat}
= \texttt{torch.cat}([\texttt{feat}_1, \ldots, \texttt{feat}_K], \texttt{dim}=-1).
\end{align}$$

This is the exact place where:

- multiple camera embeddings are concatenated with each other
- low-dimensional state is concatenated with the camera embeddings

There is no extra fusion block between `ObservationEncoder` and
`ConditionalUnet1D`.

### 3.5. Time Flattening And U-Net Conditioning

After per-timestep encoding, robomimic applies
`TensorUtils.time_distributed(...)` to the observation encoder, so tensors with
leading shape `[B, T_o, ...]` are reshaped to `[B T_o, ...]`, encoded, and then
restored as:

$$\begin{align}
\texttt{obs\_features} \in \mathbb{R}^{B \times T_o \times D_{\text{obs}}}.
\end{align}$$

`DiffusionPolicyUNet` then flattens the time axis:

$$\begin{align}
\texttt{obs\_cond}
= \texttt{obs\_features.reshape}(B, T_o D_{\text{obs}})
\in \mathbb{R}^{B \times (T_o D_{\text{obs}})}.
\end{align}$$

`ConditionalUnet1D` receives:

- `sample`: the noisy action trajectory with shape `[B, T_p, D_a]`
- `global_cond`: the flattened observation history with shape
  `[B, T_o D_{\text{obs}}]`

Inside `ConditionalUnet1D.forward(...)`, `global_cond` is concatenated with the
diffusion-step embedding and used to FiLM-modulate every residual block. The
observation history is therefore global conditioning, not an extra channel
dimension appended to the action sequence.

## 4. Implemented Rollout Feature Modes

Rollout feature extraction in
[src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
supports three modes:

- `low_dim_concat`: concatenate the current-frame prepared low-dimensional
  observation keys in robomimic config order.
- `image_embedding`: concatenate the current-frame embeddings for all RGB keys
  in robomimic config order.
- `both`: extract the current-frame output of the full policy observation
  encoder. This is the one-step parent of the U-Net conditioning vector.

For a feature corresponding to the one-step upper stream before U-Net
conditioning, use `feat_type="both"`. This extracts the output of:

`nets["policy"]["obs_encoder"]`

before robomimic flattens the history into `[B, T_o * Dobs]`.

The visual-latent strategy in the `Celina` branch is appropriate for a narrower
question: extracting a single camera stream's visual feature from
`obs_encoder.obs_nets[rgb_key]`. By default it picks one RGB key, so it does
not produce the full policy conditioning feature unless it is extended to:

- capture both `agentview_image` and `robot0_eye_in_hand_image`
- preserve robomimic's per-key concatenation order
- include the low-dimensional features if the target is the full
  $D_{\text{obs}}$ feature

Therefore, for the full one-step parent of the U-Net conditioning input, hook
the whole `obs_encoder`. For camera-specific analysis, hook the relevant
`VisualCore` submodule.

## 5. Current-Frame Selection And EMA Caveats

The rollout environment can return frame-stacked observations. In that case,
`_prepare_observation(...)` produces tensors with shape `[B, frame_stack, ...]`.
Robomimic's `TensorUtils.time_distributed(...)` applies the observation encoder
per frame and restores the feature tensor as `[B, frame_stack, D]`.

The rollout recorder stores one current-step feature per environment step. It
therefore selects the newest frame slot:

$$\begin{align}
z_t = \texttt{obs\_features}_{[:, -1, :]}.
\end{align}$$

This keeps saved rollout latents shaped `[T, Dz]` for all three feature modes.

The image policy has EMA enabled. During `_get_action_trajectory(...)`,
robomimic uses `self.ema.averaged_model` when it exists. A policy-consistent
feature hook should therefore attach to the same `obs_encoder` instance used by
the rollout path, not necessarily only `policy.nets.policy.obs_encoder`.

Diffusion policies also maintain an action queue. With `action_horizon > 1`,
the policy may not run its encoder at every environment step. A recorder that
needs one feature per environment step should force-run the observation encoder
on the current observation, as `PolicyFeatureHook.update_latent_from_obs(...)`
does in [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py).

## 6. Rollout-Free Dataset Conversion

The rollout-free converter in
[src/robomimic_interface/post_process_robomimic_h5.py](../src/robomimic_interface/post_process_robomimic_h5.py)
reads robomimic `image_v15.hdf5` demonstrations directly and writes one
`RolloutLatentTrajectory` file per source demo. It does not construct a
robomimic environment or step the diffusion policy.

The converter still requires a trained policy checkpoint. The dataset contains
raw RGB observations, but the learned `VisualCore` weights that define the
policy image embedding live in the checkpoint. In this setup, "rollout-free"
means no environment rollout, not checkpoint-free.

By default, the converter reads
`data/robomimic/lift/mh/image_v15.hdf5`. Pass `--dataset` to override that
source file.

The default invocation is:

```bash
python3 -m src.robomimic_interface.post_process_robomimic_h5 \
  --policy-name rmimic-lift-mh-image-v15-diffusion_260123 \
  --checkpoint-name models/model_epoch_300.pth
```

The converter supports the same feature modes as rollout collection:

- `low_dim_concat`: current-frame low-dimensional observations. The CLI also
  accepts `low_dim` as a backward-compatible alias for this mode.
- `image_embedding`: current-frame embeddings for all configured RGB cameras.
- `both`: current-frame full `obs_encoder` output.

For the local Lift MH image checkpoint, running the converter with
`feat_type="both"` and the default output-path inference writes these
one-step observation features under:

`data/robomimic/lift/mh/postprocessed_for_ope/both`

Frame-stack windows are built with robomimic `SequenceDataset` padding
semantics. The saved latent remains one current-frame feature per dataset step:
`[T, Dz]`. For `feat_type="both"`, this means the converter stores the one-step
observation-encoder feature with width $D_{\text{obs}}$, not the flattened
history vector with width $T_o D_{\text{obs}}$ consumed by the diffusion U-Net.
Raw stored observations remain compact by default: low-dimensional keys are
stored for metadata/debugging, while raw RGB images are omitted unless
`--store-rgb-obs` is passed.

When `--output-dir` is omitted, the converter infers the destination directory
from the source dataset path:

`<prefix>/robomimic/<task>/<quality>/postprocessed_for_ope/<feat-type>`

Converted demo files are written directly into that directory without an extra
`h5_files/` subdirectory.

The converter also validates `--feat-type` against the checkpoint
`config.json`. For the local Lift image policy, the checkpoint uses both
low-dimensional and RGB modalities, so the allowed feature mode is `both`.
Narrower feature modes such as `low_dim` / `low_dim_concat` or
`image_embedding` are rejected for that checkpoint.

## 7. Validation

This note documents existing behavior and does not change experimental
behavior, so no training rerun is required.

Before relying on numeric feature dimensions in a new experiment, run a small
checkpoint reconstruction or one-observation smoke pass and print:

- the selected encoder instance path, especially whether it is EMA-backed
- `obs_encoder.output_shape()`
- the captured one-step feature shape
- the final `obs_cond` shape passed to `ConditionalUnet1D`
