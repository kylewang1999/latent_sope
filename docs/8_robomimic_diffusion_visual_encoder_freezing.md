# Robomimic Diffusion Visual Encoder Freezing

Relevant code:

- [data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json](../data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json)
- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py)
- [third_party/robomimic/robomimic/algo/algo.py](../third_party/robomimic/robomimic/algo/algo.py)
- [third_party/robomimic/robomimic/models/obs_core.py](../third_party/robomimic/robomimic/models/obs_core.py)
- [third_party/robomimic/robomimic/models/base_nets.py](../third_party/robomimic/robomimic/models/base_nets.py)
- [third_party/robomimic/robomimic/utils/torch_utils.py](../third_party/robomimic/robomimic/utils/torch_utils.py)

## 1. Short Answer

Yes. In this repository's robomimic diffusion-policy setup, the visual encoder
is **not frozen** during training.

More specifically, the image encoder here is a `VisualCore` built around
`ResNet18Conv`, so the backbone is a ResNet-18-style CNN, not a frozen
precomputed feature extractor.

## 2. Why The Encoder Is Trainable

### 2.1 Config Selects A Learnable ResNet-18 Visual Core

The checked-in diffusion-policy config uses:

- `core_class = "VisualCore"`
- `backbone_class = "ResNet18Conv"`
- `backbone_kwargs.pretrained = false`

See
[config.json](../data/policy/rmimic-lift-mh-image-v15-diffusion_260123/config.json)
lines 168-186.

That means the policy uses the standard robomimic visual encoder stack for RGB
observations, with a randomly initialized `ResNet18Conv` backbone rather than a
pretrained frozen encoder.

### 2.2 `VisualCore` Does Not Freeze The Backbone

`VisualCore` instantiates the requested backbone and then appends pooling and an
optional linear projection. It does not call any freeze helper or disable
gradients on the backbone parameters. See
[obs_core.py](../third_party/robomimic/robomimic/models/obs_core.py) lines
61-140.

### 2.3 `ResNet18Conv` Has No Freeze Path

`ResNet18Conv` accepts `pretrained` and `input_coord_conv`, but it has no
`freeze` argument and does not set `requires_grad = False`. It just builds a
torchvision ResNet-18 trunk. See
[base_nets.py](../third_party/robomimic/robomimic/models/base_nets.py) lines
506-537.

### 2.4 The Diffusion Policy Optimizer Includes The Encoder Parameters

`DiffusionPolicyUNet` creates one `"policy"` module containing both:

- `obs_encoder`
- `noise_pred_net`

See
[diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py)
lines 126-161.

During training, the loss is backpropagated with `self.optimizers["policy"]`;
that optimizer is created over `net.parameters()`, and for diffusion policy the
relevant `net` is `self.nets["policy"]`. So the optimizer sees both the UNet
weights and the observation encoder weights. See:

- [algo.py](../third_party/robomimic/robomimic/algo/algo.py) lines 172-199
- [torch_utils.py](../third_party/robomimic/robomimic/utils/torch_utils.py) lines 106-120
- [diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py) lines 267-305

## 3. What Might Cause Confusion

### 3.1 Other Robomimic Backbones Can Be Frozen

Robomimic does have other visual backbones, such as `R3MConv` and `MVPConv`,
that explicitly support a `freeze=True` option. That is a different code path
from the one used by this diffusion-policy config.

### 3.2 Other Algorithms Expose `freeze_encoder`

Robomimic also has a `freeze_encoder` path in some non-diffusion-policy code,
for example the BC-VAE path. Diffusion policy does not use that mechanism. See:

- [bc.py](../third_party/robomimic/robomimic/algo/bc.py) lines 415-422
- [policy_nets.py](../third_party/robomimic/robomimic/models/policy_nets.py) lines 1515-1550

## 4. Conclusion

For the robomimic diffusion policy being trained in this repo, the visual
encoder is a trainable ResNet-18-based encoder and its weights are updated
jointly with the diffusion policy.

No validation reruns were needed here because this note documents the current
code path and does not change training behavior.
