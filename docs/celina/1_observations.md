## 1. Part 1: PolicyFeatureHook (`src/latent_sope/robomimic_interface/rollout.py`)

This class walks the policy's module tree to find `VisualCore` and exposes
`_visual_net`, `_activation`, `_randomizers` for use in encoding.

```python
class PolicyFeatureHook:
    def __init__(self, policy, feat_type='visual_latent', rgb_key=None):
        # lazy-import robomimic to avoid breaking training-only pipelines
        (self._ObsUtils, self._ObservationEncoder, self._RolloutPolicy,
         self._PolicyAlgo, self._TensorUtils, self._EnvBase,
         self._resolve_module) = _import_robomimic()

        self.policy = policy
        self._visual_net  = None
        self._randomizers = None
        self._activation  = None

        if feat_type == "visual_latent":
            # walk all named_modules to find the first ObservationEncoder
            obs_encoder = self._resolve_obs_encoder()

            # pick first rgb key (e.g. "agentview_image")
            self._rgb_key = rgb_key or self._pick_rgb_key(obs_encoder)

            # expose encoder components
            self._visual_net  = obs_encoder.obs_nets[self._rgb_key]       # VisualCore
            self._randomizers = list(obs_encoder.obs_randomizers[self._rgb_key])
            self._activation  = obs_encoder.activation

            # register hook (only needed for live rollout, not offline extraction)
            def _vis_hook(_module, _inp, out):
                x = out
                if self._activation is not None:
                    x = self._activation(x)
                for rand in reversed(self._randomizers):
                    if rand is not None:
                        x = rand.forward_out(x)
                x = self._TensorUtils.flatten(x, begin_axis=1).detach()
                self._last_feature = x                                    # (B, 64)

            self._vis_hook_handle = self._visual_net.register_forward_hook(_vis_hook)

    def _resolve_obs_encoder(self):
        """Walk policy module tree to find the first ObservationEncoder."""
        roots = [self.policy]
        for p in ["policy", "policy.nets.policy", "nets.policy"]:
            try:
                roots.append(resolve_module(self.policy, p))
            except Exception:
                pass

        seen, unique_roots = set(), []
        for r in roots:
            if r is None or id(r) in seen:
                continue
            seen.add(id(r))
            unique_roots.append(r)

        for r in unique_roots:
            for name, m in getattr(r, "named_modules", lambda: [])():
                if isinstance(m, self._ObservationEncoder):
                    return m

        raise RuntimeError("Could not find ObservationEncoder inside policy.")

    def _pick_rgb_key(self, obs_encoder):
        keys = list(obs_encoder.obs_shapes.keys())
        rgb_keys = [k for k in keys if self._ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb")]
        return rgb_keys[0]

    def close(self):
        if self._vis_hook_handle is not None:
            self._vis_hook_handle.remove()
```

---

## 2. Part 2: Full Offline Extraction Script

```python
import sys
import h5py
import numpy as np
import torch
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path('/workspace/latent_sope')
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'third_party' / 'robomimic'))

from src.latent_sope.robomimic_interface.checkpoints import (
    load_checkpoint,
    build_rollout_policy_from_checkpoint,
)
from src.latent_sope.robomimic_interface.rollout import PolicyFeatureHook

# ── config ────────────────────────────────────────────────────────────────────
DEVICE     = 'cuda'
BATCH_SIZE = 64
RUN_DIR    = REPO_ROOT / 'third_party/robomimic/diffusion_policy_trained_models/lift_mh/lift_mh_diffusion/20260404040817'
CKPT_PATH  = Path('models/model_epoch_600.pth')
DATASET    = REPO_ROOT / 'third_party/robomimic/datasets/lift/mh/image_v15.hdf5'
OUTPUT_DIR = REPO_ROOT / 'data/mh_latents/lift'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. load policy ────────────────────────────────────────────────────────────
ckpt   = load_checkpoint(RUN_DIR, CKPT_PATH)
policy = build_rollout_policy_from_checkpoint(ckpt, device=DEVICE, verbose=False)

# ── 2. use PolicyFeatureHook to locate VisualCore inside the policy ───────────
#    (the hook itself fires during live rollout; here we just need the modules)
hook        = PolicyFeatureHook(policy, feat_type='visual_latent')
visual_net  = hook._visual_net    # VisualCore: ResNet18 + SpatialSoftmax → (B, 64)
activation  = hook._activation    # post-encoder activation, may be None
randomizers = hook._randomizers   # obs randomizers to undo, may be empty
visual_net.eval()

# ── 3. encode each demo from HDF5 ─────────────────────────────────────────────
with h5py.File(DATASET, 'r') as f:
    demo_keys = sorted(f['data'].keys())

    for i, demo_key in enumerate(demo_keys):
        out_path = OUTPUT_DIR / f'demo_{i:04d}.npy'
        if out_path.exists():
            print(f'[skip] {demo_key}')
            continue

        images  = np.array(f[f'data/{demo_key}/obs/agentview_image'])  # (T, H, W, C) uint8
        actions = np.array(f[f'data/{demo_key}/actions'], dtype=np.float32)  # (T, 7)

        # encode in batches
        latents = []
        for t0 in range(0, len(images), BATCH_SIZE):
            img_t = torch.from_numpy(images[t0:t0+BATCH_SIZE]).float()
            img_t = img_t.permute(0, 3, 1, 2) / 255.0   # (B, C, H, W) in [0, 1]
            img_t = img_t.to(DEVICE)

            with torch.no_grad():
                z = visual_net(img_t)                    # (B, 64)
                if activation is not None:
                    z = activation(z)
                for rand in reversed(randomizers):
                    if rand is not None:
                        z = rand.forward_out(z)

            latents.append(z.cpu().numpy())

        latents = np.concatenate(latents, axis=0).astype(np.float32)  # (T, 64)
        np.save(out_path, {'latents': latents, 'actions': actions})
        print(f'[{i+1}/{len(demo_keys)}] {demo_key}: latents={latents.shape}')

hook.close()
print('Done.')
```
