"""Utilities for extracting embeddings from a (robomimic) policy.

Step 1-3 of your plan can proceed with *low-dim obs as z_t* (see
scripts/cache_latents.py).

For SOPE-guided diffusion with image embeddings, you typically want z_t to be
produced by the *same observation encoder* the robomimic policy uses.

Robomimic does not expose a single universal "get_embedding(obs)" API across
all algos and observation modalities. Therefore this module provides:
- a low-dim concatenation encoder (always works)
- an opt-in hook-based feature extractor for robomimic models (works when you can specify the module path)

You can progressively specialize this for your chosen robomimic diffusion policy implementation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class EncoderOutput:
    """Output from encoding a batch of observations."""
    z: np.ndarray  # (B, Dz)


def _flatten_time_major(x: np.ndarray) -> np.ndarray:
    """Flatten non-time dims: (T, ...) -> (T, D)."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x.reshape(x.shape[0], -1)


def resolve_module(root: Any, dotted_path: str) -> Any:
    """Resolve a dotted attribute / dict path.

    Supports:
    - attributes: obj.foo
    - dict access: obj['foo'] when obj is dict-like or torch.nn.ModuleDict

    Example:
        resolve_module(algo, "nets.policy.obs_encoder")

    This is intentionally simple; if a token isn't found, it raises KeyError/AttributeError.
    """

    cur = root
    for tok in dotted_path.split("."):
        if isinstance(cur, dict):
            cur = cur[tok]
        else:
            # torch.nn.ModuleDict behaves like a dict for __getitem__
            if hasattr(cur, "__getitem__") and not hasattr(cur, tok):
                try:
                    cur = cur[tok]
                    continue
                except Exception:
                    pass
            cur = getattr(cur, tok)
    return cur


class LowDimConcatEncoder:
    """Simple encoder that concatenates low-dim obs attributes
    specified by obs_keys."""

    def __init__(self, obs_keys: Optional[Sequence[str]] = None):
        self.obs_keys = None if obs_keys is None else list(obs_keys)
        self.obs_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
        self.obs_dims: Optional[Dict[str, int]] = None

    def __call__(self, obs_dict: Dict[str, np.ndarray]) -> EncoderOutput:
        keys = self.obs_keys or list(obs_dict.keys())
        self.obs_keys = list(keys)

        if self.obs_shapes is None or self.obs_dims is None:
            self.obs_shapes = {}
            self.obs_dims = {}
            for k in self.obs_keys:
                arr = np.asarray(obs_dict[k])
                if arr.ndim == 1: # assume (T,)
                    shape = (1,)
                else: # assume (T, ...)
                    shape = tuple(arr.shape[1:])
                dim = int(np.prod(shape))
                self.obs_shapes[k] = shape
                self.obs_dims[k] = dim

        parts = [_flatten_time_major(obs_dict[k]).astype(np.float32, copy=False) for k in self.obs_keys]
        z = np.concatenate(parts, axis=-1)
        return EncoderOutput(z=z)
    
    def decode_to_obs_dict(self, z: np.ndarray) -> Dict[str, np.ndarray]:
        """Reconstruct an obs dict from a concatenated latent.

        Requires either:
        - self.obs_shapes: dict mapping key -> shape (excluding time dim), or
        - self.obs_dims: dict mapping key -> flattened dim
        """
        if isinstance(z, dict):
            return {k: np.asarray(v) for k, v in z.items()}

        z = np.asarray(z)
        if z.ndim == 1:
            z = z[None, :]
        if z.ndim != 2:
            raise ValueError(f"Expected z with shape (T, D) or (D,), got {z.shape}")

        obs_shapes = getattr(self, "obs_shapes", None)
        obs_dims = getattr(self, "obs_dims", None)

        if obs_shapes is None and obs_dims is None:
            raise ValueError(
                "decode_to_obs_dict requires encoder.obs_shapes or encoder.obs_dims to be set "
                "(dict mapping obs key -> shape or flattened dim)."
            )

        if obs_shapes is not None:
            keys = self.obs_keys or list(obs_shapes.keys())
        else:
            keys = self.obs_keys or list(obs_dims.keys())

        obs: Dict[str, np.ndarray] = {}
        cursor = 0
        D = int(z.shape[1])

        for k in keys:
            if obs_shapes is not None:
                shape = tuple(obs_shapes[k])
                dim = int(np.prod(shape))
            else:
                shape = None
                dim = int(obs_dims[k])

            if cursor + dim > D:
                raise ValueError(f"Not enough dims in z to decode key={k}: need {dim}, have {D - cursor}")

            chunk = z[:, cursor : cursor + dim]
            if shape is not None:
                obs[k] = chunk.reshape((z.shape[0],) + shape)
            else:
                obs[k] = chunk
            cursor += dim

        if cursor != D:
            raise ValueError(f"Unused dims in z: expected {cursor}, got {D}")

        return obs


class HighDimObsEncoder:
    """Feature extractor that uses a forward hook on a module inside a policy.

    Usage pattern:
    1) Load an Algo (not just RolloutPolicy) from a robomimic checkpoint.
    2) Identify the module whose output you want to treat as embedding.
    3) Create this encoder with feature_module_path pointing to that module.
    4) Call encode_obs_batch(...), which will:
       - run a forward pass that triggers the hook
       - return the captured output as numpy

    IMPORTANT:
    - This cannot be fully "automatic" because robomimic algorithms differ.
    - You must ensure the policy forward you call actually executes the chosen module.
    """

    def __init__(
        self,
        policy: Any,
        feature_module_path: str,
        forward_fn: Optional[str] = None,
    ):
        """
        Args:
            policy: robomimic Algo or RolloutPolicy.
            feature_module_path: dotted path to a torch.nn.Module inside policy.
            forward_fn: which method name to call for the forward pass.
                - None: try __call__ first, then get_action if present.
                - "get_action": call policy.get_action(obs_dict)
        """
        self.policy = policy
        self.feature_module_path = feature_module_path
        self.forward_fn = forward_fn

        self._hook_handle = None
        self._last_feature = None

        # Register hook
        self._register_hook()

    def _register_hook(self) -> None:
        try:

            mod = resolve_module(self.policy, self.feature_module_path)
            if not hasattr(mod, "register_forward_hook"):
                raise TypeError(f"Resolved object is not a torch module: {type(mod)}")

            def _hook(_module, _inp, out):
                # Detach and stash
                if torch.is_tensor(out):
                    self._last_feature = out.detach()
                elif isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]):
                    self._last_feature = out[0].detach()
                else:
                    self._last_feature = out

            self._hook_handle = mod.register_forward_hook(_hook)
        except Exception as e:
            raise RuntimeError(
                "Failed to register feature hook. Make sure torch+robomimic are installed and the module path is correct. "
                f"path={self.feature_module_path} err={e}"
            ) from e

    def close(self) -> None:
        """Remove hook."""
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None

    def __del__(self):
        self.close()

    def _run_forward(self, obs_dict: Dict[str, Any]) -> None:
        # Reset last feature
        self._last_feature = None

        if self.forward_fn is None:
            # Try calling the object
            try:
                _ = self.policy(obs_dict)
                return
            except Exception:
                pass
            if hasattr(self.policy, "get_action"):
                _ = self.policy.get_action(obs_dict)
                return
            raise AttributeError("policy is not callable and has no get_action")

        fn = getattr(self.policy, self.forward_fn)
        _ = fn(obs_dict)

    def encode_obs_batch(self, obs_batch: Dict[str, np.ndarray], device: str = "cuda") -> EncoderOutput:
        """Encode a time-major observation batch.

        Args:
            obs_batch: dict of arrays, each (T, ...) where T is time.

        Returns:
            EncoderOutput with z of shape (T, Dz) or (T, ... flattened)
        """
        import torch

        # Convert to tensors and run forward step-by-step to mirror policy usage.
        keys = list(obs_batch.keys())
        T = int(np.asarray(obs_batch[keys[0]]).shape[0])

        feats: List[np.ndarray] = []

        self.policy.eval() if hasattr(self.policy, "eval") else None

        with torch.no_grad():
            for t in range(T):
                obs_t = {}
                for k in keys:
                    x = np.asarray(obs_batch[k][t])
                    obs_t[k] = torch.from_numpy(x).to(device=device)

                self._run_forward(obs_t)

                if self._last_feature is None:
                    raise RuntimeError(
                        "Forward did not trigger the feature hook. "
                        "Check feature_module_path and forward_fn."
                    )

                f = self._last_feature
                if torch.is_tensor(f):
                    f_np = f.detach().cpu().numpy()
                else:
                    f_np = np.asarray(f)

                feats.append(f_np.reshape(1, -1))

        z = np.concatenate(feats, axis=0).astype(np.float32)
        return EncoderOutput(z=z)


def extract_embeddings_batched(
    encoder: Any,
    obs_batch: Dict[str, np.ndarray],
    device: str = "cuda",
) -> np.ndarray:
    """Convenience wrapper returning numpy z.

    Args:
        encoder: LowDimConcatEncoder or RobomimicObsEncoder
        obs_batch: dict of arrays, time-major

    Returns:
        z: (T, Dz)
    """
    if isinstance(encoder, LowDimConcatEncoder):
        return encoder(obs_batch).z
    if hasattr(encoder, "encode_obs_batch"):
        return encoder.encode_obs_batch(obs_batch, device=device).z
    raise TypeError(f"Unsupported encoder type: {type(encoder)}")
