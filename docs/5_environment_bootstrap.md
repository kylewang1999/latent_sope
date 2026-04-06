# Environment Bootstrap

Relevant code:
- [bootstrap_env.sh](../bootstrap_env.sh)
- [bootstrap_egl.sh](../bootstrap_egl.sh)
- [third_party/robomimic/setup.py](../third_party/robomimic/setup.py)

## 1. Summary

This note consolidates the environment bootstrap strategy for robomimic, SOPE, MuJoCo EGL, and the Hugging Face / JAX compatibility layer.

The main goal is to rebuild the environment without reintroducing the old robomimic `diffusers` dependency that breaks with modern JAX.

## 2. Bootstrap Strategy

[`bootstrap_env.sh`](../bootstrap_env.sh) now:
- installs PyTorch explicitly
- installs modern JAX explicitly
- installs robomimic runtime dependencies explicitly
- installs a modern Hugging Face stack explicitly
- installs robomimic editable with `--no-deps --no-build-isolation`
- installs clean_diffuser editable with `--no-deps --no-build-isolation`

This avoids robomimic's stale dependency pins from [`third_party/robomimic/setup.py`](../third_party/robomimic/setup.py), which still include:
- `diffusers==0.11.1`
- `transformers==4.41.2`
- `huggingface_hub==0.23.4`

## 3. Why This Is Needed

The incompatibility comes from mixing an old `diffusers` release with a modern JAX release. In that configuration, later robomimic imports can fail because old `diffusers` code expects symbols such as `jax.random.KeyArray` that are no longer present in newer JAX.

The chosen fix is preventive rather than reparative:
- do not let robomimic install its pinned Hugging Face packages in the first place
- keep the desired modern stack in place throughout bootstrap

## 4. EGL Hook Behavior

[`bootstrap_egl.sh`](../bootstrap_egl.sh) installs conda activation hooks that default MuJoCo rendering to EGL:
- `MUJOCO_GL=egl`
- `PYOPENGL_PLATFORM=egl`
- `MUJOCO_EGL_DEVICE_ID=0` by default

It also checks the active environment for the old-`diffusers` / new-JAX hazard and upgrades the Hugging Face stack when the problematic combination is detected.

## 5. Validation

Run:

```bash
bash -n bootstrap_env.sh
bash -n bootstrap_egl.sh
```

These commands check shell syntax without rebuilding the environment.
