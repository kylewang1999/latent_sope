# Utility Module Migration

Relevant code:
- [src/utils.py](../src/utils.py)
- [src/diffusion/train.py](../src/diffusion/train.py)
- [src/diffusion/sope_diffuser.py](../src/diffusion/sope_diffuser.py)
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)
- [src/robomimic_interface/rollout.py](../src/robomimic_interface/rollout.py)
- [src/robomimic_interface/checkpoints.py](../src/robomimic_interface/checkpoints.py)

Before this change, utility helpers were split across `src/latent_sope/utils/common.py` and `src/latent_sope/utils/misc.py`, while the rest of the repository was already being moved from `src/latent_sope/*` into top-level `src/*` modules.

After this change, the shared utility surface lives in `src/utils.py`. The active training and robomimic code now imports from that single module, and the legacy `src/latent_sope/utils/*` files are reduced to compatibility shims that re-export from `src.utils`.

Implementation notes:
- Heavy optional dependencies in the old `common.py` were moved behind function-local imports so `from src.utils import resolve_device, set_global_seed` does not eagerly import Hydra, JAX, MuJoCo, Orbax, or W&B.
- `get_repo_root()` now resolves to the repository root from `src/utils.py`, which keeps `logs/` and `configs/` lookups aligned with the repository-level directory conventions after the module move.
- As part of the same migration path, direct imports in `src/diffusion/*` and `src/robomimic_interface/*` were updated away from deleted `src.latent_sope.*` module paths.

Validation:
- Run `python3 -m py_compile src/utils.py src/diffusion/train.py src/diffusion/sope_diffuser.py src/robomimic_interface/dataset.py src/robomimic_interface/rollout.py src/robomimic_interface/checkpoints.py src/latent_sope/utils/common.py src/latent_sope/utils/misc.py`.
- Run `python3 -c "import src.utils as u; print(u.resolve_device(prefer_cuda=False))"` as a light import smoke test.
