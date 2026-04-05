# FiLM-Conditioned Robomimic Backbone

This note documents the FiLM-conditioned diffusion path added for the
robomimic low-dim chunk-diffusion debugging setup.

Related code:

- [src/diffusion.py](../src/diffusion.py)
- [src/sope_diffuser.py](../src/sope_diffuser.py)
- [src/train.py](../src/train.py)
- [src/eval.py](../src/eval.py)
- [scripts/train_sope_film.py](../scripts/train_sope_film.py)
- [docs/12_ddpm_parameterization_and_conditioning.md](./12_ddpm_parameterization_and_conditioning.md)

## Before

The existing SOPE wrapper in
[src/sope_diffuser.py](../src/sope_diffuser.py) uses in-painting-style
conditioning:

- prefix states are inserted directly into the trajectory tensor
- SOPE's `GaussianDiffusion` repeatedly re-applies those fixed coordinates
  during training and sampling
- the denoiser predicts a sequence whose horizon includes the conditioned prefix

That matches SOPE's original conditioning mechanism, but it is the path called
out in
[docs/12_ddpm_parameterization_and_conditioning.md](./12_ddpm_parameterization_and_conditioning.md)
as a poor fit for the current robomimic debugging target.

## After

The new FiLM path in [src/diffusion.py](../src/diffusion.py):

- keeps SOPE's DDPM objective, scheduler coefficients, weighting logic, and
  checkpoint structure
- swaps the denoiser to robomimic's `ConditionalUnet1D`
- flattens `states_from` into a single global conditioning vector of shape
  `(B, frame_stack * state_dim)`
- predicts only the future transition chunk
  `states_to[:, :-1, :] || actions_to`

In other words, the change is the conditioning path and denoiser backbone, not
the outer DDPM training loop.

## Interface Notes

- `FilmDiffusionConfig` intentionally keeps the same main knobs as
  `SopeDiffusionConfig`, including `predict_epsilon`, `action_weight`,
  `loss_discount`, `diffusion_steps`, `dim_mults`, `diffuser_eef_pos_only`, and
  `conditioning_mode`.
- `conditioning_mode="prefix_states"` now means FiLM conditioning from flattened
  prefix states, not in-painting.
- `conditioning_mode="none"` keeps the unconditioned future-chunk path for the
  existing debug control.
- `dim_mults` are mapped to robomimic `down_dims` using a fixed base width of
  `256`, so `(1, 2)` becomes `(256, 512)`.
- `attention` remains in the config for knob compatibility, but robomimic's
  `ConditionalUnet1D` does not use the SOPE attention blocks.

## Validation

The following local checks were run after the change:

- `python3 -m py_compile src/diffusion.py src/train.py src/eval.py scripts/train_sope_film.py`
- FiLM construction smoke test on CPU
- FiLM loss smoke test confirming FiLM context shape `(B, frame_stack * state_dim)`
- FiLM unconditioned sampling smoke test confirming sampled chunk shape
  `(B, chunk_horizon, state_dim + action_dim)`
- `python3 scripts/train_sope_film.py --epochs 1 --max-steps 1 --batch-size 1 --num-workers 0 --diffusion-steps 1 --dim-mults 1 --num-evals 0 --wandb-mode disabled`
- `python3 scripts/train_sope.py --epochs 1 --max-steps 1 --batch-size 1 --num-workers 0 --num-evals 0 --wandb-mode disabled`

The remaining experimental checks to rerun are:

1. held-out chunk RMSE on the 19D robomimic setup
2. direct comparison against the prefix-state in-painting baseline
3. the EEF-only control to confirm the refactor did not break the unconditioned
   debug path
