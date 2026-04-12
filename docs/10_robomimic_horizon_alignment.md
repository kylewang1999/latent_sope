# Robomimic Horizon Alignment Validation

Relevant code:

- [src/diffusion.py](../src/diffusion.py)
- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py)
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)

## 1. Summary

`SopeDiffuser` now asserts that `cfg.frame_stack` matches the robomimic
policy's `observation_horizon` whenever a target or behavior policy is attached
for guidance.

## 2. Before And After

### 2.1 Before

The SOPE diffusion config and the attached robomimic guidance policy could be
constructed with different context horizons. The mismatch was not rejected at
construction time, even though the guidance adapter relies on robomimic's
windowed observation semantics.

### 2.2 After

[`SopeDiffuser.__init__`](../src/diffusion.py) now validates this invariant for
both the target policy and the behavior policy:

- `cfg.frame_stack == policy.observation_horizon`

If the horizons differ, diffuser construction now fails immediately with a
message that reports both values.

## 3. Reason

The local FiLM chunk diffuser conditions on `states_from`, which contains
exactly `frame_stack` prefix states from the dataset contract in
[`RolloutChunkDataset`](../src/robomimic_interface/dataset.py). The robomimic
guidance adapter in [`DiffusionPolicy`](../src/robomimic_interface/policy.py)
interprets action-score queries using its own `observation_horizon`. Allowing
those horizons to differ would silently mix two incompatible sequence
contracts.

## 4. Validation

Re-run:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python3 -m py_compile src/diffusion.py src/robomimic_interface/policy.py src/robomimic_interface/dataset.py
```
