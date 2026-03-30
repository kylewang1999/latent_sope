# Eval EEF Position Logging

Relevant code:
- [src/eval.py](../src/eval.py)
- [src/sope_diffuser.py](../src/sope_diffuser.py)

This note records the evaluation-side wiring for the end-effector position
diagnostics used by the SOPE chunk evaluator.

## What Changed

The chunk evaluator now computes `robot0_eef_pos` aggregate diagnostics from the
same low-dim state slice used by the training-time `diffuser_eef_pos_only`
ablation.

For Lift low-dim rollouts, that slice is `$[10:13)$`, matching
[src/sope_diffuser.py](../src/sope_diffuser.py).

When that slice is available, [src/eval.py](../src/eval.py) now:

- accumulates `rmse_eef_pos` over the future state slice only
- accumulates `mean_eef_pos` and `mean_eef_pos_gt` using the same squared-value
  convention as the existing state and action diagnostics
- includes those metrics in the `eval_metrics:*` and `eval_diagnostics:*`
  summaries sent to `wandb`

If the loaded state layout does not expose the expected EEF position slice, the
extra diagnostics remain unset instead of failing evaluation.

## Validation

Validation run after the change:

- `python3 -m py_compile src/eval.py`
