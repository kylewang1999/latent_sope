# Oracle Evaluation: All Target Policy Checkpoints

**Date:** 2026-03-12
**Notebook:** `experiments/2026-03-12/oracle_eval_all_checkpoints.ipynb`
**SLURM Job:** 7337788 (d13-04, V100)

## What was done

Evaluated 16 diffusion policy checkpoints (4 demo counts x 4 training epochs) with 50 oracle rollouts each (horizon=60). This gives us a range of target policy success rates for multi-policy OPE evaluation.

## Oracle Results (50 rollouts each)

| Checkpoint | V^pi | Std | SR | Time(s) |
|---|---|---|---|---|
| 10demos_epoch10 | 0.08 | 0.27 | 8% | 535 |
| 10demos_epoch20 | 0.52 | 0.50 | 52% | 489 |
| 10demos_epoch30 | 0.62 | 0.49 | 62% | 463 |
| 10demos_epoch40 | 0.52 | 0.50 | 52% | 472 |
| 50demos_epoch10 | 0.00 | 0.00 | 0% | 526 |
| 50demos_epoch20 | 0.60 | 0.49 | 60% | 449 |
| 50demos_epoch30 | 0.82 | 0.38 | 82% | 431 |
| 50demos_epoch40 | 0.88 | 0.33 | 88% | 422 |
| 100demos_epoch10 | 0.06 | 0.24 | 6% | 529 |
| 100demos_epoch20 | 0.42 | 0.49 | 42% | 504 |
| 100demos_epoch30 | 0.72 | 0.45 | 72% | 472 |
| 100demos_epoch40 | 0.76 | 0.43 | 76% | 449 |
| 200demos_epoch10 | 0.18 | 0.38 | 18% | 547 |
| 200demos_epoch20 | 0.24 | 0.43 | 24% | 550 |
| 200demos_epoch30 | 0.42 | 0.49 | 42% | 486 |
| 200demos_epoch40 | 0.90 | 0.30 | 90% | 424 |

## Analysis

- **SR range: 0% to 90%** — good spread for multi-policy OPE evaluation
- All epoch-10 checkpoints are weak (0–18%), consistent with early training
- Best per demo count: 10demos peaks at 62% (epoch 30), 50demos at 88% (epoch 40), 100demos at 76% (epoch 40), 200demos at 90% (epoch 40)
- 200demos has a slow start (18% at epoch 10, 24% at epoch 20) but reaches the highest SR at epoch 40 — more data needs more training
- 50demos_epoch10 is completely untrained (0% SR), useful as a negative control
- Total eval time: ~2.1 hours (16 checkpoints x ~8 min each)
