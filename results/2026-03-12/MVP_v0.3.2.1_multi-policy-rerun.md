# MVP v0.3.2.1: Shared Diffuser Rerun

**Date:** 2026-03-12
**Notebook:** `experiments/2026-03-12/MVP_v0.3.2.1_multi-policy-rerun.ipynb`
**Status:** COMPLETED — 6.9 min total (ran locally on login node GPU)
**Result:** FAIL — guidance kills all successes, 0% synthetic SR across all policies

## What was tried

Rerun of v0.3.2 multi-policy evaluation with a key architectural fix: train the diffuser and behavior policy **once** on the full behavior dataset, then reuse for all target policies. This matches the SOPE reference implementation.

### Changes from v0.3.2
1. **One diffuser** trained on 200 expert demos + 80 target rollouts (20 per policy, 4 policies) = 280 episodes
2. **One BC_Gaussian** trained on the same 280-episode dataset (used for negative guidance only)
3. **Per-policy evaluation** only builds target policy scorer + runs guided stitching + scoring (no retraining)
4. All rollout data reused from v0.3.2 (cached in `rollouts/multi_policy_*`)
5. Fixed matplotlib visualization (Agg backend + `display(fig)` instead of `plt.show()`)
6. Added base SR measurement (unguided diffuser)
7. Added per-policy trajectory visualization (real vs synthetic)

### Guidance setup
- **Positive guidance:** target policy's score function (`RobomimicDiffusionScorer` per policy)
- **Negative guidance:** shared BC_Gaussian behavior policy (trained on 200 expert + 80 target)
- Config: `action_scale=0.2, ratio=0.25` (same as v0.3.2)

### Setup
- 4 target policies: 50demos_epoch10 (0%), 200demos_epoch10 (18%), 10demos_epoch20 (52%), 200demos_epoch40 (90%)
- Oracle values from `oracle_eval_all_checkpoints.json` (50 rollouts each)
- Training data: 280 episodes → 6421 chunks, 100 batches/epoch
- Training: 25 epochs, final loss=0.159605
- BC_Gaussian: 500 epochs, final NLL=-6.9050
- Synthetic: 20 trajectories per policy, T=60

## Key Metrics

- **Spearman rank correlation:** NaN (all OPE estimates are 0)
- **Regret@1:** 0.9000
- **Mean relative error:** 100% (excl. 0% oracle policies)
- **MSE:** 0.2782
- **MAE:** 0.4000

### Base diffuser (no guidance)
- **SR: 60%** (12/20 trajectories)
- cube_z max: mean=0.8389, min=0.8301, max=0.8467

### Per-policy results (with guidance)

| Checkpoint | Oracle SR | Oracle V^pi | OPE Estimate | Synth SR | Rel Error | Time |
|-----------|-----------|-------------|--------------|----------|-----------|------|
| 50demos_epoch10 | 0% | 0.0000 | 0.0000 | 0.0% | 0.0% | 56s |
| 200demos_epoch10 | 18% | 0.1800 | 0.0000 | 0.0% | 100.0% | 59s |
| 10demos_epoch20 | 52% | 0.5200 | 0.0000 | 0.0% | 100.0% | 63s |
| 200demos_epoch40 | 90% | 0.9000 | 0.0000 | 0.0% | 100.0% | 52s |

## Comparison to prior experiments

| Version | Key change | Base SR | Guided SR range | Spearman ρ | Regret@1 | Notes |
|---------|-----------|---------|-----------------|------------|----------|-------|
| v0.3.2 | Per-policy retraining | N/A | 0–60% | 0.40 | 0.00 | Retrained diffuser per policy |
| **v0.3.2.1** | **Shared diffuser** | **60%** | **0% (all)** | **NaN** | **0.90** | **Guidance kills all successes** |

## Analysis

### What happened

The unguided diffuser generates 60% SR — reasonable given ~71% of training data is expert demos (100% SR). But with guidance (action_scale=0.2, ratio=0.25), SR drops to **0% for every policy**, including the 90% SR one.

### Why guidance is destructive

The negative guidance pushes away from the BC_Gaussian behavior policy, which was trained on all 280 episodes. Since ~71% of those episodes are expert demos that successfully lift the cube, **pushing away from the behavior policy means pushing away from the lifting pattern**. The negative guidance is effectively anti-correlated with success.

In SOPE's original setting (D4RL), the behavior data is a single mediocre policy, so pushing away from behavior + toward target makes sense. Here, the behavior data is dominated by expert demos, so negative guidance is counterproductive.

### Why v0.3.2 partially worked despite this

In v0.3.2, the BC_Gaussian was trained **per policy** on only 20 target rollouts (all failures due to the recording bug). Pushing away from a "always fail" behavior policy is less destructive — it just adds noise. The positive guidance from the target policy could still occasionally steer toward success. That's why v0.3.2 got some nonzero SR (0–60%) while v0.3.2.1 gets 0%.

### Possible fixes

1. **Reduce or remove negative guidance**: Try ratio=0 (positive guidance only) or much smaller ratio (0.05)
2. **Reduce action_scale**: 0.2 may be too aggressive; try 0.05, 0.1
3. **Train BC only on target rollouts**: So negative guidance pushes away from the target policy's behavior, not the expert's
4. **Sweep guidance hyperparameters**: Test combinations of action_scale × ratio to find a regime where guidance helps without destroying the base distribution
