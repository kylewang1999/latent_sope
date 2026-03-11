# MVP v0.2.1: Guidance Ablations — Does Guidance Actually Work?

**Date:** 2026-03-11
**Notebook:** `experiments/2026-03-11/MVP_v0.2.1_guidance_ablations.ipynb`
**Status:** Planned
**Builds on:** MVP v0.2 (guided stitching, single target policy)

---

## Motivation

MVP v0.2 showed that guidance at scale=0.1 drops the OPE estimate from 16.08 to 0.68 (oracle = 0.54). However, this does not prove guidance works — the result could be trajectory degradation that *happens* to land near the right answer. The pattern is monotonic (more guidance = less success), with no scale where guided trajectories actually match the target policy's 54% success rate.

**Key question:** Does the specific content of the guidance gradient matter, or does any perturbation of the same magnitude produce similar results?

## Experiment Design

Two ablations, both using the same setup as v0.2 (same diffusion checkpoint, same initial states, scale=0.1):

### Ablation 1: Random Guidance

Replace the BC_Gaussian gradient with **random unit vectors** at the same scale.

- Implementation: replace `grad_action` with `torch.randn_like(grad_action)`, then normalize to unit norm (same as v0.2's `normalize_grad=True`).
- Everything else identical: same diffusion model, same initial states, same scale=0.1, same number of trajectories.

**Expected outcomes:**
- If random guidance OPE ≈ 0.68 → guidance signal doesn't matter, it's just perturbation. **Guidance debunked.**
- If random guidance OPE ≈ 16 (stays near unguided) → perturbation alone doesn't reduce OPE; the BC_Gaussian gradient carries real information. **Evidence for guidance.**
- If random guidance OPE ≈ 0 (total collapse) → random noise is more destructive than BC_Gaussian gradient, which would also be informative.

### Ablation 2: Wrong-Policy Guidance

Train a BC_Gaussian on **expert demo** (state, action) pairs instead of the target policy's rollouts, then guide with it at scale=0.1.

- Implementation: use `offline_data` (200 expert demos, ~9,466 pairs) to train a second BC_Gaussian with the same architecture (11→64→64→4 + learnable log_std).
- Guide stitching with this wrong-policy BC_Gaussian at scale=0.1.

**Expected outcomes:**
- If wrong-policy guidance OPE ≈ 0.68 (same as target-policy guidance) → the policy identity doesn't matter. **Guidance debunked.**
- If wrong-policy guidance OPE ≈ 16 (stays near unguided) → guiding toward the behavior policy keeps trajectories behavior-like (makes sense), and the target-policy guidance is doing something policy-specific. **Evidence for guidance.**
- If wrong-policy guidance gives some other value → still informative for understanding what guidance does.

## Configuration

Inherits all v0.2 config. Only changes:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Guidance scale | 0.1 | Best from v0.2 |
| Conditions compared | 4 | Unguided, target-policy guided, random guided, wrong-policy guided |
| Random guidance seed | 42 | For reproducibility |
| Wrong-policy BC training data | ~9,466 pairs from 200 expert demos | Already loaded in v0.2 |

## Success Criteria

| Outcome | Interpretation |
|---------|----------------|
| Random ≈ target-policy ≈ 0.68 | Guidance is just perturbation — debunked |
| Random ≈ 16, wrong-policy ≈ 16, target-policy ≈ 0.68 | Guidance is policy-specific — strong evidence it works |
| Random ≈ 16, wrong-policy ≈ 0.68, target-policy ≈ 0.68 | Policy identity doesn't matter, but gradient structure does — weak evidence |
| Any other pattern | Requires further analysis |

## Implementation Plan

1. Load v0.2 setup (diffusion checkpoint, normalization, oracle, demo data, target rollout data)
2. Train target-policy BC_Gaussian (same as v0.2)
3. Train wrong-policy BC_Gaussian on expert demo data
4. Run 4 conditions: unguided, target-policy guided, random guided, wrong-policy guided
5. Compare OPE estimates, success rates, and trajectory shapes
6. Visualize: bar chart of OPE by condition, trajectory overlays per condition

## Results

*To be filled after running the experiment.*
