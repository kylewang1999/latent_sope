# Finding: SOPE's Behavior Data Mix

**Date:** 2026-03-12

## Key Discovery

SOPE trains its chunk diffusion model and BC behavior policy **exclusively on medium-quality behavior data** (~50% of expert performance). No expert data, no mixed-quality data.

### Evidence from codebase

All three D4RL configs use medium datasets:
- `hopper-medium-v2`
- `halfcheetah-medium-v2`
- `walker2d-medium-v2`

D4RL "medium" = rollouts from a single SAC policy stopped at ~1/3 of full training. Approximate normalized scores:

| Env | Medium score | Expert score |
|-----|-------------|-------------|
| Hopper | ~50-60 | ~110 |
| HalfCheetah | ~44 | ~93 |
| Walker2d | ~75-80 | ~108 |

### Evidence from paper

The paper states: "we use the halfcheetah-medium, hopper-medium and walker2d-medium behavior datasets."

No ablation over data quality. No experiments with medium-expert, expert, medium-replay, or random datasets. SOPE has **only ever been validated on medium-quality behavior data**.

## Why medium quality matters

The behavior data serves as a **neutral anchor** for guidance:
- Diffuser learns to generate medium-quality trajectories
- For good target policies (better than behavior): guidance steers UP → higher returns
- For bad target policies (worse than behavior): guidance steers DOWN → lower returns
- Ranking emerges from this bidirectional adjustment

Medium quality (~50%) gives maximum dynamic range in both directions.

## Our setting vs SOPE

| Aspect | SOPE | Ours |
|--------|------|------|
| Behavior data | Medium only (~50% quality) | 200 expert (100% SR) + 80 target (0-90% SR) |
| Data quality | Homogeneous, moderate | Heterogeneous, expert-dominated (71% expert) |
| Diffuser prior | ~50% quality | ~60-76% SR |
| Guidance room | Can go up or down | Can mostly only go down |
| Tested on different mixes? | No | We're the first to try |

## Why our guidance fails

With expert-heavy behavior data, the diffuser's anchor is already near the top. Guidance for ANY target policy (even 90% SR) tends to pull away from the expert-dominated prior, reducing SR.

### Theoretical counterargument (partially valid)

For a 90% SR target policy (similar to expert):
- `∇log π_target ≈ ∇log β` (both point toward expert-like actions)
- Net guidance `α·∇π - λ·∇β ≈ (α-λ)·∇expert` → small residual
- Should preserve SR

For a 0% SR target policy:
- `∇log π_target` and `∇log β` point in different directions
- Net guidance is large and destructive → SR should drop

**In theory**, ranking should still emerge. In practice (v0.3.2.2), even the 90% policy's SR drops to 0-5%. This suggests the gradient computation isn't producing the expected cancellation — possibly due to model architecture differences between the target scorer (diffusion policy) and BC (Gaussian MLP).

## Implications

1. **Adapting SOPE to expert-heavy data is novel** — the paper never tested this
2. **Our data mix may need to change** — either drop expert demos or collect a proper medium-quality behavior dataset
3. **Or the guidance implementation needs debugging** — the theoretical cancellation for good policies should work but doesn't
4. **This is a potential research contribution** — studying SOPE's sensitivity to behavior data distribution

## Options going forward

1. **Match SOPE exactly:** Train diffuser + BC on target rollouts only (no expert demos). Risk: 80 episodes may be too few.
2. **Collect proper behavior data:** Run a medium-quality policy (e.g., 10demos_epoch20, 52% SR) for 200+ rollouts. Use as D_β for both diffuser and BC.
3. **Debug gradient computation:** Verify that for the 90% policy, positive and negative gradients actually cancel as theory predicts.
4. **Abandon negative guidance:** Use positive-only guidance (worked in v0.2.5 single-policy setting).
