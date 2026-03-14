# Two Fundamental Problems with Diffusion Guidance for OPE

**Date:** 2026-03-13
**Based on:** v0.2.5.1–v0.2.5.7 + cross-policy gradient debug

## Problem 1: Quality-Consistency Tradeoff

**Source:** v0.2.5.7 action log-prob diagnostic

Guidance pushes actions toward the target policy but corrupts trajectory dynamics.
The diffuser learned p(states, actions) jointly — guidance perturbs actions at every
denoising step (256 steps) without updating states to match, creating dynamically
inconsistent trajectories.

**Evidence:**
- Stronger guidance monotonically lowers action NLL (actions become more target-like)
- But SR drops: 60% (unguided) → 58% (scale=0.0003) → 46% (scale=0.005)
- Best operating point (scale=0.0003) gives statistically significant improvement
  (p=0.003, 70% win rate) but small effect size (Cohen's d=0.238)
- v0.2.5.5 showed MSE and OPE don't correlate — guidance improves action fidelity
  at the cost of trajectory coherence

**Implication:** Even with a perfect scorer, guidance is capped at small effect sizes
because the trajectory quality degrades before the guidance signal can meaningfully
shift outcomes.

## Problem 2: Scorer Indistinguishability Across Policies

**Source:** Cross-policy gradient debug of v0.2.5.6

The diffusion score function at t=5 (sigma=0.105) gives nearly identical gradient
directions for policies with 42–90% SR (cosine similarity 0.86–0.95). The score
reflects the shared UNet architecture's prior about "what clean Lift actions look
like," not policy-specific behavior.

**Evidence:**
- Pairwise cosine similarity between 42%, 54%, 82%, 90% SR policies: 0.86–0.95
- Gradient magnitudes don't correlate with oracle SR (Spearman=0.54, p=0.27)
- v0.2.5.6 cross-policy test: guided OPE is flat at 58–62% for all 6 target
  policies (oracle ranges 8–90%), Spearman rho=-0.10

**Implication:** Guidance cannot rank policies because the scorer points in the same
direction for all of them. This is not a scale problem — increasing guidance strength
just pushes harder in the same (shared) direction.

## How They Compound

Even if problem 2 is fixed (e.g., with a BC-Gaussian scorer that gives
policy-specific gradients), problem 1 still limits how much guidance can steer
trajectories before corrupting dynamics. And even if problem 1 is fixed (e.g.,
late-step guidance), problem 2 means the steering signal is the same for all policies.

Both must be addressed for cross-policy guidance to work.
