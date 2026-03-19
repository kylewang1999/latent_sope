# Addressing the Action Diversity Problem

**Date:** 2026-03-17
**Status:** PLANNING
**Context:** Diagnostic 3 (gradient direction test) showed +72% convergence on Lift vs -5.9% divergence on Hopper, proving all Lift BC policies share the same action landscape. Guidance-based multi-policy OPE requires policies that disagree on actions.

## Problem Summary

All current Lift target policies are BC_Gaussian trained on the same demonstration data with the same objective. They differ only in the amount of data (10/25/50/100/200 demos) and training epochs. This produces policies that are functionally identical from the perspective of gradient-based guidance:

- Cosine similarity of scorer gradients: 0.72–0.95 (vs 0.64 on Hopper where SOPE works)
- Gradient direction test: +72% convergence (all policies optimize toward the same actions)
- KL diversity: 7.6x less than D4RL cross-dataset

The root cause is not an implementation bug — it's a **policy diversity problem**. SOPE assumes target policies produce distinguishable action distributions. BC policies trained on the same demonstrations violate that assumption.

## Strategy: Introduce Structurally Different Policies

To achieve the divergence signal seen in Hopper (-5.9%), we need target policies that optimize different objectives or operate on different data distributions. Four approaches, ordered by expected impact:

### Option 1: Train RL Policies on Lift (Highest Impact)

Train offline RL policies (CQL, BCQ) alongside BC on the same Lift task. RL policies optimize returns rather than imitating demonstrations — they will find different action strategies, especially in states where the demonstration behavior is suboptimal.

**Why this works:** D4RL's random/medium/expert policies are different RL training snapshots. The diversity comes from different optimization objectives and convergence points, not different data. This is the closest analogue to what makes SOPE work.

**What's available in robomimic:**
- BCQ (Batch-Constrained Q-learning) — conservative offline RL, stays near demonstrations but optimizes returns
- CQL (Conservative Q-Learning) — penalizes OOD actions, produces tighter action distributions than BC
- Both support Lift out of the box with low-dim observations

**Expected diversity:** High. RL policies balance exploitation (lift the cube efficiently) vs constraint (stay near demonstrations), producing genuinely different action preferences from pure BC. Different RL algorithms (BCQ vs CQL) will also differ from each other.

**Effort:** Medium. Need to train 2-3 RL policies (~1-2 hours each on V100). Robomimic configs exist for BCQ and CQL on Lift.

### Option 2: Mix Policy Classes

Combine existing BC policies with:
- One BCQ policy
- One CQL policy
- One untrained/random policy (baseline anchor)

Even 2-3 fundamentally different policy types would produce the divergence signal. The random policy guarantees at least one policy that strongly disagrees with all others.

**Expected diversity:** High for cross-class pairs, low for within-BC pairs. The pairwise distance matrix would show block structure.

**Effort:** Low-medium. Random policy is trivial. BCQ/CQL require training.

### Option 3: BC on Different Data Distributions

Train BC on subsets of demonstrations that capture different behavioral modes:
- BC on early-episode frames only (reaching behavior)
- BC on late-episode frames only (grasping/lifting behavior)
- BC on failed/low-reward trajectories (if available)
- BC on demonstrations with injected action noise

**Why this might work:** Forces policies to specialize in different parts of the task. A "reaching specialist" and a "lifting specialist" will disagree on actions in ambiguous states.

**Expected diversity:** Moderate. Still BC with the same objective, but on genuinely different data distributions. May produce intermediate diversity (between current setup and RL policies).

**Effort:** Low. Only requires data preprocessing and retraining BC.

### Option 4: Perturb the BC Objective

Modify the BC training to produce systematically different policies:
- Different action noise injection scales during training
- Different action prediction horizons (1-step vs 4-step vs 8-step chunking)
- Reward-weighted BC with different temperature parameters
- Dropout at inference time (different dropout masks = different policies)

**Expected diversity:** Low-moderate. These are perturbations around the same optimum, not fundamentally different objectives. May be insufficient.

**Effort:** Low.

## Recommended Plan

1. **First:** Train one CQL and one BCQ policy on Lift using robomimic's existing configs. This is the minimum viable test of whether RL policies produce the needed divergence.
2. **Validate:** Run Diagnostic 3 (gradient direction test) on the new policy set {BC_10d, BC_200d, CQL, BCQ, random}. Target: mean GD convergence < 0% (divergence).
3. **If divergence confirmed:** Run the full guidance pipeline (Step 4) with the mixed policy set and evaluate multi-policy OPE ranking (Spearman rho).
4. **If still insufficient:** Escalate to Option 3 (data distribution splitting) on top of the RL policies.

## Success Criteria

- Diagnostic 3 GD convergence: **negative** (divergence, matching Hopper's -5.9%)
- Cosine similarity of scorer gradients: **< 0.70** (below Hopper's 0.64 would be ideal)
- Multi-policy Spearman rho: **> 0.5** (meaningful ranking signal)
