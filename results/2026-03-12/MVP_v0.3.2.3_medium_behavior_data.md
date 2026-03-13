# MVP v0.3.3: Medium-Quality Behavior Data (Planned)

**Date:** 2026-03-12
**Status:** PLANNED — collecting rollouts (SLURM job 7368312)

## Motivation

v0.3.2–v0.3.2.2 failed because our behavior data was expert-dominated (200 expert demos at 100% SR + 80 target rollouts). This biases the diffuser's prior too high, leaving guidance no room to steer upward for good policies.

SOPE's reference implementation trains exclusively on **medium-quality behavior data** (~50% SR) — D4RL `hopper-medium-v2`, `halfcheetah-medium-v2`, `walker2d-medium-v2`. The medium-quality anchor gives guidance bidirectional range: steer up for good target policies, steer down for bad ones.

We've been doing something SOPE never tested: mixing expert demos into the training data. Time to match SOPE's assumption and use only medium-quality rollouts as D_β.

## Plan

### Step 1: Collect 100 rollouts for 8 target policies (in progress)

Selected 8 checkpoints for a good SR spread:

| Checkpoint | Oracle SR | Role |
|---|---|---|
| 50demos_epoch10 | 0% | Negative control |
| 10demos_epoch10 | 8% | Low |
| 200demos_epoch10 | 18% | Low-medium |
| 100demos_epoch20 | 42% | Medium |
| 10demos_epoch20 | 52% | Medium |
| 10demos_epoch30 | 62% | Medium-high |
| 100demos_epoch40 | 76% | High |
| 200demos_epoch40 | 90% | Near-expert |

SLURM job 7368312 collecting 100 rollouts each. ~3-4 hours estimated.

### Step 2: Train diffuser + BC on medium-quality behavior data only

**No expert demos.** Use only the collected target rollouts as D_β:
- Option A: Pool all 800 rollouts (100 × 8 policies) as behavior data. Average quality ~50% SR.
- Option B: Use a single medium-quality policy's rollouts (e.g., 10demos_epoch20 at 52% SR, 100 rollouts) as D_β. Cleanest match to SOPE.
- Option C: Use rollouts from policies near 50% SR only (42%, 52%, 62%) as behavior data. ~300 rollouts of medium quality.

### Step 3: Evaluate with guidance

With a medium-quality anchor:
- Unguided diffuser should generate ~50% SR trajectories
- Positive guidance toward 90% policy should increase SR
- Positive guidance toward 0% policy should decrease SR
- Negative guidance (pushing away from medium behavior) should be constructive for good policies
- Ranking should emerge naturally, matching SOPE's framework

### Key questions

1. **Which data mix?** Option A gives the most data (800 episodes) but heterogeneous quality. Option B is cleanest but only 100 episodes — may be too few for good diffusion training. Option C is a middle ground.
2. **Is 100 rollouts enough per policy?** SOPE trains on thousands of D4RL trajectories. 100-800 of ours may be thin.
3. **Will the observation recording bug matter less?** With medium-quality behavior data, the diffuser doesn't need to see cube_z > 0.84 to generate useful trajectories — it just needs to capture the behavior distribution. Guidance does the steering.
4. **Guidance hyperparameters:** Should be closer to SOPE defaults (scale=0.5, ratio=0.5) now that behavior data is medium quality.

## What this tests

- Whether SOPE's guidance framework works on robomimic when behavior data quality matches the D4RL assumption
- Whether removing expert demos fixes the guidance destruction observed in v0.3.2.x
- Whether the ranking (Spearman ρ) improves with a neutral anchor
