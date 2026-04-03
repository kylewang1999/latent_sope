# Handoff: Create MVP v0.2.5.16 Dense Reward Experiment

## Goal

Create `experiments/2026-04-03/MVP_v0.2.5.16_dense_reward.ipynb` that builds on v0.2.5.14d (`experiments/2026-03-26/MVP_v0.2.5.14d_action_sweep_large.ipynb`) but replaces the sparse/sigmoid reward with the dense staged reward from `results/2026-04-03/dense_reward_design.md`.

## What's Done

1. Read the dense reward design doc — 3-component staged reward: reach + grasp + lift
2. Read the full v0.2.5.14d notebook (15 cells) — action scale sweep with guidance
3. **Calibrated gripper_qpos thresholds** from 50 rollout files — THIS IS CRITICAL:

### Calibration Results (from rollouts/target_policy_50/)

**The design doc's grasp formula is WRONG for this data.** It uses `q1 + q2` as "total gripper opening", but:
- `q1` ranges 0.001 (closed) to 0.040 (open) — positive
- `q2` ranges -0.040 (open) to -0.001 (closed) — negative (opposite sign!)
- `q1 + q2 ≈ 0` always (range: -0.015 to +0.004) — useless for grasp detection

**Fix:** Use `q1` alone (index 17) as gripper opening. Low = closed, high = open.

Recalibrated thresholds:
- **Grasp closure:** `sigma((-q1 + 0.02) / 0.005)` — fires when q1 < 0.02 (closed)
  - Open (q1=0.039): sigma(-3.8) ≈ 0.02 ✓
  - Closed (q1=0.001): sigma(3.8) ≈ 0.98 ✓
- **Grasp proximity:** `sigma((-dist + 0.03) / 0.01)` — fires when dist < 0.03m
  - Close (dist=0.01): sigma(2) ≈ 0.88 ✓
  - Far (dist=0.1): sigma(-7) ≈ 0.001 ✓
- **Reach:** `exp(-10 * dist)` — as designed, works fine
  - dist range: 0.003–0.238, median 0.075
- **Lift:** `clip((cube_z - 0.8) / 0.1, 0, 1)` — as designed
  - cube_z range: 0.817–0.840

### Distance stats
- `object[7:10]` = gripper_to_cube displacement → `dist = norm(object[7:10])`
- min=0.003, max=0.238, mean=0.090, median=0.075

## What Needs To Be Done

### 1. Create the notebook

Base it on v0.2.5.14d (copy structure exactly). Key changes:

**Cell 2 (reward functions):** Replace `hard_reward`/`sigmoid_reward` with:
```python
def dense_reward(states):
    """Dense staged reward: reach + grasp + lift. states: (B, T, 19) or (T, 19)"""
    squeeze = states.ndim == 2
    if squeeze:
        states = states[None]
    
    # Extract obs
    cube_z = states[:, :, 2]                          # (B, T)
    disp = states[:, :, 7:10]                         # (B, T, 3)
    dist = np.linalg.norm(disp, axis=-1)              # (B, T)
    q1 = states[:, :, 17]                             # (B, T) gripper opening
    
    # Reach: exp(-beta * dist)
    r_reach = np.exp(-10.0 * dist)
    
    # Grasp: sigmoid(closed) * sigmoid(close)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    r_grasp = sigmoid((-q1 + 0.02) / 0.005) * sigmoid((-dist + 0.03) / 0.01)
    
    # Lift: clip((z - 0.8) / 0.1, 0, 1)
    r_lift = np.clip((cube_z - 0.8) / 0.1, 0.0, 1.0)
    
    # Weighted sum
    r = 0.3 * r_reach + 0.3 * r_grasp + 0.4 * r_lift
    
    if squeeze:
        return r[0]
    return r.astype(np.float32)
```

Keep `hard_reward` and `sigmoid_reward` too for comparison. Add:
```python
def compute_ope_dense(states, gamma=1.0):
    rewards = dense_reward(states)  # (B, T)
    return (rewards * (gamma ** np.arange(states.shape[1]))[None, :]).sum(axis=1)
```

**Cell 9 (sweep):** Add `ope_dense` to sweep_results alongside `ope_hard` and `ope_sigmoid`.

**Cell 10 (metrics):** Add `rho_dense`, `rmse_dense`, `log_rmse_dense`, `regret_dense` metrics.

**Cells 11-14 (output/plots):** Add dense reward to all tables and figures. Include a new panel/line for dense reward in every plot.

**Cell 8 (unguided):** Add `unguided_dense = compute_ope_dense(unguided_states, GAMMA)`.

Everything else (imports, paths, diffuser loading, trajectory generation, scorer loading) stays identical to v0.2.5.14d.

### 2. Bug-check before submission

Verify:
- All imports resolve
- `dense_reward()` handles the (B, T, 19) shape correctly
- No division by zero in normalization
- Sigmoid doesn't overflow (values are small enough, should be fine)
- Paths exist: ORACLE_JSON, TARGET_ROLLOUT_DIR, DIFFUSION_SAVE_DIR, DEMO_HDF5
- `apply_conditioning` import from correct module
- `EMA` usage matches actual API
- All sweep_results keys are accessed consistently

### 3. Submit to SLURM

```bash
bash scripts/submit_notebook.sh experiments/2026-04-03/MVP_v0.2.5.16_dense_reward.ipynb
```

### 4. After completion, log results

Write `results/2026-04-03/v0.2.5.16_dense_reward_results.md` with key metrics.

## Key Files

- **Base notebook:** `experiments/2026-03-26/MVP_v0.2.5.14d_action_sweep_large.ipynb`
- **Dense reward design:** `results/2026-04-03/dense_reward_design.md`
- **Reward model code:** `src/latent_sope/eval/reward_model.py`
- **Rollout data:** `rollouts/target_policy_50/rollout_*.h5`
- **Diffuser weights:** `diffusion_ckpts/mvp_v0252_traj_mse/diffusion_model_ema.pt`
- **Oracle results:** `results/2026-03-12/oracle_eval_all_checkpoints.json`
