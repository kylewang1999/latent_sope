# Project Roadmap

## 1. Current Direction

The current repository centers on a local SOPE-inspired diffusion stack built
around:

- [src/diffusion.py](../src/diffusion.py) for model and wrapper logic
- [src/sampling.py](../src/sampling.py) for local sampling and guided sampling
- [src/train.py](../src/train.py) for training orchestration
- [src/eval.py](../src/eval.py) for chunk, rollout, and OPE evaluation
- [src/robomimic_interface/policy.py](../src/robomimic_interface/policy.py) for
  robomimic diffusion-policy guidance

The main research direction is to make the local rollout-backed diffuser
numerically consistent enough that chunk RMSE, rollout MSE, guidance behavior,
and OPE estimates can be interpreted together.

## 2. Active Workstreams

### 2.1 Guidance Semantics

- keep the local guidance contract diffusion-only
- validate how much error is introduced by the fixed
  `DiffusionPolicyScoreConfig.score_timestep` approximation
- improve confidence that guided and unguided sampling differ only through the
  intended action-space drift term

### 2.2 Rollout Quality

- track chunk RMSE, rollout MSE, and autoregressive trajectory quality together
- continue checking whether FiLM conditioning and prefix handling are aligned
  with the rollout dataset contract
- use the canonical docs below instead of older meeting-log notes when
  reasoning about current behavior

### 2.3 Reward Modeling

- keep the reward predictor as a transition-level immediate-reward regressor
- stress-test sparse-reward behavior before changing the model family
- treat reward-class imbalance and reward-transform conventions as the next
  likely sources of error if OPE quality stalls

## 3. Canonical Docs

Use these notes as the current owners of each topic:

- [docs/1_robomimic_diffusion_score_guidance.md](./1_robomimic_diffusion_score_guidance.md)
  for robomimic guidance, FiLM conditioning, parameterization caveats,
  visual-backbone behavior, the local policy-score surface, and horizon /
  checkpoint semantics for diffusion-policy guidance
- [docs/2_reward_predictor.md](./2_reward_predictor.md) for reward-model
  semantics
- [docs/3_training_pipeline.md](./3_training_pipeline.md) for training and
  evaluation entrypoints together with rollout-dataset workflows,
  environment-bootstrap behavior, chunk-contract semantics, and debug modes
- [docs/5_autoregressive_trajectory_generation.md](./5_autoregressive_trajectory_generation.md)
  for rollout generation, guided and unguided trajectory OPE, and rollout
  reporting metrics
- [docs/6_guided_trajectory_sampling.md](./6_guided_trajectory_sampling.md)
  for the detailed action-guidance derivation, DDPM guidance math, and
  score-contract interpretation

## 4. Operational Notes

### 4.1 Worktree Sync Strategy

This repository is a worktree family under `/home/kyle/repos/wkt_sope/`:

- `main/` is the primary checkout and owns the shared Git admin directory
- `rei/` and `celina/` are linked worktrees
- `rei/.git` and `celina/.git` are pointer files, not standalone repositories

When syncing this repo family to another server, do not `rsync` copied `.git`
artifacts directly. The linked-worktree `.git` pointer files reference
machine-local absolute paths under `main/.git/worktrees/...`, so copying them
naively usually produces broken remote worktrees.

### 4.2 Recommended Remote Bootstrap

The safe pattern is:

1. create a real Git checkout for `main/` on the remote
2. recreate `rei/` and `celina/` there with `git worktree add`
3. `rsync` the full worktree family while excluding every `.git` path

If the remote can reach GitHub:

```bash
ssh REMOTE 'mkdir -p ~/wkt_sope'
ssh REMOTE 'git clone https://github.com/kylewang1999/latent_sope.git ~/wkt_sope/main'
ssh REMOTE 'cd ~/wkt_sope/main && git fetch origin main Rei Celina'
ssh REMOTE 'cd ~/wkt_sope/main && git checkout main'
ssh REMOTE 'cd ~/wkt_sope/main && git submodule update --init'
ssh REMOTE 'cd ~/wkt_sope/main && git worktree add ../rei Rei'
ssh REMOTE 'cd ~/wkt_sope/main && git worktree add ../celina Celina'
```

If the remote cannot reach GitHub, seed it from a bundle created from the local
`main/` checkout:

```bash
git -C /home/kyle/repos/wkt_sope/main bundle create /tmp/latent_sope.bundle --all
scp /tmp/latent_sope.bundle REMOTE:~/wkt_sope/
ssh REMOTE 'git clone ~/wkt_sope/latent_sope.bundle ~/wkt_sope/main'
ssh REMOTE 'cd ~/wkt_sope/main && git checkout main'
ssh REMOTE 'cd ~/wkt_sope/main && git submodule update --init'
ssh REMOTE 'cd ~/wkt_sope/main && git worktree add ../rei Rei'
ssh REMOTE 'cd ~/wkt_sope/main && git worktree add ../celina Celina'
```

Notes:

- `git submodule update --init` is the safe default for this repo
- avoid `--recursive` until the nested `third_party/sope` submodule layout is
  cleaned up
- the remote worktree layout should be created by Git, not by copied `.git`
  files

### 4.3 Repeated Sync Step

After the remote bootstrap exists, sync the whole worktree family from the
parent directory:

```bash
rsync -az --delete --info=progress2 \
  --exclude='.git' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='.mypy_cache/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.DS_Store' \
  --exclude='*.pyc' \
  --exclude='logs/' \
  /home/kyle/repos/wkt_sope/ \
  REMOTE:~/wkt_sope/
```

The important detail is `--exclude='.git'`, not `--exclude='.git/'`. The
former excludes both Git directories and Git pointer files named `.git`.

If [`rsync_to_carc.sh`](../rsync_to_carc.sh) is reused for this workflow, it
should follow the same rule and exclude `.git` rather than `.git/`.

## 5. Validation Priorities

When diffusion behavior changes, prefer the smallest checks that still exercise
the affected contract:

1. `py_compile` on the touched modules in the `latent_sope` environment
2. one deterministic guided-versus-unguided chunk-sampling smoke check
3. one small chunk or rollout evaluation if the change affects training,
   sampling, normalization, or reward accounting
