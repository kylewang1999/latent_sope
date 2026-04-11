# AGENTS.md


This file is a reusable template for research-oriented codebases. It keeps the structure and operating patterns of the original project instructions while removing repository-specific assumptions.

## 1. Project roadmap

- Maintain a short roadmap document in `docs/` that explains the current research direction, major milestones, and active workstreams.
- When a roadmap exists, treat it as the default high-level reference before making structural changes.

## 2. Repository expectations

### 2.1. Documentation

- For any non-trivial change to training, inference, evaluation, data processing, or model architecture code, create a short companion note in `docs/`.
- The note should link directly to the relevant code, summarize the behavior before and after the change, and record any important assumptions, equations, ablations, or implementation caveats.
- Prefer small, focused documents over broad narrative writeups. One change or subsystem per file is usually the right granularity.
- If the repository uses numbered docs, keep the numbering monotonic, for example `1_overview.md`, `2_training.md`, `3_sampling.md`.
- For `.md` files in `docs/`, number second- and third-level headings consistently, for example `## 1. Section` and `### 1.1 Subsection`.
- If a change affects experimental behavior, document what should be re-run to validate it.
- Treat `docs/reports/` as reserved for the user's manual advisor-meeting or weekly-summary notes. Do not create or update files under `docs/reports/` unless the user explicitly asks for that location.
- When a new technical or operational note is needed by default, place it directly under `docs/` or another explicitly requested topical subdirectory, not under `docs/reports/`.

### 2.2. Equation formatting

- When generating `.md` files or Markdown snippets, wrap multiline equations as `$$\begin{align} ... \end{align}$$`.
- Use `$...$` for inline equations in Markdown.
- Do not use `\[ ... \]` for multiline equations in Markdown output.
  """_summary_
  """- Use `\mathbb{E}` for expectation, `\text{Var}` for variance, and `\text{Cov}` for covariance.
- Replace `\geq` with `\geqslant`, and `\leq` with `\leqslant`.

### 2.3. Permissions

- Prefer making reasonable local assumptions and proceeding without unnecessary permission prompts.
- If a task requires inspecting notebook structure, generated artifacts, or adjacent context before editing, default to the least disruptive read path first.
- Only escalate permissions when the task genuinely requires it, and keep the scope of escalation narrow.

### 2.3.1. Test environment

- Before running validation, smoke tests, training entrypoints, or other repo-local execution checks, activate the conda environment `latent_sope`.
- Prefer commands of the form `source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && <command>` so the environment choice is explicit and reproducible in logs.

### 2.4. Worktree structure

- The repository is organized as a small git worktree family under `/home/kyle/repos/wkt_sope/`:
  - `main/`: the primary worktree for the `main` branch. Treat this as the authoritative branch and the default place for actively maintained code.
  - `rei/`: a linked worktree used for branch `Rei`, primarily for undergraduate mentee development.
  - `celina/`: a linked worktree used for branch `Celina`, primarily for undergraduate mentee development.
- Unless the task explicitly targets student work, default to operating in `main/`.
- Be careful when comparing, copying, cherry-picking, rebasing, or merging across these worktrees because they share the same underlying repository.
- Do not treat code from `rei/` or `celina/` as trusted by default. Review the implementation carefully, validate behavior locally, and verify assumptions before bringing any of that code into `main`.
- Prefer selective integration strategies such as targeted cherry-picks or manual patches over broad merges from `Rei` or `Celina` into `main` unless the user explicitly asks for a full merge.
- When conflicts arise between `main` and student branches, preserve the intent and correctness of `main` unless careful review shows the student change is correct.

### 2.5. Directory structure conventions

- `src/`: primary source code.
- `scripts/`: training, evaluation, reproduction, and onboarding entrypoints.
- `docs/`: design notes, research notes, and change-oriented technical documentation.
- `data/`: datasets, cached preprocessing outputs, or dataset metadata when versioning policy allows.
- `logs/`: local logs, run outputs, and temporary experiment artifacts when appropriate.
- `third_party/`: vendored code as submodules.
- If the project relies on many filesystem locations, keep a central path utility or registry instead of scattering path construction across the codebase.

### 2.6. Configuration conventions

- Configuration classes should be dataclasses and should be colocated with the object or subsystem they configure unless there is a strong reason not to.
- Use clear names such as `ModelConfig`, `TrainerConfig`, `DatasetConfig`, or `<Subsystem>Config`.
- Prefer explicit typed fields with sensible defaults over unstructured dictionaries.
- Keep configuration stable enough to serialize into experiment metadata for reproducibility.

Example:

```python
@dataclass
class EnvironmentConfig:
    name: str = "example-env"
    dataset_path: Path | None = None
    horizon: int | None = None
    max_episode_steps: int | None = None
```

### 2.7. Training and experiment conventions

- Use a single epoch-level progress bar for long-running training loops and update it with averaged metrics at the end of each epoch.
- Log training, validation, and evaluation metrics to the project-standard experiment tracker, for example `wandb` if that is the chosen tool.
- Record enough metadata to reproduce a run: code version, config, seed, dataset identifier, and checkpoint path.
- Prefer explicit validation and evaluation entrypoints over embedding all checks inside the training loop.
- When changing training behavior, state whether the change affects optimization, sampling, data flow, or only logging and instrumentation.
- When training from many rollout trajectory files, prefer file-level splits over chunk-level splits for train / eval so chunks from the same trajectory do not leak across splits.
- For count-like scheduling knobs such as `num_evals` and `num_saves`, convert them into epoch intervals inside the training code. `num_evals` should use floored division `epochs // num_evals` with a clamp to at least `1` when enabled, while `num_saves` should remain conservative and use a ceil-based interval so requested checkpoint counts are not undershot.

### 2.8. Reproducibility

- Use explicit random seeds for training and evaluation whenever practical.
- Keep defaults deterministic enough for debugging, while allowing opt-in performance-oriented settings when needed.
- Avoid hidden state in scripts. Important inputs should come from config, CLI arguments, or clearly named environment variables.
- If a result depends on external assets, document where they come from and how they are versioned.

### 2.9. Research code design patterns

- Prefer simple, inspectable implementations over abstract frameworks unless the abstraction clearly reduces repeated complexity.
- Separate core math or model logic from orchestration code such as CLI parsing, logging, checkpointing, and plotting.
- Add small helper functions around repeated tensor transformations, schedule computations, or batching logic rather than duplicating opaque code blocks.
- Preserve a clear path from paper equation or algorithm step to implementation.
- If behavior is subtle, add a short code comment explaining the invariant or the reason a step exists.

### 2.10. Validation expectations

- For behavior-changing modifications, run the smallest meaningful validation available before considering the task complete.
- Prefer targeted tests, smoke runs, shape checks, or one-batch sanity passes over no validation.
- If full training is too expensive, document the cheaper verification that was run and the residual risk.

### 2.11. Change hygiene

- Do not mix unrelated refactors into research changes unless the cleanup is required to make the change safe or understandable.
- Preserve backwards compatibility for configs, checkpoints, and scripts when practical. If not practical, document the migration clearly.
- When a file becomes difficult to reason about, improve naming and structure first, then make the behavioral change.
