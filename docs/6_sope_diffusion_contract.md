# SOPE Diffusion Contract, DDPM Parameterization, And Conditioning

This note consolidates the repository's SOPE diffusion contract and DDPM
parameterization notes.

Relevant code:

- [third_party/sope/opelab/core/baselines/diffusion/diffusion.py](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py)
- [third_party/sope/opelab/core/baselines/diffusion/helpers.py](../third_party/sope/opelab/core/baselines/diffusion/helpers.py)
- [third_party/sope/opelab/core/baselines/diffuser.py](../third_party/sope/opelab/core/baselines/diffuser.py)
- [third_party/sope/opelab/examples/d4rl/diffusion_trainer.py](../third_party/sope/opelab/examples/d4rl/diffusion_trainer.py)
- [third_party/robomimic/robomimic/algo/diffusion_policy.py](../third_party/robomimic/robomimic/algo/diffusion_policy.py)
- [third_party/robomimic/robomimic/models/diffusion_policy_nets.py](../third_party/robomimic/robomimic/models/diffusion_policy_nets.py)
- [src/diffusion.py](../src/diffusion.py)
- [src/train.py](../src/train.py)
- [src/eval.py](../src/eval.py)
- [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py)
- [src/sope_diffuser.py](../src/sope_diffuser.py)

## 1. Summary

The active local wrapper still follows the standard DDPM structure, but its
conditioning contract differs from both upstream SOPE's in-painting-style path
and robomimic's explicit FiLM-style context conditioning.

For the current repository:

- the model still denoises full state-action transition chunks
- `states_from` is the intended conditioning prefix
- `actions_from` is not part of the observed-prefix conditioning contract
- the local wrapper should be read as a SOPE-inspired implementation, not a
  line-by-line port of upstream sampling behavior

## 2. DDPM Reminder

The fixed forward process is:

$$\begin{align}
q(x_t \mid x_0)
=
\mathcal{N}\left(
x_t;
\sqrt{\bar{\alpha}_t}\, x_0,
(1 - \bar{\alpha}_t) I
\right),
\end{align}$$

with closed-form sample

$$\begin{align}
x_t
=
\sqrt{\bar{\alpha}_t}\, x_0
+
\sqrt{1 - \bar{\alpha}_t}\, \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I).
\end{align}$$

In epsilon-prediction mode, the reverse model estimates
$\epsilon_\theta(x_t, t)$ and reconstructs

$$\begin{align}
\hat{x}_0(x_t, t)
=
\frac{1}{\sqrt{\bar{\alpha}_t}}
\left(
x_t - \sqrt{1 - \bar{\alpha}_t}\,\epsilon_\theta(x_t, t)
\right).
\end{align}$$

Both `predict_epsilon=True` and `predict_epsilon=False` are valid DDPM
parameterizations, but they change optimization targets and can produce
different sampling behavior even with the same scheduler family.

## 3. How SOPE Implements Conditioning

Upstream SOPE uses in-painting-style conditioning. `apply_conditioning(...)`
does not add an external context vector; it overwrites selected tensor slices at
selected timesteps.

That conditioning acts in both training and sampling:

$$\begin{align}
\tilde{x}_t &= \text{apply\_conditioning}(x_t, \text{cond}), \\
\hat{\epsilon} &= \epsilon_\theta(\tilde{x}_t, t), \\
\hat{\epsilon}_{\text{cond}} &= \text{apply\_conditioning}(\hat{\epsilon}, \text{cond}).
\end{align}$$

During reverse sampling, conditioning is re-applied after each denoising step.
It therefore clamps parts of the trajectory tensor rather than changing the
definition of $\epsilon_\theta$.

## 4. Local Chunk Contract

The local rollout-backed dataset uses chunk fields that need to be interpreted
carefully against upstream SOPE:

- `states_from`: observed prefix state stack
- `actions_from`: aligned historical actions
- `states_to`: future state targets
- `actions_to`: future action targets

The SOPE-aligned local reading is:

- condition on `states_from`
- predict the future transition chunk
- do not treat `actions_from` as part of the observed prefix that must be
  clamped during denoising

This is why the earlier mapping that treated the full historical prefix as
supervised generated content was inconsistent with the intended SOPE contract.

## 5. SOPE Versus Robomimic Conditioning

The current repository mixes two conditioning styles:

### 5.1 SOPE style

SOPE's default diffusion baseline conditions by overwriting known trajectory
entries inside the sampled tensor.

### 5.2 Robomimic style

Robomimic diffusion policy conditions through an explicit context vector passed
into `ConditionalUnet1D`, where the context modulates convolution features via
FiLM-like scale and bias parameters.

### 5.3 Practical implication

The local `FilmConditionedBackbone` path won out as the canonical backbone
because it maps cleanly onto robomimic's tested implementation, but it should
not be described as identical to upstream SOPE conditioning.

## 6. Debug And Ablation Modes

Two debugging directions matter for this codepath:

### 6.1 EEF-only loss masking

The full state-action chunk is still modeled, but the loss can be restricted to
the end-effector position slice to test whether the backbone can learn a simpler
projection of the task.

### 6.2 EEF-only unconditioned mode

The newer debug mode changes the dataset, conditioning, and evaluation path so
the model trains on an intentionally simplified subset of the state without the
usual prefix conditioning behavior.

These modes are debugging tools. They are not the main contract that the local
SOPE wrapper is trying to preserve.

## 7. Sampling Implications

The choice between epsilon prediction and clean-sample prediction affects how
sample quality and reconstruction error should be interpreted. Low training loss
does not guarantee low rollout RMSE if:

- the target parameterization is mismatched to the sampling path
- the conditioning scheme is inconsistent with the data layout
- normalization or prefix handling is wrong

This is the main reason the diffusion and conditioning contract has to be read
together instead of treating DDPM equations and dataset mapping as independent
topics.

## 8. Validation

For behavior-changing work in this part of the stack, rerun at least:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && python -m py_compile src/diffusion.py src/train.py src/eval.py src/robomimic_interface/dataset.py src/sope_diffuser.py
```

If the change affects conditioning or normalization semantics, also rerun the
smallest chunk-MSE or rollout-evaluation workflow that exercises the updated
path.
