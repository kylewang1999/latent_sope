# Guidance for SOPE-style chunk diffusion

This note is written to be consistent with the DDPM formulation in `ebm_vi_ddpm.pdf`, the STITCH-OPE guidance equations in `stitch_ope.pdf`, and the action-only projected guidance pattern in the pasted SOPE code.

Notation
- $\tau = x_{1:H}$ is a trajectory chunk of horizon $H$.
- $x_t = (s_t, a_t)$ is the concatenated state-action vector at trajectory time $t$.
- By abuse of notation, $s_t$ may mean the state window $s_{t-W:t}$ when the policy or chunk diffusion conditions on a past window.
- $k$ denotes diffusion time.
- $\log p(\tau^k \mid \tau^{k+1})$ denotes the reverse chunk-DDPM kernel from diffusion level $k+1$ to $k$.
- $\pi(a\mid s)$ is the target policy and $\beta(a\mid s)$ is the behavior policy.
- Unless stated otherwise, all diffusion models use the standard `predict-epsilon` parameterization.

## 1 Action-score estimation for a diffusion policy in the predict-epsilon parameterization

Assume the policy diffusion model is conditional on the state context $s_t$ and only diffuses the action. For each trajectory time $t$, define the forward noising process
$$\begin{align}
q(a_t^k \mid a_t^0, s_t)
= \mathcal{N}\!\left(a_t^k; \sqrt{\bar\alpha_k}\, a_t^0, (1-\bar\alpha_k) I\right),
\qquad
a_t^k = \sqrt{\bar\alpha_k}\, a_t^0 + \sqrt{1-\bar\alpha_k}\, \epsilon_t,
\end{align}$$
with $\epsilon_t \sim \mathcal{N}(0,I)$. The reverse model can be parameterized by predicting the noise $\epsilon$ rather than $x_0$; this is the `predict-epsilon` parameterization. In that formulation, the training loss is
$$\begin{align}
L(\theta)
= \mathbb{E}_{k, a_t^0, \epsilon_t}
\left\lVert \epsilon_t - \hat\epsilon_\theta\!\left(a_t^k, k, s_t\right) \right\rVert^2
\end{align}$$
Recall that the gaussian log-likelihood is
$$\begin{align}
\log q\left(a_t^k \mid a_t^0, s_t\right)=-\frac{1}{2\left(1-\bar{\alpha}_k\right)}\left\|a_t^k-\sqrt{\bar{\alpha}_k} a_t^0\right\|^2+C
\end{align}$$
therefore the score of the **forward Gaussian corruption kernel** is immediate:
$$\begin{align}
\nabla_{a_t^k} \log q(a_t^k \mid a_t^0, s_t)
&= -\frac{a_t^k - \sqrt{\bar\alpha_k} a_t^0}{1-\bar\alpha_k}= -\frac{\epsilon_t}{\sqrt{1-\bar\alpha_k}}.
\end{align}$$
Now define the **noisy policy marginal** which is marginalized over the un-noised behavior policy distribution $\pi\left(a_t^0 \mid s_t\right)$:
$$\begin{align}
\pi_k(a_t^k \mid s_t)
:= \int q(a_t^k \mid a_t^0, s_t)\, \pi(a_t^0 \mid s_t)\, da_t^0. \tag{k-th-noisy-policy}
\end{align}$$
Intuitively this is can be thought of as convolving the uncorrupted gaussian density function $\pi(\cdot)$ with the gaussian corruption kernel $q(\cdot)$.

Differentiating the noisy marginal under the integral sign:
$$\begin{align}
\nabla_{a_t^k} \pi_k(a_t^k \mid s_t)
&=
\nabla_{a_t^k}\int q(a_t^k \mid a_t^0, s_t)\,\pi(a_t^0 \mid s_t)\,da_t^0 \\
&=
\int \nabla_{a_t^k} q(a_t^k \mid a_t^0, s_t)\,\pi(a_t^0 \mid s_t)\,da_t^0 \\
&=
\int q(a_t^k \mid a_t^0, s_t)\,\pi(a_t^0 \mid s_t)\,\nabla_{a_t^k}\log q(a_t^k \mid a_t^0, s_t)\,da_t^0.
\end{align}$$
Dividing by $\pi_k(a_t^k \mid s_t)$ and using Bayes' rule that $p(a_t^0\mid a_t^k,s_t)=\frac{q(a_t^k\mid a_t^0,s_t)\,\pi(a_t^0\mid s_t)}{\pi_k(a_t^k\mid s_t)}$,
$$\begin{align}
\nabla_{a_t^k}\log \pi_k(a_t^k\mid s_t)
&=
\int \frac{q(a_t^k\mid a_t^0,s_t)\,\pi(a_t^0\mid s_t)}{\pi_k(a_t^k\mid s_t)}
\nabla_{a_t^k}\log q(a_t^k\mid a_t^0,s_t)\,da_t^0 \\
&=
\mathbb{E}_{a_t^0\mid a_t^k,s_t}
\left[
\nabla_{a_t^k}\log q(a_t^k\mid a_t^0,s_t)
\right].
\end{align}$$
For the Gaussian forward kernel $q\left(a_t^k \mid a_t^0, s_t\right)$ this becomes 
$$\begin{align}
&\nabla_{a_t^k}\log \pi_k(a_t^k\mid s_t)
=
\mathbb{E}_{a_t^0\mid a_t^k,s_t}
\left[
-\frac{a_t^k-\sqrt{\bar\alpha_k}a_t^0}{1-\bar\alpha_k}
\right] \\
&=
\frac{\sqrt{\bar\alpha_k}\,\mathbb{E}[a_t^0\mid a_t^k,s_t]-a_t^k}{1-\bar\alpha_k} =
-\frac{1}{\sqrt{1-\bar\alpha_k}}\,\mathbb{E}[\epsilon_t\mid a_t^k,s_t],
\end{align}$$
Since a `predict-epsilon` diffusion policy is trained to approximate the posterior mean noise,
$$\begin{align}
\hat\epsilon_\theta(a_t^k, k, s_t) \approx \mathbb{E}[\epsilon_t \mid a_t^k, s_t],
\end{align}$$
we obtain the most convenient score estimator:
$$\begin{align}
\boxed{
\nabla_{a_t^k} \log \pi_k(a_t^k \mid s_t)
\approx
-\frac{\hat\epsilon_\theta(a_t^k, k, s_t)}{\sqrt{1-\bar\alpha_k}}
}
\end{align}$$
and analogously for the behavior policy, $\nabla_{a_t^k} \log \beta_k(a_t^k \mid s_t)\approx-\frac{\hat\epsilon_\beta(a_t^k, k, s_t)}{\sqrt{1-\bar\alpha_k}}$.

For a whole chunk $\tau^k = x_1^k, \ldots, x_H^k$, define the per-time-step action guidance
$$\begin{align}
\hat g^{\pi}_{a,t}(\tau^k)
:= \nabla_{a_t^k} \log \pi_k(a_t^k \mid s_t)
\approx -\frac{\hat\epsilon_\pi(a_t^k, k, s_t)}{\sqrt{1-\bar\alpha_k}},
\end{align}$$
and, if desired, the target-minus-behavior version
$$\begin{align}

\hat g^{\mathrm{pol}}_{a,t}(\tau^k)

:= \alpha\, \hat g^{\pi}_{a,t}(\tau^k) - \lambda\, \hat g^{\beta}_{a,t}(\tau^k),

\end{align}$$
with $\alpha, \lambda \geqslant 0$.

### 1.1 Tweedie's identity and the denoising score matching intuition

> [!lemma] Tweedie's identity 
> Consider the random variable $x \sim p(x)$ under additive gaussian corruption
>$$
> \tilde{x}=x+\sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I),
> $$
> Let $\tilde x\sim p_\sigma(\tilde{x})$ be the marginal density of the noisy variable. Then
>$$
> \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})=\frac{\mathbb{E}[x \mid \tilde{x}]-\tilde{x}}{\sigma^2} .
> $$

Tweedie's identity is then a convenient check of the same result. Applying it to eq(k-th-noisy-policy) with
$$\begin{align}
p_\sigma(\tilde x) \leftarrow \pi_k\left(a_t^k \mid s_t\right),\quad p(x)\leftarrow \pi(a_t^0\mid s_t),\quad \sigma^2 = 1-\bar\alpha_k,
\end{align}$$
gives use the "Tweedie form"
$$\begin{align}
&\nabla_{a_t^k}\log \pi_k(a_t^k\mid s_t)
= 
\frac{\sqrt{\bar\alpha_k}\,\mathbb{E}[a_t^0\mid a_t^k,s_t]-a_t^k}{1-\bar\alpha_k}
\end{align}$$
Moreover, since $a_t^k=\mathbb{E}[a_t^k|a_t^k, s_t]$,
$$\begin{align}
\frac{\sqrt{\bar\alpha_k}\,\mathbb{E}[a_t^0\mid a_t^k,s_t]-a_t^k}{1-\bar\alpha_k}=\frac{\mathbb{E}[\sqrt{\bar\alpha_k}a_t^0-a_t^k\mid a_t^k,s_t]}{1-\bar\alpha_k}
= 
-\frac{1}{\sqrt{1-\bar\alpha_k}}\,\mathbb{E}[\epsilon_t\mid a_t^k,s_t],
\end{align}$$
where the last equality uses the forward reparameterization
$$\begin{align}
\sqrt{\bar\alpha_k}a_t^0-a_t^k
=
\sqrt{\bar\alpha_k}a_t^0-\left(\sqrt{\bar\alpha_k}a_t^0+\sqrt{1-\bar\alpha_k}\,\epsilon_t\right)
=
-\sqrt{1-\bar\alpha_k}\,\epsilon_t.
\end{align}$$
Taking the conditional expectation given $(a_t^k,s_t)$ and dividing by $1-\bar\alpha_k$ then yields exactly the same expression as above.

This posterior-mean or "Tweedie form" also shows how to estimate the same score under a `predict-x0` parameterization: if the network directly predicts $\hat a_\theta^0(a_t^k,k,s_t)\approx \mathbb{E}[a_t^0\mid a_t^k,s_t]$, then
$$\begin{align}
\nabla_{a_t^k} \log \pi_k\left(a_t^k \mid s_t\right)
\approx
\frac{\sqrt{\bar{\alpha}_k}\,\hat a_\theta^0(a_t^k,k,s_t)-a_t^k}{1-\bar{\alpha}_k},
\end{align}$$
which is equivalent to the `predict-epsilon` form after converting between $\hat a_\theta^0$ and $\hat\epsilon_\theta$.

## 2 FiLM-conditioned action-only guidance on a trajectory chunk

Now suppose the **chunk diffusion model** generates the joint chunk $\tau^k = x_1^k, \ldots, x_H^k$ with $x_t^k = (s_t^k, a_t^k)$, and conditions on auxiliary context $c$ through FiLM-style conditioning inside the chunk denoiser. In particular, $c$ is not imposed by pinning-down a subset of coordinates of $\tau^k$; rather, it is provided as an input to the network that parameterizes the reverse dynamics.

For the current setup, we fix $c\leftarrow s$ to be identical to the conditional information consumed by the behavior policy $\pi(a|s)$.

At reverse step $k+1 \to k$, the chunk DDPM defines a conditional Gaussian reverse kernel
$$\begin{align}
p_\theta(\tau^k \mid \tau^{k+1}, c)
= \mathcal{N}\!\left(\mu_\theta^k(\tau^{k+1}, c), \Sigma_k\right),
\end{align}$$
where $\mu_\theta^k(\tau^{k+1}, c)$ is produced by the FiLM-conditioned chunk network. Therefore the **reverse-kernel score** is explicitly available:
$$\begin{align}
\nabla_{\tau^k}\log p_\theta(\tau^k \mid \tau^{k+1}, c)
= -\Sigma_k^{-1}\!\left(\tau^k - \mu_\theta^k(\tau^{k+1}, c)\right).
\end{align}$$
If the chunk model uses the standard `predict-epsilon` parameterization, it also gives the implied noisy-time conditional score of the corrupted chunk density,
$$\begin{align}
\nabla_{\tau^k}\log p_k(\tau^k \mid c)
\approx
-\frac{\hat\epsilon_{\mathrm{chunk}}(\tau^k, k, c)}{\sqrt{1-\bar\alpha_k}},
\end{align}$$
which is a full trajectory-chunk score over both state and action coordinates.

To connect this with the original STITCH-OPE logic, recall that under the MDP factorization one can write the target-trajectory density relative to the behavior-trajectory density using the per-step importance ratio. At noisy diffusion time $k$, the direct analogue is to use the target-minus-behavior **action score correction**
$$\begin{align}
g_{\mathrm{act}}(\tau^k)
:= \big( [0_{d_s}, \hat g^{\mathrm{pol}}_{a,1}], [0_{d_s}, \hat g^{\mathrm{pol}}_{a,2}], \ldots, [0_{d_s}, \hat g^{\mathrm{pol}}_{a,H}] \big),
\end{align}$$
where
$$\begin{align}
\hat g^{\mathrm{pol}}_{a,t}(\tau^k)
:=
\alpha\, \nabla_{a_t^k}\log \pi_k(a_t^k\mid s_t^k)
- \lambda\, \nabla_{a_t^k}\log \beta_k(a_t^k\mid s_t^k).
\end{align}$$
This is the same target-minus-behavior structure as STITCH-OPE Eq. (6), except that here the policy terms are evaluated at noisy actions $a_t^k$ and estimated with diffusion-policy scores. The state block of $g_{\mathrm{act}}(\tau^k)$ is zero and only the action block is nonzero.

The practical guidance rule is therefore to start from the FiLM-conditioned chunk reverse step and then add this action-only correction:
$$\begin{align}
\mu_\theta^k(\tau^{k+1}, c)
\longrightarrow
\mu_\theta^k(\tau^{k+1}, c)
+ \eta_k\, \Sigma_k\, g_{\mathrm{act}}\!\left(\mu_\theta^k(\tau^{k+1}, c)\right).
\end{align}$$
Equivalently, at the score level one can view the sampler as combining
$$\begin{align}
\nabla_{\tau^k}\log p_\theta(\tau^k \mid \tau^{k+1}, c)
\quad\text{and}\quad
g_{\mathrm{act}}(\tau^k).
\end{align}$$
The first term is the full state-action score supplied by the FiLM-conditioned chunk model, and the second term is the STITCH-style action-only importance-ratio correction.

A per-time-step view is equally helpful. Since $x_t^k = (s_t^k, a_t^k)$ and only the action score is known, the local extra guidance is
$$\begin{align}
g_t^k = \big(0_{d_s},\, \hat g^{\mathrm{pol}}_{a,t}(\tau^k)\big).
\end{align}$$
Thus the policy term directly modifies only $a_t^k$, whereas the chunk model contributes a full $(s_t^k, a_t^k)$ score through its conditioned reverse kernel.

### 2.1 Intuition for why this can still move the states

This action-only guidance is not the full target-trajectory score correction, but the states can still move because the FiLM-conditioned chunk model already supplies a full conditional state-action score while the policy term adds only an action correction; once the actions are nudged toward higher target-policy likelihood relative to the behavior policy, the chunk-model score propagates that change to compatible states under the context $c$ and learned dynamics. This is most plausible under moderate policy shift and a strong chunk model; under severe shift, the missing explicit state term $\nabla_{s_t^k} \log \pi_k(a_t^k \mid s_t)$ may matter.

### 2.2 How the guidance changes the sampling step

Without policy guidance, the reverse sampler at diffusion step $k+1 \to k$ uses the FiLM-conditioned chunk model to form the Gaussian kernel
$$\begin{align}
p_\theta(\tau^k \mid \tau^{k+1}, c)
=
\mathcal{N}\!\left(\mu_\theta^k(\tau^{k+1}, c), \Sigma_k\right),
\end{align}$$
so the nominal sampling step is simply
$$\begin{align}
\tau^k
=
\mu_\theta^k(\tau^{k+1}, c) + \Sigma_k^{1/2}\xi,
\qquad
\xi \sim \mathcal N(0,I).
\end{align}$$
When guidance is applied, the covariance is left unchanged and only the mean is shifted by the action-only correction. Using the padded policy guidance vector $g_{\mathrm{act}}$, the guided mean becomes
$$\begin{align}
\tilde\mu_\theta^k(\tau^{k+1}, c)
=
\mu_\theta^k(\tau^{k+1}, c)
+ \eta_k\,\Sigma_k\, g_{\mathrm{act}}\!\left(\mu_\theta^k(\tau^{k+1}, c)\right),
\end{align}$$
and the corresponding guided sampling step is
$$\begin{align}
\tau^k
=
\tilde\mu_\theta^k(\tau^{k+1}, c) + \Sigma_k^{1/2}\xi,
\qquad
\xi \sim \mathcal N(0,I).
\end{align}$$
Thus, relative to the nominal reverse step, guidance does not introduce a separate resampling or projection stage; it only biases the reverse-step mean in the action directions favored by the target-minus-behavior score, while the FiLM-conditioned chunk model and the original reverse covariance continue to determine how that perturbation propagates through the full trajectory state.

## 3 Connection to the SOPE code

The pasted implementation matches the projected action-only guidance picture quite closely.
### 3.1 What the chunk DDPM is doing
The chunk DDPM is defined over a tensor of shape `(batch, horizon, state_dim + action_dim)`. The forward noising is
$$\begin{align}
\tau^k = \sqrt{\bar\alpha_k}\, \tau^0 + \sqrt{1-\bar\alpha_k}\, \epsilon,
\end{align}$$
which is implemented by `q_sample(...)`. Because the chunk model uses `predict_epsilon=True`, the code first reconstructs $\hat\tau^0$ from the model’s predicted noise and then computes the Gaussian posterior mean for the reverse step. In notation,
$$\begin{align}
\hat\tau_\theta^0(\tau^k, k)
= \frac{\tau^k - \sqrt{1-\bar\alpha_k}\, \hat\epsilon_{\mathrm{chunk}}(\tau^k, k)}{\sqrt{\bar\alpha_k}},
\end{align}$$then
$$\begin{align}
p_\theta(\tau^{k-1} \mid \tau^k, c)
= q\!\left(\tau^{k-1} \mid \tau^k, \hat\tau_\theta^0(\tau^k,k,c)\right).
\end{align}$$
This is exactly the standard `predict-epsilon` DDPM construction.
### 3.2 What the policy guidance code is actually computing

  For an explicit-density policy, `gradlog(...)` sets
- `state_t.requires_grad_(False)`,
- `action_t.requires_grad_(True)`,
and differentiates the total log-likelihood only with respect to the action block. Therefore the returned guidance tensor is
$$\begin{align}

g_{\mathrm{code}}(\tau)

= \big(0_{d_s}, \nabla_{a_1} \log \pi(a_1\mid s_1), \ldots, 0_{d_s}, \nabla_{a_H} \log \pi(a_H\mid s_H)\big),

\end{align}$$
possibly normalized.

For a diffusion policy, `gradlog_diffusion(...)` is even more explicit: it calls `policy.grad_log_prob(states, actions)`, reshapes the returned action gradients, writes zeros into all state coordinates, and writes the gradients only into the action coordinates. So for the diffusion-policy path, the code is implementing exactly the padded action-only guidance vector described above.

The negative guidance term is handled by computing the same object for the behavior policy and subtracting it:
$$\begin{align}

g_{\mathrm{guide}}(\tau)

= g_{\pi,\mathrm{act}}(\tau) - \texttt{ratio}\, g_{\beta,\mathrm{act}}(\tau),

\end{align}$$
up to optional clipping and normalization.
### 3.4 Why one might iterate the guidance several times

The repeated inner loop over `k_guide` is sensible for three reasons.
1. First, classifier-style guidance is locally linear: it uses a gradient evaluated at a current expansion point. After one update, the expansion point has changed, so recomputing the guidance is more faithful than taking one very large step.
2. Projection / conditioning is re-applied after each update, the inner loop is effectively doing **projected gradient ascent** on the target-minus-behavior guidance objective inside one reverse step. A single large step could violate the conditioning constraints badly before the projection pulls the sample back.
3. Multiple small guided steps are usually more stable than a single large step. In the pasted code, the shift is not explicitly preconditioned by the reverse covariance $\Sigma_k$ as in exact classifier guidance. Instead, it is scaled by `action_scale` and optionally by a schedule multiplier. Repeating smaller projected steps partly compensates for that simplification.

## 4 If the chunk DDPM uses `predict-x0` instead of `predict-epsilon`

  
The cleanest way to phrase this is:
- the **policies** still provide action scores through their `predict-epsilon` parameterization,
- the **chunk DDPM** now directly predicts $\hat\tau_\theta^0(\tau^k, k, c)$ rather than $\hat\epsilon_{\mathrm{chunk}}(\tau^k, k, c)$.

### 4.1 How to estimate the full chunk score from a `predict-x0` chunk DDPM

The chunk DDPM does not directly give $\log p(\tau)$, but at diffusion level $k$ it does give the noisy-time conditional score of the corrupted chunk density. Starting from the forward corruption
$$\begin{align}

q(\tau^k \mid \tau^0, c)

= \mathcal{N}\!\left(\tau^k; \sqrt{\bar\alpha_k}\, \tau^0, (1-\bar\alpha_k)I\right),

\end{align}$$
its kernel score is
$$\begin{align}
\nabla_{\tau^k} \log q(\tau^k \mid \tau^0, c)
= \frac{\sqrt{\bar\alpha_k}\, \tau^0 - \tau^k}{1-\bar\alpha_k}.
\end{align}$$
Averaging over the posterior of $\tau^0$ given $\tau^k$ gives the noisy-time chunk score
$$\begin{align}

\nabla_{\tau^k} \log p_k(\tau^k \mid c)

&= \mathbb{E}\!\left[\nabla_{\tau^k} \log q(\tau^k \mid \tau^0, c) \mid \tau^k, c\right] \\

&= \frac{\sqrt{\bar\alpha_k}\, \mathbb{E}[\tau^0 \mid \tau^k, c] - \tau^k}{1-\bar\alpha_k}.

\end{align}$$
If the `predict-x0` chunk model directly outputs
$$\begin{align}

\hat\tau_\theta^0(\tau^k, k, c) \approx \mathbb{E}[\tau^0 \mid \tau^k, c],

\end{align}$$
then the full chunk score estimator is
$$\begin{align}

\boxed{

\nabla_{\tau^k} \log p_k(\tau^k \mid c)

\approx

\frac{\sqrt{\bar\alpha_k}\, \hat\tau_\theta^0(\tau^k, k, c) - \tau^k}{1-\bar\alpha_k}

}

\end{align}$$
This is a **trajectory-chunk score** with the same dimension as $\tau^k$, so it covers both state and action coordinates. If desired, one can convert the `predict-x0` output to an equivalent implied noise prediction
$$\begin{align}

\hat\epsilon_{\mathrm{chunk}}(\tau^k, k, c)

= \frac{\tau^k - \sqrt{\bar\alpha_k}\, \hat\tau_\theta^0(\tau^k, k, c)}{\sqrt{1-\bar\alpha_k}},

\end{align}$$
and then recover the same score as
$$\begin{align}

\nabla_{\tau^k} \log p_k(\tau^k \mid c)

\approx

-\frac{\hat\epsilon_{\mathrm{chunk}}(\tau^k, k, c)}{\sqrt{1-\bar\alpha_k}}.

\end{align}$$
So `predict-x0` and `predict-epsilon` differ only in which quantity the network predicts natively; the implied noisy-time score is the same.

### 4.2 Does the guidance scheme from Sections 1–3 change?

The short answer is: **the policy-guidance part does not change, but the chunk-model part becomes easier to express in score form.** The target and behavior policies are still diffusion policies in `predict-epsilon` form, so their action-only guidance remains
$$\begin{align}

\hat g^{\mathrm{pol}}_{a,t}(\tau^k)

= -\alpha\, \frac{\hat\epsilon_\pi(a_t^k, k, s_t)}{\sqrt{1-\bar\alpha_k}}

+ \lambda\, \frac{\hat\epsilon_\beta(a_t^k, k, s_t)}{\sqrt{1-\bar\alpha_k}}.

\end{align}$$
Therefore the padded whole-chunk guidance still has the same form:
$$\begin{align}

g_{\mathrm{act}}(\tau^k)

= \big( [0_{d_s}, \hat g^{\mathrm{pol}}_{a,1}], \ldots, [0_{d_s}, \hat g^{\mathrm{pol}}_{a,H}] \big).

\end{align}$$
What changes is only the **unguided chunk-model term**. With `predict-x0`, you directly get $\hat\tau_\theta^0$, from which you compute the reverse Gaussian mean or, equivalently, the chunk score. Then you add the same action-only policy guidance on top of that FiLM-conditioned chunk-model score. So the algorithmic template remains
$$\begin{align}

\text{FiLM-conditioned chunk reverse step} \quad + \quad \text{action-only policy guidance}.

\end{align}$$
The main new thing you gain is that the chunk DDPM itself now gives a natural **full chunk score estimate**
$$\begin{align}

\nabla_{\tau^k} \log p_k(\tau^k \mid c)

\approx

\frac{\sqrt{\bar\alpha_k}\, \hat\tau_\theta^0 - \tau^k}{1-\bar\alpha_k},

\end{align}$$
which may be useful for analysis or for designing more principled preconditioning, but it does not force you to change the action-only policy-guidance scheme.

## 5 Bottom line

The cleanest synthesis is:
1. A `predict-epsilon` **policy diffusion model** naturally gives a noisy-time **action score**,
$$\begin{align}
\nabla_{a_t^k} \log \pi_k(a_t^k \mid s_t)
\approx -\frac{\hat\epsilon_\pi(a_t^k, k, s_t)}{\sqrt{1-\bar\alpha_k}}.
\end{align}$$
2. If the **chunk diffusion model** is joint over $(s,a)$ but only the policy action score is available, the natural SOPE-style guidance is to **pad zeros in the state block** and add that action-only term on top of the full FiLM-conditioned chunk-model score or reverse kernel.
3. In this FiLM-conditioned view, the sampler computes an action-only guide, subtracts a behavior-policy action guide if requested, and adds the result to an already conditioned chunk reverse step.
4. If the **chunk DDPM** is switched to `predict-x0`, then the chunk model itself yields a full noisy-time **trajectory score** through
$$\begin{align}
\nabla_{\tau^k} \log p_k(\tau^k \mid c)
\approx
\frac{\sqrt{\bar\alpha_k}\, \hat\tau_\theta^0 - \tau^k}{1-\bar\alpha_k},
\end{align}$$
but the **policy-guidance part** remains the same action-only padded guidance unless you also build a model for an explicit state-gradient term.

## 6 Self checks 

1\. What does marginalize mean? Let $X,Y\in \mathbb{R}_{}^{d}$ be two random variables with, respective probability density functions (PDFs) $X\sim p_X(x)$ and $Y\sim p_Y(y)$. Express the marginalization of the joint PDF $p_{XY}(x,y)$ over $p_X(x)$.

2\. Derive the score forward Gaussian corruption kernel, that is, show that 
$$\begin{align}
\nabla_{a_t^k} \log q(a_t^k \mid a_t^0, s_t)
&= -\frac{a_t^k - \sqrt{\bar\alpha_k} a_t^0}{1-\bar\alpha_k}= -\frac{\epsilon_t}{\sqrt{1-\bar\alpha_k}}.
\end{align}$$
