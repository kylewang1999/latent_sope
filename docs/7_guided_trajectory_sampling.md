# Guidance for SOPE-style chunk diffusion

Notation
- $\tau = x_{1:H}$ is a trajectory chunk of horizon $H$.
	- $x_t = (s_t, a_t)$ is the concatenated state-action vector at trajectory time $t$.
	- Abusing notation, $s_t$ may mean the state window $s_{t-W:t}$ when the policy or chunk diffusion conditions on a past window.
- $k$ denotes diffusion time.
- $\log p(\tau^k \mid \tau^{k+1})$ denotes the reverse chunk-DDPM kernel from diffusion level $k+1$ to $k$.
- $\pi(a\mid s)$ is the target policy and $\beta(a\mid s)$ is the behavior policy.

## 1. Action-score estimation for a diffusion policy

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
Recall that the Gaussian log-likelihood is
$$\begin{align}
\log q\left(a_t^k \mid a_t^0, s_t\right)=-\frac{1}{2\left(1-\bar{\alpha}_k\right)}\left\|a_t^k-\sqrt{\bar{\alpha}_k} a_t^0\right\|^2+C
\end{align}$$
therefore the score of the **forward Gaussian corruption kernel** is immediate:
$$\begin{align}
\nabla_{a_t^k} \log q(a_t^k \mid a_t^0, s_t)
&= -\frac{a_t^k - \sqrt{\bar\alpha_k} a_t^0}{1-\bar\alpha_k}= -\frac{\epsilon_t}{\sqrt{1-\bar\alpha_k}}.
\end{align}$$
Now define the **noisy policy marginal**, obtained by marginalizing over the clean policy distribution $\pi\left(a_t^0 \mid s_t\right)$:
$$
\begin{align}
\pi_k(a_t^k \mid s_t)
:= \int q(a_t^k \mid a_t^0, s_t)\, \pi(a_t^0 \mid s_t)\, da_t^0.
\tag{1}\end{align}
$$

Differentiating w.r.t. $a_t^k$,
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

### 1.1 Tweedie's identity and the predict-x0 parameterization

> [!lemma] Affine-Gaussian Tweedie identity
> Consider the random variable $x \sim p(x)$ observed through the affine Gaussian channel
>$$
> \tilde x = A x + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I),
> $$
> where $A$ is a fixed matrix and $\sigma > 0$. Let
>$$
> p_{A,\sigma}(\tilde x) := \int \mathcal{N}(\tilde x; A x, \sigma^2 I)\, p(x)\, dx
> $$
> be the marginal density of the noisy variable. Then
>$$
> \nabla_{\tilde x} \log p_{A,\sigma}(\tilde x)
> =
> \frac{\mathbb{E}[A x \mid \tilde x]-\tilde x}{\sigma^2}.
> $$
> For $A = I$, this reduces to the usual additive-noise form of Tweedie's identity.

This identity is a convenient check of the same result because the DDPM forward kernel is affine-Gaussian. For fixed $s_t$, apply it to (1) with
$$\begin{align}
\tilde x \leftarrow a_t^k,\quad x \leftarrow a_t^0,\quad A \leftarrow \sqrt{\bar\alpha_k}\, I,\quad \sigma^2 \leftarrow 1-\bar\alpha_k,\quad p(x) \leftarrow \pi(a_t^0\mid s_t),
\end{align}$$
so that
$$\begin{align}
\pi_k(a_t^k\mid s_t)
=
\int \mathcal N\!\left(a_t^k;\sqrt{\bar\alpha_k}\,a_t^0,(1-\bar\alpha_k)I\right)\,\pi(a_t^0\mid s_t)\,da_t^0.
\end{align}$$
Then Tweedie's identity gives
$$\begin{align}
&\nabla_{a_t^k}\log \pi_k(a_t^k\mid s_t)
= 
\frac{\sqrt{\bar\alpha_k}\,\mathbb{E}[a_t^0\mid a_t^k,s_t]-a_t^k}{1-\bar\alpha_k}
\end{align}$$
Moreover, since $a_t^k=\mathbb{E}[a_t^k|a_t^k, s_t]$,
$$\begin{align}
\frac{\sqrt{\bar\alpha_k}\,\mathbb{E}[a_t^0\mid a_t^k,s_t]-a_t^k}{1-\bar\alpha_k}=\frac{\mathbb{E}[\sqrt{\bar\alpha_k}a_t^0-a_t^k\mid a_t^k,s_t]}{1-\bar\alpha_k}
= 
-\frac{\mathbb{E}[\epsilon_t\mid a_t^k,s_t]}{\sqrt{1-\bar\alpha_k}},
\end{align}$$
where the last equality uses that $a_t^k=\sqrt{\bar\alpha_k} a_t^0+\sqrt{1-\bar\alpha_k} \epsilon_t$. This posterior-mean or "Tweedie form" also shows how to estimate the same score under a `predict-x0` parameterization: if the network directly predicts $\hat a_\theta^0(a_t^k,k,s_t)\approx \mathbb{E}[a_t^0\mid a_t^k,s_t]$, then
$$\boxed{\begin{align}
\nabla_{a_t^k} \log \pi_k\left(a_t^k \mid s_t\right)
\approx
\frac{\sqrt{\bar{\alpha}_k}\,\hat a_\theta^0(a_t^k,k,s_t)-a_t^k}{1-\bar{\alpha}_k},
\end{align}}$$
which is equivalent to the `predict-epsilon` form after converting between $\hat a_\theta^0$ and $\hat\epsilon_\theta$.

## 2. From trajectory scores to guided chunk sampling

Now suppose the **chunk diffusion model** generates the joint chunk $\tau^k = (x_1^k, \ldots, x_H^k)$ with $x_t^k = (s_t^k, a_t^k)$, and conditions on auxiliary context $c$ through FiLM-style conditioning inside the chunk denoiser. For the current setup, we fix $c \leftarrow s$ to be identical to the conditional information consumed by the behavior policy $\pi(a \mid s)$. We have the forward noising process
$$\begin{align}
q(\tau^k \mid \tau^0, c)
= \mathcal{N}\!\left(\tau^k; \sqrt{\bar\alpha_k}\, \tau^0, (1-\bar\alpha_k) I\right),
\qquad
\tau^k = \sqrt{\bar\alpha_k}\, \tau^0 + \sqrt{1-\bar\alpha_k}\, \epsilon,
\end{align}$$
Re-using the results discussed in Section 1, we see that
$$\begin{align}
\nabla_{\tau^k}\log p_k(\tau^k \mid c)
\approx \begin{cases}
\dfrac{\sqrt{\bar{\alpha}_k} \hat{\tau}_\theta^0\left(\tau^k, k, c\right)-\tau^k}{1-\bar{\alpha}_k}&\text{(predict-x0)}\\
-\dfrac{\hat\epsilon_\theta(\tau^k, k, c)}{\sqrt{1-\bar\alpha_k}} &\text{(predict-$\epsilon$)}
\end{cases}
\tag{2}\end{align}$$

Now it remains to address
1\. How does (2) manifest in the denoising sampling process?
2\. How should the policy score estimates be used to guide that sampling process?

### 2.1 Relating trajectory score to posterior mean

Write the standard DDPM one-step coefficients as $\alpha_k := 1-\beta_k$ and $\bar\alpha_k := \prod_{j=1}^k \alpha_j$. For the chunk diffusion model, the reverse transition from diffusion level $k$ to $k-1$ is Gaussian:
$$\begin{align}
p_\theta(\tau^{k-1}\mid \tau^k, c)
=
\mathcal N\!\left(\tau^{k-1};\mu_\theta^k(\tau^k,c), \tilde\beta_k I\right),
\end{align}$$
where $\beta_k=1-\alpha_k,\; \tilde{\beta}_k=\frac{1-\bar{\alpha}_{k-1}}{1-\bar{\alpha}_k} \beta_k$, and $\mu_\theta^k\left(\tau^k, c\right)$ can be expressed as, following Section 1,
$$\begin{align}
\mu_\theta^k(\tau^k,c)
&=\begin{cases}  
\frac{\sqrt{\bar\alpha_{k-1}}\,\beta_k}{1-\bar\alpha_k}\,\hat\tau_\theta^0(\tau^k,k,c)
+
\frac{\sqrt{\alpha_k}\,(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}\,\tau^k &\text{(predict-x0)}\\
\frac{1}{\sqrt{\alpha_k}}
\left(
\tau^k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,\hat\epsilon_\theta(\tau^k,k,c)
\right)&(\text{predict-$\epsilon$})
\end{cases}  
\end{align}$$
This is the next explicit SOPE code step: once a denoised chunk estimate `tau_0`
has been formed via $\hat{\tau}_\theta^0\left(\tau^k, k, c\right)$, the reverse Gaussian moments associated with $\mu_\theta^k\left(\tau^k, c\right)$ are assembled by `q_posterior(tau_0, tau_t, t)`. It is worth noting that if `predict-epsilon` mode is chosen, the SOPE code converts $\hat{\epsilon}_\theta\left(\tau^k, k, c\right)$ to $\hat{\tau}_\theta^0\left(\tau^k, k, c\right)$ at sampling time via
$$\begin{align}
\hat{\tau}_\theta^0\left(\tau^k, k, c\right)
=
\frac{\tau^k-\sqrt{1-\bar{\alpha}_k}\,\hat{\epsilon}_\theta\left(\tau^k, k, c\right)}{\sqrt{\bar{\alpha}_k}},
\end{align}$$
This can be a potential reason for why `predict-x0` performs better empirically on robomimic state trajectories.

We now proceed to explicitly relate $\mu_\theta^k\left(\tau^k, c\right)$ to the score $\nabla_{\tau^k} \log p_k\left(\tau^k \mid c\right)$.

> [!prp] Score to reverse mean formula
> The predicted reverse posterior mean can be related to the trajectory score via 
>$$\begin{align}
>\mu_\theta^k(\tau^k,c)
>\approx
>\frac{1}{\sqrt{\alpha_k}}\left(\tau^k + \beta_k\, \nabla_{\tau^k}\log p_k(\tau^k\mid c)\right).
>\end{align}$$
^2ed263

`\begin{proof}` The proposition follows immediately by plugging (2) into [Ho et al., 2020, Eq. (11)](https://arxiv.org/pdf/2006.11239.pdf). We nevertheless spell out the longer derivation because it exposes the posterior-mean interpretation of the reverse step, which will be useful when we later discuss how an auxiliary policy score perturbs the sampler. Start with the exact reverse posterior when $\tau^0$ is known, i.e.
$$\begin{align}
q(\tau^{k-1}\mid \tau^k,\tau^0,c)
=
\mathcal N\!\left(\tau^{k-1};\mu_q(\tau^k,\tau^0),\tilde\beta_k I\right),
\tag{3}\end{align}$$

with mean ([Ho et al., 2020, Eq. (7)](https://arxiv.org/pdf/2006.11239.pdf))
$$\begin{align}
\mu_q(\tau^k,\tau^0)
=
\frac{\sqrt{\bar\alpha_{k-1}}\,\beta_k}{1-\bar\alpha_k}\,\tau^0
+
\frac{\sqrt{\alpha_k}\,(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}\,\tau^k.
\tag{4}\end{align}$$

But $\tau^0$ is unknown at sampling time, so the sampler cannot use the exact reverse posterior $q(\tau^{k-1}\mid \tau^k,\tau^0,c)$. The relevant reverse kernel is therefore the marginal posterior $q(\tau^{k-1}\mid \tau^k,c)$, obtained by integrating out the unobserved clean chunk $\tau^0$; this is the reverse-process modeling target in DDPM. By the tower-rule,
$$\begin{align}
\mathbb E_{q(\tau^{k-1}\mid \tau^k,c)}[\tau^{k-1}]
&=
\mathbb E_{q(\tau^0\mid \tau^k,c)}\!\left[
\mathbb E_{q(\tau^{k-1}\mid \tau^k,\tau^0,c)}[\tau^{k-1}]
\right] =
\mathbb E_{q(\tau^0\mid \tau^k,c)}\!\left[\mu_q(\tau^k,\tau^0)\right],
\end{align}$$
where the second equality uses (3). The learned reverse kernel $p_\theta(\tau^{k-1}\mid \tau^k,c)$ is trained to approximate this true reverse kernel, so its mean satisfies
$$\begin{align}
\mu_\theta^k(\tau^k,c)
\approx
\mathbb E_{q(\tau^0\mid \tau^k,c)}\!\left[\mu_q(\tau^k,\tau^0)\right].
\tag{5}\end{align}$$

Next, apply the same affine-Gaussian Tweedie identity used in Section 1.1 to the noised chunk marginal
$$\begin{align}
\tilde x \leftarrow \tau^k,\quad x \leftarrow \tau^0,\quad A \leftarrow \sqrt{\bar\alpha_k}\,I,\quad \sigma^2 \leftarrow 1-\bar\alpha_k,\quad p(x) \leftarrow p(\tau^0\mid c),
\end{align}$$
so that $p_k(\tau^k\mid c)=\int \mathcal N\!\left(\tau^k;\sqrt{\bar\alpha_k}\,\tau^0,(1-\bar\alpha_k)I\right)\,p(\tau^0\mid c)\,d\tau^0.$ Tweedie's identity then yields
$$\begin{align}
\nabla_{\tau^k}\log p_k(\tau^k\mid c)
=
\frac{\sqrt{\bar\alpha_k}\,\mathbb E_{q(\tau^0\mid \tau^k,c)}[\tau^0]-\tau^k}{1-\bar\alpha_k},
\end{align}$$
rearranging,
$$\begin{align}
\mathbb E_{q(\tau^0\mid \tau^k,c)}[\tau^0]
=
\frac{\tau^k+(1-\bar\alpha_k)\nabla_{\tau^k}\log p_k(\tau^k\mid c)}{\sqrt{\bar\alpha_k}}.
\tag{6}\end{align}$$

Plugging (4) and (6) into (5),
$$\begin{align}
\mu_\theta^k(\tau^k,c)
&\approx
\frac{\sqrt{\bar\alpha_{k-1}}\,\beta_k}{1-\bar\alpha_k}
\cdot
\frac{\tau^k+(1-\bar\alpha_k)\nabla_{\tau^k}\log p_k(\tau^k\mid c)}{\sqrt{\bar\alpha_k}}
+
\frac{\sqrt{\alpha_k}\,(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}\,\tau^k \\
&=
\frac{\beta_k}{\sqrt{\alpha_k}(1-\bar\alpha_k)}\,\tau^k
+
\frac{\beta_k}{\sqrt{\alpha_k}}\,\nabla_{\tau^k}\log p_k(\tau^k\mid c)
+
\frac{\sqrt{\alpha_k}\,(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}\,\tau^k \\
&=
\frac{1}{\sqrt{\alpha_k}}\left(\tau^k + \beta_k\, \nabla_{\tau^k}\log p_k(\tau^k\mid c)\right),
\end{align}$$
where the last step uses $\bar\alpha_k=\alpha_k\bar\alpha_{k-1}$ and $\beta_k=1-\alpha_k$.`\end{proof}`

### 2.2 Applying the policy score estimates to guide sampling

For each trajectory time $t$, the policy-derived action score is
$$\begin{align}
\hat g_{a,t}(\tau^k)
:=
\alpha\, \nabla_{a_t^k}\log \pi_k(a_t^k\mid s_t^k)
- \lambda\, \nabla_{a_t^k}\log \beta_k(a_t^k\mid s_t^k).
\end{align}$$
This object lives only in $\mathbb{R}^{d_a}$, so to combine it with the trajectory score one first lifts it to the full chunk space by padding zeros into the state coordinates:
$$\begin{align}
g_t^k
=
\left(0_{d_s}, \hat g_{a,t}(\tau^k)\right),
\qquad
g_{\mathrm{guide}}(\tau^k)
:=
\left(g_1^k,\ldots,g_H^k\right)\in\mathbb R^{H(d_s+d_a)}.
\end{align}$$
Substituting $\nabla_{\tau^k} \log p_k\left(\tau^k \mid c\right)\leftarrow \nabla_{\tau^k} \log p_k\left(\tau^k \mid c\right)+g_{\mathrm{guide}}(\tau^k)$ into [[#^2ed263]] results in
$$\begin{align}
\mu_{\mathrm{guided}}^k(\tau^k,c)
&\approx
\frac{1}{\sqrt{\alpha_k}}
\left(
\tau^k + \beta_k
\left[
\nabla_{\tau^k}\log p_k(\tau^k\mid c)
+
g_{\mathrm{guide}}(\tau^k)
\right]
\right) \\
&=
\mu_\theta^k(\tau^k,c)
+
\frac{\beta_k}{\sqrt{\alpha_k}}\, g_{\mathrm{guide}}(\tau^k).
\end{align}$$
This gives a clean interpretation of score guidance in the chunk-DDPM sampler: guidance simply shifts the unguided reverse-step mean $\mu_\theta^k(\tau^k,c)$ by the scaled action-only correction $\frac{\beta_k}{\sqrt{\alpha_k}}\, g_{\mathrm{guide}}(\tau^k)$.

So, once the reverse kernel is written in score form as in Proposition 2.1, ==there is no separate "type mismatch"==: an additive score correction induces a corresponding additive shift of the reverse posterior mean. ==What remains problematic is that the padded vector $g_{\mathrm{guide}}(\tau^k)$ is not the score of the same joint chunk model $p_k(\tau^k\mid c)$==; it is an action-only auxiliary score built from the target and behavior policies. In practice, the code allows the user to replace the exact DDPM prefactor $\beta_k/\sqrt{\alpha_k}$ by a user-chosen scale such as `action_scale`, which makes the procedure even more heuristic.

An additional implementation heuristic in SOPE code is to apply the guidance term $\frac{\beta_k}{\sqrt{\alpha_k}}\, g_{\mathrm{guide}}(\tau^k)$ several times with a smaller step size, updating the current noisy chunk after each sub-step and then re-evaluating $g_{\mathrm{guide}}$ at the new point. 

Another implementation detail is that the upstream SOPE diffusion-policy guidance path assumes a **single-step action-score contract**. The diffusion policy is configured to predict one action at a time, and `policy.grad_log_prob(state, action)` is therefore expected to return the score of a single action conditioned on a single state, not the joint score of a full action chunk. Accordingly, if the current trajectory chunk has shape $(N,T,d_s+d_a)$, the effective logic is:

1. start with chunk tensor $(N, T, d_s + d_a)$
2. extract per-timestep state/action pairs
3. flatten to a batch of $NT$ pairs
4. query `policy.grad_log_prob` on those pairs
5. reshape the returned action scores to $(N, T, d_a)$
6. pad zeros on the state coordinates to obtain $(N, T, d_s + d_a)$

So the guidance term is computed timestep-wise: each $(s_t^k,a_t^k)$ pair receives its own action score, rather than the policy producing a joint score for the whole action chunk. This is a natural fit for the upstream SOPE diffusion policy, but now that our robomimic diffusion-policy backbone predicts action chunks, how best to query the action score requires more careful thought.

## 3. Repository code map

The derivations above should be treated as the authoritative reference for the diffusion equations and guidance math. The links below are the concrete repository entrypoints that implement the same contracts.

### 3.1 Chunk-DDPM reverse mean and parameterization

- [`FilmGaussianDiffusion.predict_start_from_noise`](../src/diffusion.py#L279) handles the `predict-epsilon` versus `predict-x0` contract and reconstructs $\hat{\tau}_\theta^0(\tau^k, k, c)$ from the denoiser output.
- [`FilmGaussianDiffusion.q_posterior`](../src/diffusion.py#L288) assembles the closed-form reverse Gaussian moments once $\hat{\tau}_\theta^0$ has been formed.
- [`FilmGaussianDiffusion.p_mean_variance`](../src/diffusion.py#L328) runs the denoiser, forms `x_recon`, and returns the unguided reverse-step mean and variance used by the sampler.
- [`FilmGaussianDiffusion.p_sample_loop`](../src/diffusion.py#L345) and [`FilmGaussianDiffusion.conditional_sample`](../src/diffusion.py#L372) expose the public guided chunk-sampling path.

### 3.2 Guidance mean shift

- [`guided_sampling`](../src/sampling.py#L234) is the local implementation of the mean-shift heuristic: it calls `model.p_mean_variance(...)`, leaves the model variance unchanged, and adds the guidance tensor directly to `model_mean`.
- [`_compute_film_policy_gradient`](../src/sampling.py#L124), [`_compute_film_negative_gradient`](../src/sampling.py#L143), [`_combine_film_guide`](../src/sampling.py#L166), and [`_scale_film_guide`](../src/sampling.py#L191) implement the target-policy term, behavior-policy subtraction, guide combination rule, and user- or schedule-dependent scaling.
- [`run_film_p_sample_loop`](../src/sampling.py#L308) repeats that guided reverse-step update over the full diffusion chain and optionally records per-step model predictions and guidance tensors.

### 3.3 Single-step action-score contract

- [`gradlog_diffusion`](../third_party/sope/opelab/core/baselines/diffusion/diffusion.py#L94) is the upstream SOPE helper that realizes the timestep-wise contract described at the end of Section 2.2: flatten the $N \times T$ state-action pairs, call `policy.grad_log_prob`, reshape, and pad zeros on the state coordinates.
- [`DiffusionScorePolicy`](../src/sampling.py#L18) records the corresponding local interface: `grad_log_prob(state, action)`.
- [`DiffusionPolicy.grad_log_prob`](../src/robomimic_interface/policy.py#L144) is the current robomimic adapter. It converts the denoiser output into the `predict-epsilon` score estimate $-\hat{\epsilon}/\sqrt{1-\bar{\alpha}_k}$, but it currently uses the fixed timestep specified by [`DiffusionPolicyScoreConfig.score_timestep`](../src/robomimic_interface/policy.py#L12) rather than the active chunk-sampler timestep.
- [`DiffusionPolicy._prepare_obs_cond`](../src/robomimic_interface/policy.py#L67) and [`DiffusionPolicy._prepare_action_sequence`](../src/robomimic_interface/policy.py#L96) show how a single `(state, action)` pair is expanded to match the robomimic observation and action horizons before the score is queried.

### 3.4 Conditioning semantics

- [`apply_conditioning`](../third_party/sope/opelab/core/baselines/diffusion/helpers.py#L159) is the upstream in-painting-style conditioning primitive discussed in the older SOPE contract note.
- [`FilmConditionedBackbone`](../src/diffusion.py#L183) and [`FilmGaussianDiffusion.p_losses`](../src/diffusion.py#L416) are the local FiLM-conditioned counterparts: the prefix is supplied through `cond`, not by clamping entries inside the trajectory tensor after each denoising step.
