# TODOs

## 2026-03-26, W13

1\. Q: How is [`rei/src/latent_sope/robomimic_interface/guidance.py`](../../rei/src/latent_sope/robomimic_interface/guidance.py) used?
- Implement my version of the guidance and check if it works.
2\. Verify SOPE diffusion is good on [`rmimic-lift-ph-lowdim_diffusion_260130`](../../data/policy/rmimic-lift-ph-lowdim_diffusion_260130/rollout.mp4).
3. Is normalization stats computed/used properly?
 - 2026-03-28 update: fixed a dataset normalization bug where `actions_from`
   remained in raw units while `states_from`, `states_to`, and `actions_to`
   were normalized. The bug affected the historical prefix concatenated into
   the diffusion training target in
   [src/sope_diffuser.py](../src/sope_diffuser.py).
 - 2026-03-28 follow-up: the evaluator now reports raw-space chunk MSE,
   normalized-space chunk MSE, a persistence baseline, and dataset scale
   summaries from [src/eval.py](../src/eval.py) and
   [src/robomimic_interface/dataset.py](../src/robomimic_interface/dataset.py).
 - 2026-03-28 update: [src/sope_diffuser.py](../src/sope_diffuser.py) now
   treats the historical prefix as state-conditioning only, matching the
   original SOPE conditioning scheme more closely. Prefix action channels are
   removed from the supervised target and prefix-step loss weights are zeroed.
 - See [docs/10_sope_diffusion_contract.md](./10_sope_diffusion_contract.md)
   for the merged note covering chunk-field mapping, DDPM `q` vs `p_\theta`,
   and how `apply_conditioning(...)` acts around the denoiser and reverse
   process.
 - Follow-up: re-run chunk-MSE evaluation after this fix and compare the model
   against both normalized-space reconstruction and the persistence baseline.


## 2026-03-11, W9

### Questions & Discussions

1. Investigate SOPE diffusion guidance when the behavior policy $\beta$ and target policy $\pi$ have different input signatures, such as visual embeddings from different models.
  - Note from Yutai: SOPE primarily uses a GMM backbone to evaluate log-likelihoods.
  - Check how to use the [robomimic diffusion policy](./third_party/robomimic/robomimic/algo/diffusion_policy.py) for SOPE guidance, and whether it exposes the necessary API.
  - Clarify how SOPE should handle mismatched input signatures between $\beta(\cdot)$ and $\pi(\cdot)$.
  - Consider using the ELBO as an estimated score.
2. Figure out how to obtain a reward predictor $\hat R(s, a)$ for sparse-reward tasks.
  - In [SOPE Fig. 1.C](https://stitch-ope.github.io/paper/paper.pdf#page=4.61), the paper says STITCH-OPE trains a neural network on behavior transitions to predict the immediate reward.
  - Review [reward.py](third_party/sope/opelab/core/reward.py).
  - Dose SOPE's experiment with D4RL use sparse or dense reward? Note that Robomimic's reward for the lift task uses **sparse reward.**

### GPTs Plans

1. Trace the SOPE guidance path end to end.
  - Read the SOPE code that computes behavior-policy likelihoods or scores, starting from the policy evaluation entry point and following calls into the guidance logic.
  - Identify the exact contract SOPE assumes for $\beta$: input tensors, output format, and whether it requires normalized log-likelihoods, scores, or only relative ranking.
  - Check whether those assumptions are tied specifically to the current GMM-style backbone or are abstract enough to support other policy families.
  - Short answer: the actual guidance path is `examples/diffusion_policy/main_diffusion.py -> core/baselines/diffuser.py -> core/baselines/diffusion/diffusion.py`, and it does not use a generic scorer abstraction.
  - Short answer: for non-diffusion policies, guidance expects `log_prob_extended(state, action)` or `gaussian_log_prob(state, action)` and differentiates them w.r.t. `action`; for SOPE's own `DiffusionPolicy`, it special-cases `grad_log_prob(state, action)` directly.
  - Short answer: the contract is therefore "same raw state/action tensors as the diffusion model, plus a differentiable action-score interface", not "any policy with comparable rankings". In the current code, this is tied to the existing policy classes rather than a clean backbone-agnostic API.
2. Audit the robomimic diffusion policy interface against SOPE's needs.
  - Inspect [diffusion_policy.py](./third_party/robomimic/robomimic/algo/diffusion_policy.py) for methods that can expose action likelihood surrogates, denoising scores, ELBO terms, or intermediate noise-prediction outputs.
  - Determine whether the current API is sufficient as-is, or whether SOPE would need an adapter layer that wraps robomimic's diffusion policy in a GMM-like scoring interface.
  - Write down the minimum API needed for integration, such as `score(obs, act)`, `log_prob(obs, act)`, or `elbo(obs, act)`.
  - Short answer: robomimic's implementation exposes training via noise-prediction MSE and rollout via `get_action` / `_get_action_trajectory`; it does not expose `log_prob`, ELBO, or a policy score API.
  - Short answer: the only directly reusable internals are the observation encoder, the conditional U-Net, and the diffusion scheduler. That is enough to build a custom wrapper, but not enough to plug into SOPE as-is.
  - Short answer: the minimum viable adapter is `sample(obs)`, `sample_tensor(obs)`, and either `grad_log_prob(obs, act)` or `score(obs, act)`; adding `log_prob` / `elbo` would require extra derivation that robomimic does not currently provide.
3. Resolve the mismatched-input-signature question for $\beta$ and $\pi$.
  - Separate the problem into representation mismatch and policy-family mismatch.
  - Check whether SOPE only needs comparable action scores conditioned on each model's own observation encoding, or whether it implicitly assumes both policies consume identical state representations.
  - If identical inputs are required, define options: shared frozen encoder, offline feature translation layer, or re-encoding raw observations for both policies.
  - If only relative scores are needed, evaluate whether a common latent space can be avoided.
  - Short answer: policy-family mismatch is partly supported, but representation mismatch is not. The sampler passes the same `state_t_flat` tensor to both target and behavior policies during guidance.
  - Short answer: current SOPE implicitly assumes both policies can score actions from the same environment state representation and action space. There is no hook for separate visual encoders or feature translators inside the guidance code.
  - Short answer: if $\beta$ and $\pi$ use different embeddings, the practical fix is to re-express both on a shared raw observation input or add an external adapter that maps the shared state into each policy's expected input before calling its score function.
  - Short answer: a common latent space is avoidable only if each policy wrapper can accept the same raw observation and do its own internal encoding. Different precomputed feature signatures are not supported by the current sampler.
4. Evaluate ELBO as the fallback guidance signal.
  - Verify whether the robomimic diffusion implementation exposes enough terms to estimate an ELBO consistently at evaluation time.
  - Compare the practical tradeoff between exact likelihood, ELBO proxy, and denoising-score proxy for SOPE guidance.
  - Note failure modes, especially calibration issues when comparing scores across policies with different encoders or architectures.
  - Short answer: robomimic does not expose ELBO terms. Its training loop only computes noise-prediction MSE, so ELBO would have to be reconstructed manually from the scheduler and model outputs.
  - Short answer: for SOPE guidance, a denoising-score proxy is the most natural fit because the original Diffusion Policy paper frames the policy as learning the action-distribution score, and SOPE already supports a `grad_log_prob` path for diffusion policies.
  - Short answer: exact likelihood is the cleanest signal when available, ELBO is a weaker and harder-to-calibrate proxy, and raw score guidance is likely the easiest integration path. ELBO comparisons across policies with different encoders are especially likely to be miscalibrated.
5. Reconstruct how SOPE trains a reward predictor for sparse-reward tasks.
  - Review [reward.py](third_party/sope/opelab/core/reward.py) and identify the training inputs, targets, loss, and any dataset preprocessing assumptions.
  - Check where the reward model is instantiated and how it is consumed during OPE, including whether it predicts immediate reward, return-to-go, or terminal success.
  - Confirm whether sparse rewards are learned directly from transition labels or indirectly from trajectory-level signals.
  - Short answer: the reward model is trained in `core/data.py` on transition-level pairs `[s, a] -> r` using immediate rewards from the offline dataset. It uses an MLP with shape `[64, 64, 1]`, MSE loss, and `RewardEnsembleEstimator`, but currently with only one bootstrap seed.
  - Short answer: it does not use `next_state`, return-to-go, or trajectory-level labels. This matches the STITCH-OPE paper statement that the method trains a neural network on behavior transitions to predict immediate reward.
  - Short answer: `core/baselines/diffuser.py` uses the learned predictor only when an environment reward function is unavailable; otherwise it calls the environment-specific `reward_fn`.
  - Short answer: for sparse rewards, SOPE is learning directly from sparse per-transition labels. That is valid in principle, but if positives are extremely rare, the current regressor may need reweighting or additional task-specific handling.
6. Turn the investigation into implementation decisions.
  - Summarize whether diffusion-policy support should be added via direct likelihood estimation, ELBO scoring, or a custom adapter.
  - Summarize whether sparse-reward prediction can reuse SOPE's existing reward model or needs task-specific changes.
  - Convert the answers into concrete follow-up tasks in this file once the code reading is complete.
  - Short answer: for robomimic diffusion policies, the best path is a custom adapter that exposes `sample_tensor` and `grad_log_prob` / score guidance on top of robomimic's encoder + U-Net. Direct likelihood is not available today, and ELBO is possible only with substantial extra work.
  - Short answer: for the mismatched-input issue, support is feasible only if both policies can be wrapped around the same raw observation interface. If they depend on incompatible precomputed embeddings, guidance code needs a new adapter boundary before integration.
  - Short answer: SOPE's existing reward predictor is reusable for sparse rewards if the dataset already contains per-step reward labels and enough positive examples. Otherwise the first likely extension is not a new reward architecture, but better handling of class imbalance / reward sparsity.
  - Follow-up tasks:
    - Define a wrapper API for robomimic diffusion policies: `sample_tensor`, `grad_log_prob`, and explicit observation preprocessing from raw env observations.
    - Decide whether to standardize on raw observations or shared encoders before mixing $\beta$ and $\pi$ from different visual backbones.
    - Stress-test the current reward regressor on a truly sparse dataset before changing its architecture.
