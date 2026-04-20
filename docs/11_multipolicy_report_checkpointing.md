# Multipolicy Report Checkpointing

Relevant code:

- [src/eval.py](../src/eval.py)
- [scripts/test_ope_guided_multipolicy.py](../scripts/test_ope_guided_multipolicy.py)

## 1. Summary

`evaluate_guided_multipolicy_ope(...)` now accepts the output JSON path and
rewrites that file immediately after each target-policy evaluation finishes.

This changes report persistence only. It does not change the guided rollout,
true rollout, reward-prediction, or aggregate metric computations.

## 2. Implementation

The evaluator now:

1. appends one `MultipolicyOPEPolicyReport` after each completed target-policy
   pass
2. rebuilds the top-level `MultipolicyOPEReport` from the currently finished
   prefix of `policies`
3. writes the JSON through a same-directory temporary file and atomically
   replaces the previous report

That means a later rollout failure or manual interruption still leaves a valid
JSON containing every policy that finished before the failure.

## 3. Validation

Validation run:

```bash
source /home/kyle/miniforge3/etc/profile.d/conda.sh && conda activate latent_sope && \
python -m py_compile src/eval.py scripts/test_ope_guided_multipolicy.py
```
