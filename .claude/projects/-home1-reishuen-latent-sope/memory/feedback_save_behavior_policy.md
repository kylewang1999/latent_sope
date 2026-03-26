---
name: save_behavior_policy
description: User wants behavior policy checkpoints saved to disk after training
type: feedback
---

Save the behavior policy to disk after training so it can be reused across experiments.

**Why:** Training the behavior policy takes time (~1000 epochs) and the user doesn't want to retrain it every run.

**How to apply:** After training the DiffusionBehaviorPolicy in experiment notebooks, add a cell to save the model state_dict (e.g., `torch.save(behavior_policy.model.state_dict(), save_path)`). Also add a load path option to skip training if checkpoint exists.
