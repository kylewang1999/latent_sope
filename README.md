## 0. Keeping submodules in sync

This repo uses git submodules to manage dependencies. When you fetch/pull updates, make sure submodules are updated too.

**Case 1** - Fresh clone:
```
git clone --recurse-submodules https://github.com/kylewang1999/latent_sope.git
```

**Case 2** - You've cloned the repo and want to update the submodules:

```bash
git fetch --recurse-submodules # Fetch the latest changes from the remote repositories (including changes to the submodules), but don't apply them yet. OR:
git pull --recurse-submodules # Pull the latest changes from the remote repositories (including changes to the submodules), and apply them to the local repo. OR:
git submodule update --init --recursive # Update the submodules to the latest commit, and apply them to the local repo.
```

## 1. Setup Instructions

### 1.1 Setup to Develop on [CARC cluster](https://www.carc.usc.edu/user-guides/quick-start-guides/intro-to-carc)

Follow the instructions in [carc_usage_advanced](https://github.com/kylewang1999/carc_usage/blob/main/carc_tutorial_advanced.md) to setup your development tools to code on CARC. 


### 1.2 Set up the development environment

1\. Remove the existing environment named `latent_sope` (if it exists)
```bash
conda deactivate && conda remove -n latent_sope --all -y
```


2\. Create a new environment named `latent_sope` with Python 3.10 and then activate it
```bash
conda create -n latent_sope python=3.10 -y
conda activate latent_sope
```

> [!CAUTION]
> This conda environment installs CUDA 12-based wheels. It may not work on some GPUs (e.g., L40S) if the node's NVIDIA driver or CUDA compatibility is not aligned with CUDA 12. The easiest way to fix this by relinquishing the current compute node and requesting a new one with a different GPU (v100/a100/a40).

3\. Run [bootstrap_env.sh](bootstrap_env.sh) first to install packages into the new env.

```bash
bash bootstrap_env.sh
```

4\. Then run [bootstrap_egl.sh](bootstrap_egl.sh) while that env is active to set up the EGL environment variables (for headless rendering on servers).
```bash
bash bootstrap_egl.sh
```

5\. Re-activate the env apply the EGL variables in your shell:
```bash
conda deactivate && conda activate latent_sope
```

### Test the environment

1\. Click through [scripts/hello_robomimic.ipynb](scripts/hello_robomimic.ipynb) to test the environment.
