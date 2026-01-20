## 1. Setup Instructions



### 1.1 Setup to Develop on [CARC cluster](https://www.carc.usc.edu/user-guides/quick-start-guides/intro-to-carc)


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