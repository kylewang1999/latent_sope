# Target Policy Collection

Relevant references:

- [data/policy/rmimic-lift-ph-lowdim_diffusion_260130](../data/policy/rmimic-lift-ph-lowdim_diffusion_260130)

## 1. Goal

Please collect a set of robomimic `ph` low-dimensional-state diffusion target
policies for Lift.

Each collected target policy should be intentionally weaker than the reference
behavior policy at
[data/policy/rmimic-lift-ph-lowdim_diffusion_260130](../data/policy/rmimic-lift-ph-lowdim_diffusion_260130).
The directory name should explain why the target policy is weaker, for example:

- fewer training trajectories
- fewer training epochs
- another clearly stated reduction in training quality

Examples of acceptable suffixes:

- `epoch10-numtraj20`
- `epoch25-numtraj50`

## 2. Deliverable Layout

The handoff should contain a top-level `policy/` directory.

At depth 1, the submitted directory should look like:

```text
policy/
  rmimic-lift-ph-lowdim_diffusion_epoch10-numtraj20/
  rmimic-lift-ph-lowdim_diffusion_epoch25-numtraj50/
  ...
```

Inside each policy directory, use the simplified handoff format below instead
of submitting the full robomimic training run directory.

Inside `policy/`, place one subdirectory per collected target policy, using the
format:

`rmimic-lift-ph-lowdim_diffusion_<additional-description>`

Here, `<additional-description>` should describe how that target policy was
made inferior to the reference behavior policy.

## 3. Required Contents

Within each
`rmimic-lift-ph-lowdim_diffusion_<additional-description>/` directory, include:

- a single checkpoint file named `model.pth`
- a `videos/` subdirectory containing exactly 5 example rendered rollout
  `.mp4` files

Do not add extra checkpoint nesting such as `models/` inside the submitted
policy directory unless requested later.

## 4. Example Directory Tree

```text
policy/
  rmimic-lift-ph-lowdim_diffusion_epoch10-numtraj20/
    model.pth
    videos/
      rollout_01.mp4
      rollout_02.mp4
      rollout_03.mp4
      rollout_04.mp4
      rollout_05.mp4
  rmimic-lift-ph-lowdim_diffusion_epoch25-numtraj50/
    model.pth
    videos/
      rollout_01.mp4
      rollout_02.mp4
      rollout_03.mp4
      rollout_04.mp4
      rollout_05.mp4
```

## 5. Naming Intent

The purpose of the suffix is to make the degradation mechanism visible from the
directory name alone. Someone reading the directory should be able to tell why
that target policy is expected to perform worse than the reference behavior
policy.
