# AdSNN: Sparse Computation in Adaptive Spiking Neural Networks

MATLAB code for the paper:

> D. Zambrano, R. Nusselder, H. S. Scholte, S. M. Bohté.
> **Sparse Computation in Adaptive Spiking Neural Networks.**
> *Frontiers in Neuroscience* 12:987 (2019).
> [doi:10.3389/fnins.2018.00987](https://doi.org/10.3389/fnins.2018.00987)

Deep networks are trained with the effective transfer function of an
Adaptive Spiking Neuron (ASN). The analog units are then replaced one-to-one
with adaptive spiking neurons, giving Adaptive SNNs (AdSNNs) that classify
competitively at low average firing rates. An "arousal" mechanism raises
coding precision only when needed and lowers the firing rate further.

The work was done in the Machine Learning group at
[CWI](https://www.cwi.nl), Amsterdam.

## Repository layout

The code is built on a modified copy of [MatConvNet](https://www.vlfeat.org/matconvnet/)
1.0-beta22, in `matconvnet-1.0-beta22/`. The AdSNN-specific parts are:

| Path (under `matconvnet-1.0-beta22/`) | Content |
|---|---|
| `matlab/vl_nnasn.m`, `matlab/vl_nnsumtanh*.m` | ASN transfer function as a MatConvNet layer (analog training/inference) |
| `matlab/snn_*.m` | Adaptive spiking neuron layers (CPU and GPU variants) |
| `matlab/simplenn/snn_simplenn*.m` | Time-stepped simulation of a converted spiking network |
| `matlab/simplenn/vl_simplenn*.m` | MatConvNet core, extended with the `asn`, `sumtanh`, `tanh` and `conv_shortcut` layer types |
| `examples/{mnist,cifar,cifar100,imagenet,iris,sonar}/` | Network definitions, Keras weight import (`readweights*.m`) and experiments (`snn_*.m`, `*_SUMSPIKES_*.m`, `*arousal*`) |
| `examples/components/` | Experiments on single network components |
| `figures/` | Scripts used to produce the paper figures |
| `asnn_root.m`, `setup_asnn.m` | Path helpers (see below) |

All other files are stock MatConvNet.

## Getting started

Requirements: MATLAB, a C++ compiler and,
for the ImageNet experiments, a CUDA GPU.

```matlab
cd cwi-asnn/matconvnet-1.0-beta22
vl_compilenn                         % add 'enableGpu', true for GPU support
setup_asnn                           % adds everything to the path
```

The experiment scripts expect to run from `matconvnet-1.0-beta22/` and use
`asnn_root` to find their files. Datasets and trained models go in
`matconvnet-1.0-beta22/data/`, which is git-ignored.

The networks in the paper were trained in Keras with the ASN transfer function,
and their weights were imported from `.h5` files with the `readweights*.m`
scripts. The Keras training code and the trained weights are **not** part of
this repository.

## Status

This is research code, published as it was used for the paper (2017–2019) and
cleaned up only to remove machine-specific paths. Expect experiment scripts
with hard-coded parameters rather than a polished library. Note that
`examples/cnn_train.m` enables GPU 1 by default (`opts.gpus = 1`).

## Citation

```bibtex
@article{zambrano2019sparse,
  title   = {Sparse Computation in Adaptive Spiking Neural Networks},
  author  = {Zambrano, Davide and Nusselder, Roeland and Scholte, H. Steven and Boht{\'e}, Sander M.},
  journal = {Frontiers in Neuroscience},
  volume  = {12},
  pages   = {987},
  year    = {2019},
  doi     = {10.3389/fnins.2018.00987}
}
```

## License

MIT for the AdSNN code (see `LICENSE`). The included MatConvNet code keeps its
own license (`matconvnet-1.0-beta22/COPYING`).
