# Sparse PointPillars

This is a Pytorch implementation of our [Sparse PointPillars](http://vedder.io/publications/sparse_point_pillars_snn_workshop.pdf) work presented at the Sparse Neural Networks 2021 Workshop.

This repo is built on top of [second.pytorch](https://github.com/traveller59/second.pytorch), the official codebase of the 3D detection model SECOND and the codebase used for the [official implementation of PointPillars](https://github.com/nutonomy/second.pytorch), which has been updated to provide a sanctioned implementation of PointPillars.

## Codebase Map

This codebases offers sparse backbones implemented using either [a fork of `spconv`](https://github.com/kylevedder/spconv) or [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) sparse operations. The implementation of the COO format Pillar Feature Net is found in [`second/pytorch/models/pointpillars.py`](https://github.com/kylevedder/SparsePointPillars/blob/master/second/pytorch/models/pointpillars.py), the `spconv` backbones are found in [`second/pytorch/models/rpn_spconv_sparse.py`](https://github.com/kylevedder/SparsePointPillars/blob/master/second/pytorch/models/rpn_spconv_sparse.py), and the Minkowski Engine backbones are found in [`second/pytorch/models/rpn_mink_sparse.py`](https://github.com/kylevedder/SparsePointPillars/blob/master/second/pytorch/models/rpn_mink_sparse.py).

## Development Environment Setup

 To reproduce our development environment, we provide both Docker files for CUDA 10 and 11 (inside [`docker/`](https://github.com/kylevedder/SparsePointPillars/tree/master/docker)) as well as a `conda` environment for CUDA 11 ([`environment.yml`](https://github.com/kylevedder/SparsePointPillars/blob/master/environment.yml)). When using a `conda` environment, the root of the repo must be added to your `PYTHONPATH`.