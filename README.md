# NN_DE_solver

This repository contains programmes for project on finding Control Variates via solving differential equation. 
Two methods were used to solve equation: finite difference method and [Deep Galerking Method](https://arxiv.org/abs/1708.07469) 
(https://arxiv.org/abs/1708.07469).

## DGM

No parallelisations implemented, thus takes a lot of time to train. Sampling strategy can be changed via substituting area_sampler

### Files
- DGM_net.py: defines neural network
- details.py: defines functions for area samplers, differential equation and training procedure
- DGM_ndim.ipynb: performs experiment, .py file is preferred to this one
- DGM_ndim.py: .py version of DGM_ndim.ipynb

## Poisson equation solver

Finite-difference algorithm based on paper [Learning neural PDE solvers with convergence guarantees](https://openreview.net/pdf?id=rklaWn0qK7) 
(https://openreview.net/pdf?id=rklaWn0qK7)

# DISCLAIMER

Finite difference algorithm is based on project in repo 
https://github.com/francescobardi/pde_solver_deep_learned

- nnpde: files inside it are discribed in file description.md
- my_solver 2dim.ipynb: performs experiment in 2 dimensions
- my_solver 3dim.ipynb: performs experiment in 3 dimensions

### Learning Process

1. Random Dirichlet problems are created and solved with
..., which gaurantees availability of correct answer

2. These solved problems are passed to __JacobyWithConv__ solver, which
trains new ConvNet, which is H.
