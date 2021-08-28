# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnpde.helpers import check_dimensions
from nnpde.utils.misc import apply_n_times

def _reset_boundary_(u, boundary_index, boundary_values):
    """ Reset values at the boundary of the domain


    Parameters
    ----------
    u : tensor-like, shape = [*, *, n, n]
        variable to reset.

    boundary_index : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    boundary_values : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.

    Returns
    -------
    u : tensor-like, shape = [*, *, n, n]
        resetted values.


    """
    

    return u * boundary_index + boundary_values

def n_dim_conv(n):
    if n == 1:
        return nn.Conv1d(1, 1, 3, padding = 1, bias = False)
    elif n == 2:
        return nn.Conv2d(1, 1, 3, padding = 1, bias = False)
    elif n == 3:
        return nn.Conv3d(1, 1, 3, padding = 1, bias = False)
    else:
        raise ValueError('only n < 4 supported')

def _jacobi_iteration_step_(u, boundary_index, boundary_values, forcing_term, grid):
    """ Jacobi method iteration step, defined as a convolution.
    Resets the boundary.


    Parameters
    ----------
    u : tensor-like, shape = [*, *, [N]]
        variable to reset.

    boundary_index : tensor-like, shape = [*, *, [N]]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    boundary_values : tensor-like, shape = [*, *, [N]]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.

    forcing_term : tensor-like, shape = [*, *, [N]]
        matrix describing the forcing term.


    Returns
    -------
    u : tensor-like, shape = [*, *, [N]]
        resetted values.
    """

    net = None

    l = len(grid.N.tolist())
    grad_nns = []

    net = n_dim_conv(l)
    for i in range(l):
        grad_nns.append(n_dim_conv(l))

    wsz = [3 for _ in range(l)] #kernel is of size 3x3x...
    initial_weights = torch.zeros([1,1,*wsz])
    cent_coord = [1 for _ in range(l)]
    coef = 1/(2**l)
    for i in range(l):
        cent_coord[i] += 1
        initial_weights[tuple([0, 0, *cent_coord])] = coef
        cent_coord[i] -= 2
        initial_weights[tuple([0, 0, *cent_coord])] = coef
        cent_coord[i] += 1

        coord_grad_w = torch.zeros([1,1,*wsz])
        cent_coord[i] += 1
        coord_grad_w[tuple([0, 0, *cent_coord])] = grid.h*coef/2
        cent_coord[i] -= 2
        coord_grad_w[tuple([0, 0, *cent_coord])] = -grid.h*coef/2
        cent_coord[i] += 1
        grad_nns[i].weight = nn.Parameter(coord_grad_w)

    net.weight = nn.Parameter(initial_weights)
    # The final model will be defined as a convolutional network, but this step
    # is fixed.
    for param in net.parameters():
        param.requires_grad = False

    for i in range(l):
        for param in grad_nns[i].parameters():
            param.requires_grad = False
        
    ans = net(u)
    for i in range(l):
        ans += grad_nns[i](u)*grid.grad_log[0, i].reshape([1, 1, *grid.N.tolist()])
    ans -=  grid.h**2*coef * forcing_term

    return _reset_boundary_(ans, boundary_index, boundary_values)


def jacobi_method(boundary_index, boundary_values, forcing_term, grid, initial_u = None, k = 1000):
    """ Compute jacobi method solution by convolution


    Parameters
    ----------
    boundary_index : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    boundary_values : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.

    forcing_term : tensor-like, shape = [*, *, n, n]
        matrix describing the forcing term.

    initial_u : tensor-like, shape = [*, *, n, n]
        Initial values.

    Returns
    -------
    u : tensor-like, shape = [*, *, n, n]
        solution matrix.
    """
    if initial_u is None:
        u = torch.zeros(grid.N.tolist())
    else:
        u = initial_u

    u = _reset_boundary_(u, boundary_index, boundary_values)

    def step(u_k):
        return _jacobi_iteration_step_(u_k, boundary_index, boundary_values, forcing_term, grid)

    return apply_n_times(step, k)(u)


# TODO rename this
def H_method(net, boundary_index, boundary_values, forcing_term, grid, initial_u=None, k=1000):
    """ Compute solution by H method

    Parameters
    ----------
    net = neural network representing H

    boundary_index : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: 1.0 for inner points 0.0 elsewhere.

    boundary_values : tensor-like, shape = [*, *, n, n]
        matrix describing the domain: desired values for boundary points 0.0 elsewhere.

    forcing_term : tensor-like, shape = [*, *, n, n]
        matrix describing the forcing term.

    initial_u : tensor-like, shape = [*, *, n, n]
        Initial values.

    Returns
    -------
    u : tensor-like, shape = [*, *, n, n]
        solution matrix.
    """

    u = _reset_boundary_(initial_u, boundary_index, boundary_values)

    def step(u_n):
        jac_it = _jacobi_iteration_step_(u_n, boundary_index, boundary_values, forcing_term, grid)
        u_n = jac_it + net(jac_it - u_n, boundary_index)
        return _reset_boundary_(u_n, boundary_index, boundary_values)

    return apply_n_times(step, k)(u)
