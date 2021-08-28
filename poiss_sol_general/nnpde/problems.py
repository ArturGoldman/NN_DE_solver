import numpy as np
import torch
import torch.nn as nn

from nnpde import geometries
from nnpde.utils import misc
import nnpde.iterative_methods as im


class DirichletProblem:
    """Define a Dirichlet problem instance

    The groud truth solution is computed on instantiation.

    Parameters
    ----------
    B_idx : tensor-like, shape = [1, 1, [N]]
        variable to reset.

    B : tensor-like, shape = [1, 1, [N]], optional

    f : tensor-like, shape = [1, 1, [N]], optional
        variable to reset.

    k : int, optional, default 20
        Number of iterations to use for oTODO.

    k_ground_truth : int, optional, default 1000
        Number of iterations to used to obtain ground truth solution
        with Jacobi method.

    inital_u : tensor-like, shape = [1, 1, [N]], optional
               Default = torch.rand(1, 1, [N], requires_grad=True)
               Initial solution

    inital_u_jacobi : tensor-like, shape = [1, 1, [N]], optional
               Default = torch.rand(1, 1, [N], requires_grad=True)

    grid: object of class Grid_Object.


    Returns
    -------
    self : object
        Returns an instance of self.
    """

    def __init__(self,
                 B_idx=None,
                 B=None,
                 f=None,
                 k=20,
                 k_ground_truth=1000,
                 initial_ground_truth=None,
                 initial_u=None,
                 grid = None):

        # Initialize Geometry and Boundary Conditions
        if B_idx is None:
            self.B_idx, self.B = grid.set_borders()

        # Initialize f
        if f is None:
            self.f = torch.zeros([1, 1, *grid.N.tolist()])
        else:
            self.f = f

        # Initialize parameters to compute ground truth solution
        if initial_ground_truth is None:
            self.initial_ground_truth = misc.normal_distributed_tensor(grid.N, requires_grad=False)
        else:
            self.initial_ground_truth = initial_ground_truth

        self.k_ground_truth = k_ground_truth
        self._ground_truth = None

        # Initialize parameters to obtain u
        if initial_u is None:
            self.initial_u = misc.normal_distributed_tensor(grid.N, requires_grad=True)
        else:
            self.initial_u = initial_u

        self.k = k
        
        self.grid = grid

        # Compute ground truth solution using Jacobi method
        self.ground_truth = im.jacobi_method(
            self.B_idx, self.B, self.f, self.grid, self.initial_ground_truth, self.k_ground_truth)

    def compute_solution(self, net):
        """Compute solution using optim method
        """
        self.u = im.H_method(net, self.B_idx, self.B,
                             self.f, self.grid, self.initial_u, self.k)
        return self.u
    
class after_solver:
    def __init__(self,
                 B_idx=None,
                 B=None,
                 f=None,
                 k=1000,
                 initial_u=None,
                 grid = None,
                 model = None, u_H = None):

        # Initialize Geometry and Boundary Conditions
        if B_idx is None:
            self.B_idx, self.B = grid.set_borders()

        self.f = f

        # Initialize parameters to obtain u
        if initial_u is None:
            self.initial_u = misc.normal_distributed_tensor(grid.N, requires_grad=True)
        else:
            self.initial_u = initial_u

        self.k = k

        self.grid = grid

        if model is None:
            self.u_H = u_H
        else:
            # Compute ground truth solution using H method
            self.u_H = im.H_method(model.net, self.B_idx, self.B, self.f, 
                                   self.grid,
                                   initial_u=self.initial_u, k=self.k)
        self.sec_der = self.calc_sec()
        
        

    def n_dim_conv(self, n):
        if n == 1:
            return nn.Conv1d(1, 1, 3, padding = 1, bias = False)
        elif n == 2:
            return nn.Conv2d(1, 1, 3, padding = 1, bias = False)
        elif n == 3:
            return nn.Conv3d(1, 1, 3, padding = 1, bias = False)
        else:
            raise ValueError('only n < 4 supported')


    def calc_sec(self):
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
        u : tensor-like, shape = [*, *, n, n]
            resetted values.
        """

        net = None

        l = len(self.grid.N.tolist())
        grad_nns = []

        net = self.n_dim_conv(l)
        for i in range(l):
            grad_nns.append(self.n_dim_conv(l))

        wsz = [3 for _ in range(l)]
        initial_weights = torch.zeros([1,1,*wsz])
        cent_coord = [1 for _ in range(l)]
        coef = 2**l
        for i in range(l):
            cent_coord[i] += 1
            initial_weights[tuple([0, 0, *cent_coord])] = 1
            cent_coord[i] -= 2
            initial_weights[tuple([0, 0, *cent_coord])] = 1
            cent_coord[i] += 1

            coord_grad_w = torch.zeros([1,1,*wsz])
            cent_coord[i] += 1
            coord_grad_w[tuple([0, 0, *cent_coord])] = 1
            cent_coord[i] -= 2
            coord_grad_w[tuple([0, 0, *cent_coord])] = -1
            cent_coord[i] += 1
            grad_nns[i].weight = nn.Parameter(coord_grad_w)

        initial_weights[tuple([0, 0, *cent_coord])] = -coef
        net.weight = nn.Parameter(initial_weights)
        # The final model will be defined as a convolutional network, but this step
        # is fixed.
        for param in net.parameters():
            param.requires_grad = False

        for i in range(l):
            for param in grad_nns[i].parameters():
                param.requires_grad = False

        ans = net(self.u_H)/self.grid.h**2
        for i in range(l):
            ans += grad_nns[i](self.u_H)*self.grid.grad_log[0, i].reshape([1, 1, *self.grid.N.tolist()])/(2*self.grid.h)

        return ans


