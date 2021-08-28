import os
import math
from importlib import reload
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display

import nnpde.iterative_methods as im
from nnpde.metrics import least_squares_loss as LSE
from nnpde import geometries, helpers
from nnpde.utils.logs import enable_logging, logging 
from nnpde.problems import DirichletProblem, after_solver
from nnpde.utils import plots
import nnpde.model as M 
import nnpde.model_testing as MT
import nnpde.problems as PDEF
from nnpde.grid_search import grid_search

from tqdm.notebook import tqdm

class Grid_Object:
    """Geometry class

    Parameters
    ----------
    n_dims : int, dimensionality of data
    
    cube_coords: borders of cube. given as [n_dims, 2] tensor    

    N: torch.tensor [n_dims]. number of grid points on each side
    
    h: float !!!. note that if N is given it has to make sure,
        that h in each dimension is the same

    Returns
    -------
    self : object
        Returns an instance of self.
    """

    def __init__(self,
                 n_dims,
                 cube_coords,
                 N = None, h = None):

        self.n_dims = n_dims
        self.cube_coords = cube_coords
        self.N = N
        self.h = h

        if h is None:
            self.h = (cube_coords[0, 1] - 
                                   cube_coords[0, 0])/(N[0]-1)

        if N is None:
            ans = list()
            for i in range(n_dims):
                ans.append(math.floor((cube_coords[i, 1] - cube_coords[i, 0])/(h)))
            self.N = torch.tensor(ans, dtype=torch.long) + 1
            self.h = (cube_coords[0, 1] - 
                                   cube_coords[0, 0])/(self.N[0]-1)
            # N might be different from given h, so it is easier to shif h for found N

        self.grad_log = None

    def get_logs(self, distribution):
        grd_size = torch.cat((torch.tensor([self.n_dims]), self.N))
        self.grad_log = torch.zeros(grd_size.tolist())

        ind = torch.zeros(self.n_dims, dtype=torch.int)
        over = False

        while not over:
            point = []
            for i in range(self.n_dims):
                x = self.cube_coords[i, 0] + self.h*ind[i]
                point.append(x)

            point = torch.tensor(point, requires_grad = True)
            val = distribution.log_prob(point)
            val.backward()

            for i in range(self.n_dims):
                self.grad_log[tuple([i, *ind.tolist()])] = point.grad[i]
            over = True

            for i in range(self.n_dims):
                ind[i] += 1
                if ind[i] == self.N[i]:
                    ind[i] = 0
                else:
                    over = False
                    break
        self.grad_log = self.grad_log.reshape([1, *grd_size.tolist()])


    def set_borders(self):
        B_idx = torch.zeros(self.N.tolist())
        B = torch.zeros(self.N.tolist())

        ind = torch.zeros(self.n_dims, dtype=torch.int)
        over = False

        while not over:
            if torch.any(ind == 0) or torch.any(ind == self.N-1):
                B[tuple(ind.tolist())] = torch.rand([1])
                B_idx[tuple(ind.tolist())] = 0
            else:
                B[tuple(ind.tolist())] = 0
                B_idx[tuple(ind.tolist())] = 1

            over = True

            for i in range(self.n_dims):
                ind[i] += 1
                if ind[i] == self.N[i]:
                    ind[i] = 0
                else:
                    over = False
                    break
        return B_idx.reshape([1, 1, *self.N.tolist()]), B.reshape([1, 1, *self.N.tolist()])

    def create_f_grid(self, function):
        ans = torch.zeros(self.N.tolist())

        ind = torch.zeros(self.n_dims, dtype=torch.int)
        over = False

        while not over:

            point = []
            for i in range(self.n_dims):
                x = self.cube_coords[i, 0] + self.h*ind[i]
                point.append(x)
            
            point = torch.tensor(point).reshape(1, -1)

            ans[tuple(ind.tolist())] = function(point)
            over = True

            for i in range(self.n_dims):
                ind[i] += 1
                if ind[i] == self.N[i]:
                    ind[i] = 0
                else:
                    over = False
                    break
        return ans.reshape([1, 1, *self.N.tolist()])



class Solve_Area:
    """Implemented solver for stein cv

    Parameters
    ----------
    n_dims : int, dimensionality of data
    
    cube_coords: borders of cube. given as [n_dims, 2] tensor

    N: torch.tensor [n_dims]. number of grid points on each side
    
    h: torch.tensor [n_dims], distance between adjacent nodes on each coordinate
    

    Returns
    -------
    self : object
        Returns an instance of self.
    """

    def __init__(self,
                 n_dims,
                 cube_coords,
                 distribution,
                 N = None, h = None):


        if N is None and h is None:
            raise ValueError('Specify N or h')

        self.my_grid = Grid_Object(n_dims, cube_coords, N, h)

        self.my_grid.get_logs(distribution) # calculates gradient in all nodes of grid and saves them as GridObject attribute

        self.my_solver = None

        self.aslv = None

    def train_model(self, base_parameters, prob_inst = 20, f = None):
        # For each problem instance define number of iteration to perform to obtain the solution
        problem_instances = []
        for i in tqdm(range(prob_inst)):
            k = np.random.randint(1, 20)
            problem_instances.append(DirichletProblem(k=k, grid = self.my_grid, f = f))

        self.my_solver = M.JacobyWithConv(**base_parameters)
        self.my_solver.fit(problem_instances)
        self.my_solver.net.eval()

    def solve_setting(self, function, samples, k = 1000):
        mean_val = torch.mean(function(samples))
        f_table = self.my_grid.create_f_grid(function) - mean_val
        self.aslv = after_solver(f = f_table, grid = self.my_grid, 
                            k = k, model = self.my_solver)

    def get_cv(self, points):
        #points : [n_points, n_dims]
        # note: this works for currently set aslv
        # current heuristic: cv of point equals to cv of closest node on grid
        ans = []
        for point in points:
            closest_on_grid = []
            n = len(self.my_grid.N.tolist())
            for i in range(n):
                coords = torch.linspace(self.my_grid.cube_coords[i, 0],
                                      self.my_grid.cube_coords[i, 1],
                                      self.my_grid.N[i])
                j = torch.argmin(torch.abs(coords - point[i]))
                closest_on_grid.append(j)
            ans.append(self.aslv.sec_der[tuple([0, 0, *closest_on_grid])])
        return torch.tensor(ans)



