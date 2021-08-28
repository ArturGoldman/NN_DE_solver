import logging
import copy
from functools import reduce

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from nnpde import helpers, metrics
import nnpde.iterative_methods as im
from nnpde.utils.misc import chunks, set_seed


def n_dim_conv(n):
    if n == 1:
        return nn.Conv1d(1, 1, 3, padding = 1, bias = False)
    elif n == 2:
        return nn.Conv2d(1, 1, 3, padding = 1, bias = False)
    elif n == 3:
        return nn.Conv3d(1, 1, 3, padding = 1, bias = False)
    else:
        raise ValueError('only n < 4 supported')


class _ConvNet_(nn.Module):
    def __init__(self, n_dims, nb_layers):
        super(_ConvNet_, self).__init__()

        self.convLayers = nn.ModuleList([n_dim_conv(n_dims)
                                         for _ in range(nb_layers)])

    def forward(self, x, boundary):
        return reduce(lambda acc, el: el(acc) * boundary, self.convLayers, x)


class JacobyWithConv:
    """A class to obtain the optimal weights"""

    def __init__(self,
                 net=None,
                 batch_size=1,
                 learning_rate=1e-6,
                 max_epochs=1000,
                 nb_layers=3,
                 tol=1e-6,
                 stable_count=50,
                 n_dims = None,
                 optimizer='SGD',
                 check_spectral_radius=False,
                 random_seed=None):

        if random_seed is not None:
            set_seed(random_seed)

        if net is None:
            self.nb_layers = nb_layers
            self.net = _ConvNet_(n_dims = n_dims, nb_layers=self.nb_layers)
        else:
            self.net = net

        self.learning_rate = learning_rate
        if optimizer == 'Adadelta':
            logging.info(f"Using optimizer {optimizer}")
            self.optim = torch.optim.Adadelta(self.net.parameters())
        else:
            self.optim = torch.optim.SGD(
                self.net.parameters(), lr=learning_rate)

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tol = tol
        self.stable_count = stable_count


    def _optimization_step_(self, problem_instances):

        shuffled_problem_instances = np.random.permutation(problem_instances)

        for problem_chunk in chunks(shuffled_problem_instances, self.batch_size):
            self.net.zero_grad()

            # Compute loss using only batch
            loss = metrics.compute_loss(self.net, problem_chunk)

            # Backpropagate loss function
            loss.backward(retain_graph=True)

            # Update weights
            self.optim.step()

    def fit(self, problem_instances):
        """
             Returns
             -------
             self : object
                 Returns the instance (self).
        """
        # Initialization
        losses = []
        prev_total_loss = metrics.compute_loss(
            self.net, problem_instances).item()
        convergence_counter = 0
        logging.info(
            f"Training with max_epochs: {self.max_epochs}, tol: {self.tol}. Initial loss is {prev_total_loss}")

        # Optimization loop
        for n_epoch in range(self.max_epochs):

            # Update weights
            self._optimization_step_(problem_instances)

            # Compute total loss
            total_loss = metrics.compute_loss(
                self.net, problem_instances).item()

            # Store lossses for visualization
            losses.append(total_loss)

            # Check convergence
            if np.abs(total_loss - prev_total_loss) < self.tol:
                convergence_counter += 1
                # print(convergence_counter)
                # print(self.stable_count)
                if convergence_counter > self.stable_count:
                    logging.info(
                        f"Convergence reached")
                    break
            else:
                convergence_counter = 0

            prev_total_loss = total_loss

            # Display information every 100 iterations
            if n_epoch % 100 == 0:
                logging.info(
                    f"Epoch {n_epoch} with total loss {prev_total_loss}")

        #self.H = helpers.conv_net_to_matrix(self.net, self.N)
        self.losses = losses
        logging.info(
            f"{n_epoch} epochs with total loss {total_loss}")

        return self
