import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tqdm import trange
from multiprocessing import Pool


def diff_gradient(f, X, cr_gr = True):
    # VERY INEFFICIENT gradient IMPLEMENTATION
    ans = []
    for x in X:
        res = f(x)
        grad = torch.autograd.grad(res, x, create_graph = cr_gr)
        ans.append(grad[0])
    return torch.stack(ans)


def diff_laplacian(f, X, cr_gr = True):
    # VERY INEFFICIENT LAPLACIEN IMPLEMENTATION
    ans = []
    for x in X:
        hess = torch.autograd.functional.hessian(f, x, create_graph = cr_gr)
        ans.append(torch.trace(hess))
    return torch.stack(ans)

def LPDE(u, log_p, X, get_grad, get_lapl, cr_gr = True):
    # left part of differential equation
    # u: solution encoded by NN
    # log_p: function, calculates log density
    # X: tensor, [n_samples, n_dim]

    u_grad = torch.squeeze(get_grad(u, X, cr_gr))


    u_laplacian = get_lapl(u, X, cr_gr)
    log_p_grad = get_grad(log_p, X, cr_gr)

    return u_laplacian + (u_grad * log_p_grad).sum(dim = 1)

def RPDE(f, mean_est, X):
    # right pat of differential equation
    return f(X) - mean_est


class UniformAreaSampler:
    def __init__(self, N):
        # N: [2x2...x2] tensor of borders
        self.N = N
        self.dist = torch.distributions.uniform.Uniform(N[:, 0], N[:, 1])

    def sample(self, n):
        return self.dist.sample(n)

    def sample_border(self, n):
        samps = self.dist.sample(n)
        for i in range(samps.shape[0]):
            idx = torch.randint(high = self.N.shape[0], size = (1,))
            bit = torch.randint(high = 2, size = (1,))
            samps[i, idx] = self.N[idx, bit]
        return samps

class MeshGrid:
    def __init__(self, cube_coords, N):
        self.cube_coords = cube_coords
        self.N = N
        self.xs = []
        for i in range(cube_coords.shape[0]):
            self.xs.append(torch.linspace(cube_coords[i,0], cube_coords[i,1], N[i]))
        self.dist = torch.distributions.uniform.Uniform(cube_coords[:, 0], cube_coords[:, 1])

    def sample(self, n):
        n = n[0]
        coords = []
        for i in range(self.cube_coords.shape[0]):
            xinds = torch.randint(high = self.N[i], size = (n,))
            coords.append(self.xs[i][xinds])
        

        return torch.stack(coords).T

    def sample_border(self, n):
        samps = self.dist.sample(n)
        for i in range(samps.shape[0]):
            idx = torch.randint(high = self.cube_coords.shape[0], size = (1,))
            bit = torch.randint(high = 2, size = (1,))
            samps[i, idx] = self.cube_coords[idx, bit]
        return samps

class SquareBorder:
    def __init__(self, cube_coords, N):
        self.cube_coords = cube_coords
        self.N = N
        self.xs = torch.linspace(cube_coords[0,0], cube_coords[0,1], N[0])
        self.ys = torch.linspace(cube_coords[1,0], cube_coords[1,1], N[1])
        self.dist = torch.distributions.uniform.Uniform(cube_coords[:, 0], cube_coords[:, 1])

    def sample(self, n):
        samps = self.dist.sample(n)
        for i in range(samps.shape[0]):
            idx = torch.randint(high = self.cube_coords.shape[0], size = (1,))
            bit = torch.randint(high = 2, size = (1,))
            samps[i, idx] = self.cube_coords[idx, bit]
        return samps


def training(u, f, dist, area_sampler, border_sampler = None,
             lr = 1e-3, sample_sz = 10**3, epochs_amnt = 1, border_sz = 10**2,
             batch_sz = 10**2, it_num = 10**2, tol = 1e-4, verbose = 1,
             resample_its = True, str_name = "", dev = "cpu"):
    # f: function for which we train method
    # area_sampler: method of sampling points in the area. has methods: sample()
    # dist: distribution class. has methods: sample(), log_prob()

    optimizer = torch.optim.Adam(u.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    mean_est = f(dist.sample([sample_sz])).mean()

    loss = float('Inf')

    if border_sampler is not None:
        border_samp = border_sampler.sample([border_sz])
    batch = area_sampler.sample([batch_sz])


    for epochs in range(epochs_amnt):
        loss_history = []
        its = 0

        if not resample_its:
            batch = area_sampler.sample([batch_sz]).to(dev)
            batch.requires_grad=True

            if border_sampler is not None:
                border_samp = border_sampler.sample([border_sz])

        with trange(its, it_num) as progress_bar:
            for its in progress_bar:

                if resample_its:
                    batch = area_sampler.sample([batch_sz]).to(dev)
                    if border_sampler is not None:
                        border_samp = border_sampler.sample([border_sz])

                    batch.requires_grad=True

                lpart = LPDE(u, dist.log_prob, batch, diff_gradient, diff_laplacian)
                rpart = RPDE(f, mean_est, batch)
                loss = ((lpart - rpart)**2).mean()
                if border_sampler is not None:
                    border = (torch.squeeze(u(border_samp))**2).mean()
                    loss += border
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                loss_history.append(loss.item())

                if verbose == 2 and its % 10 == 0:

                    clear_output(wait=True)
                    plt.figure(figsize = (12, 9))
                    plt.plot(loss_history)
                    plt.grid()
                    plt.show()

                if verbose == 1 and its % 50 == 0:
                    print("Iteration {}, loss {}".format(its, loss))

        if verbose == 3:
            plt.figure(figsize = (12, 9))
            plt.plot(loss_history, linewidth = 2)
            plt.ylabel("loss")
            plt.grid()
            plt.savefig('training'+str_name+'{}.eps'.format(epochs), format='eps')
        scheduler.step()




