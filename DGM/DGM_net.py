import torch
from tqdm.notebook import tqdm

def identity_func(x):
    return x

def const_tanh(x, c = 50.):
    return c*torch.tanh(x)

class DGM_Layer(torch.nn.Module):
    def __init__(self, dim, hidden_size, activation = torch.tanh):
        super(DGM_Layer, self).__init__()
        self.Uz = torch.nn.Linear(dim, hidden_size)
        self.Ug = torch.nn.Linear(dim, hidden_size)
        self.Ur = torch.nn.Linear(dim, hidden_size)
        self.Uh = torch.nn.Linear(dim, hidden_size)

        self.Wz = torch.nn.Linear(hidden_size, hidden_size)
        self.Wg = torch.nn.Linear(hidden_size, hidden_size)
        self.Wr = torch.nn.Linear(hidden_size, hidden_size)
        self.Wh = torch.nn.Linear(hidden_size, hidden_size)

        self.activation = activation

    def forward(self, x, s):
        Z = self.activation(self.Uz(x)+self.Wz(s))
        G = self.activation(self.Ug(x)+self.Wg(s))
        R = self.activation(self.Ur(x)+self.Wr(s))
        H = self.activation(self.Uh(x)+self.Wh(s*R))

        return (1-G)*H+Z*s


class DGM_Net(torch.nn.Module):
    def __init__(self, dim, hidden_size, L,
                 activation = torch.tanh):
        super(DGM_Net, self).__init__()
        self.L = L
        self.starter = torch.nn.Linear(dim, hidden_size)
        self.DGMS = torch.nn.ModuleList()
        for _ in range(L):
            self.DGMS.append(DGM_Layer(dim, hidden_size, activation))
        #self.unwinder = torch.nn.Linear(dim+hidden_size, hidden_size)
        self.finisher = torch.nn.Linear(hidden_size, 1)

        self.activation = activation

    def forward(self, x):
        s = self.starter(x)
        for i in range(self.L):
            s = self.DGMS[i](x, s)
            #s = self.activation(s)
        #s = self.activation(self.unwinder(torch.cat((x, s), dim = 1) ))
        res = self.finisher(s)
        return res

