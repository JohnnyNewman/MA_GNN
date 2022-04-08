


import torch
from torch import nn
import torch.nn.functional as F

from pykeops.torch import LazyTensor



def laplacian_kernel(x, y, sigma=0.1):
    x_i = LazyTensor(x[:, None, :])  # (M, 1, 1)
    y_j = LazyTensor(y[None, :, :])  # (1, N, 1)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
    # return (-D_ij.sqrt() / LazyTensor(sigma[:, None], axis=1)).exp()  # (M, N) symbolic Laplacian kernel matrix
    return (-D_ij.sqrt() / sigma).exp()  # (M, N) symbolic Laplacian kernel matrix


class NNModel2(nn.Module):
    def __init__(self, n_lvls, n_nodes, n_hidden, n_ouput, trial_points, trial_points_lvl):
        super(NNModel2, self).__init__()

        self.nodeEmb = nn.Embedding(n_lvls*n_nodes, n_hidden) # lvl*nodes*embdim

        self._n_lvls = n_lvls
        self._n_nodes = n_nodes
        self._n_hidden = n_hidden
        self._n_ouput = n_ouput

        c1, c2, c3, c4 = 40, 40, 40, 40

        self.lin1 = nn.Linear(n_lvls*n_hidden, c1)
        self.lin2 = nn.Linear(c1, c2)
        self.lin3 = nn.Linear(c2, c3)
        self.lin4 = nn.Linear(c3, c4)
        self.lin5 = nn.Linear(c4, n_ouput)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()
        

        #self.dropout = nn.AlphaDropout(p=0.05)
        self.dropout_hidden = nn.Dropout(p=0.03)
        self.dropout = nn.Dropout(p=0.01)

        # self.theta1 = nn.parameter.Parameter([1.])
        # self.theta2 = nn.parameter.Parameter([1.])

        self._set_trial_pts(trial_points, trial_points_lvl)
        #self.all_levels = torch.unique(trial_points_lvl).cuda()
        self.alpha = nn.parameter.Parameter(0.001*torch.ones((n_lvls))).requires_grad_()
        self.sigma = nn.parameter.Parameter(0.1*torch.ones((n_lvls))).requires_grad_()
        #self.sigma = nn.parameter.Parameter(0.1*torch.ones((n_lvls, n_nodes))).requires_grad_()
    
    def _set_trial_pts(self, x, lvls):

        # x = torch.tensor(x).cuda()

        self._trial_points = x
        self._trial_points_lvl = lvls
        #self._trial_points_lvl = torch.tensor(lvls, dtype=int).cuda()
        # self._K_xx = laplacian_kernel(x, x)
        #a = K_xx.solve(qoi_train, alpha=alpha)

    def _interpolate(self, lvl, x_out, alpha=1.):

        inds = torch.where(self._trial_points_lvl == lvl)[0].tolist()
        n_inds = len(inds)
        #inds = inds[torch.randperm(n_nodes)[:n_nodes // 2**lvl].tolist()]
        # inds = [inds[i] for i in torch.randperm(n_inds)[:n_inds // 2**lvl]]

        x = self._trial_points[inds]
        W = self.nodeEmb.weight[inds]

        # print(x.shape, W.shape)

        # K_xx = self._K_xx[inds,inds].copy()

        K_xx = laplacian_kernel(x, x, self.sigma[lvl-1])
        # K_xx = laplacian_kernel(x, x, self.sigma[lvl-1][inds])
        # print(K_xx.shape, self.sigma[lvl-1][inds].shape)

        a = K_xx.solve(W, alpha=self.alpha[lvl-1])

        K_tx = laplacian_kernel(x_out, x, self.sigma[lvl-1])
        # K_tx = laplacian_kernel(x_out, x, self.sigma[lvl-1][inds])
        emb_interp = K_tx @ a

        # del K_xx
        # del K_tx
        # del a
        
        return emb_interp

    def forward(self, x):

        h_list = []
        h = torch.hstack([self._interpolate(lvl, x) for lvl in [1,2,3,4,5,6,7]])
        #h = self._interpolate(1, x)

        # print(h.shape)

        z1 = self.dropout(self.act1(self.lin1(self.dropout_hidden(h))))
        z2 = self.dropout(self.act2(self.lin2(z1)))
        z3 = self.dropout(self.act3(self.lin3(z2)))
        z4 = self.dropout(self.act4(self.lin4(z3)))

        y = self.lin5(z4)

        return y

    # def res(self, x, t):

    #     u = self.forward(x, t)

    #     ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    #     uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
    #     ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    #     return ut + u*ux - 0.01/torch.pi*uxx


if __name__ == "__main__":
    print("hello!")