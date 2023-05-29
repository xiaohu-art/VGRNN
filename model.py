import torch.nn as nn
import torch

class VGRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, eps, conv='GCN', bias=False):
        super(VGRNN, self).__init__()
        
        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers