import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch_geometric.nn import GCNConv

class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()
        
        self.act = act
        self.dropout = dropout
    
    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)

class graph_gru_gcn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True, device="cuda"):
        super(graph_gru_gcn, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer
        
        self.device = device

        # gru weights
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        
        for i in range(self.n_layer):
            if i==0:
                self.weight_xz.append(GCNConv(input_size, hidden_size, act=lambda x:x, bias=bias).to(device))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias).to(device))
                self.weight_xr.append(GCNConv(input_size, hidden_size, act=lambda x:x, bias=bias).to(device))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias).to(device))
                self.weight_xh.append(GCNConv(input_size, hidden_size, act=lambda x:x, bias=bias).to(device))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias).to(device))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias).to(device))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias).to(device))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias).to(device))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias).to(device))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias).to(device))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias).to(device))
    
    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size()).to(self.device)
        for i in range(self.n_layer):
            if i==0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i-1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i-1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i-1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))

            h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g

        
        out = h_out
        return out, h_out

class VGRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, eps, conv='GCN', bias=False, device="cuda"):
        super(VGRNN, self).__init__()
        
        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        self.device = device

        if conv == "GCN":
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU()).to(device)
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU()).to(device)

            self.enc = GCNConv(h_dim + h_dim, h_dim).to(device)
            self.enc_mean = GCNConv(h_dim, z_dim, act=lambda x:x).to(device)
            self.enc_std = GCNConv(h_dim, z_dim, act=F.softplus).to(device)

            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU()).to(device)
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim)).to(device)
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus()).to(device)

            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias, device).to(device)

        else:
            raise NotImplementedError
        
        # self.decoder = InnerProductDecoder(act=lambda x: x).to(device)
        self.decoder = InnerProductDecoder().to(device)
        
    def forward(self, x, edge_idx_list, adj_orig_dense_list, hidden_in=None):
        assert len(adj_orig_dense_list) == len(edge_idx_list)

        kld_loss = 0
        nll_loss = 0
        all_enc_mean, all_enc_std = [], []
        all_prior_mean, all_prior_std = [], []

        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        else:
            h = Variable(hidden_in).to(self.device)

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t]).to(self.device)
            
            # Encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], dim=1), edge_idx_list[t]).to(self.device)
            enc_mean_t = self.enc_mean(enc_t, edge_idx_list[t]).to(self.device)
            enc_std_t = self.enc_std(enc_t, edge_idx_list[t]).to(self.device)

            # Prior
            prior_t = self.prior(h[-1]).to(self.device)
            prior_mean_t = self.prior_mean(prior_t).to(self.device)
            prior_std_t = self.prior_std(prior_t).to(self.device)

            # Sampling and Reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t).to(self.device)
            phi_z_t = self.phi_z(z_t).to(self.device)

            # Decoder
            dec_t = self.decoder(z_t).to(self.device)
            
            # Recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1), edge_idx_list[t], h)

            nnodes = adj_orig_dense_list[t].size()[0]
            enc_mean_t_sl = enc_mean_t[0:nnodes, :]
            enc_std_t_sl = enc_std_t[0:nnodes, :]
            prior_mean_t_sl = prior_mean_t[0:nnodes, :]
            prior_std_t_sl = prior_std_t[0:nnodes, :]
            dec_t_sl = dec_t[0:nnodes, 0:nnodes]

            # Computing losses
            kld_loss += self._kld_gauss(enc_mean_t_sl, enc_std_t_sl, prior_mean_t_sl, prior_std_t_sl)
            adj_orig_dense_list[t] = adj_orig_dense_list[t].to(self.device)
            nll_loss += self._nll_bernoulli(dec_t_sl, adj_orig_dense_list[t])
            
            all_enc_std.append(enc_std_t_sl)
            all_enc_mean.append(enc_mean_t_sl)
            all_prior_mean.append(prior_mean_t_sl)
            all_prior_std.append(prior_std_t_sl)

        return kld_loss, nll_loss, all_enc_mean, all_prior_mean, h
    

    def _reparameterized_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps1 = Variable(eps1).to(self.device)
        return eps1.mul(std).add_(mean)
    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
     
    def _init_weights(self, stdv):
        pass

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) -
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) + 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=0), dim=0)
    
    def _kld_gauss_bak(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)
    
    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits
                                                          , target=target_adj_dense
                                                          , pos_weight=posw
                                                          , reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0,1])
        return - nll_loss