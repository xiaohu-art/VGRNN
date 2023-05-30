#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import hydra
import time

from utils import loader, mask_edges_det
from model import VGRNN, MessagePassing

device = "cuda:0" if torch.cuda.is_available() else "cpu"

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    seed = cfg.seed
    np.random.seed(seed)

    '''
    loading data
    '''
    adj_time_list, adj_orig_dense_list = loader(cfg.datasets)

    '''
    masking edges
    '''
    adj_train_l, train_edges_l, \
        val_edges_l, val_edges_false_l, \
            test_edges_l, test_edges_false_l = mask_edges_det(adj_time_list)

    '''
    creating edge list
    '''
    edge_idx_list = []
    for train_edges in train_edges_l:
        edge_idx = torch.tensor(np.transpose(train_edges), dtype=torch.long).to(device)
        edge_idx_list.append(edge_idx)
    
    
    seq_len = len(train_edges_l)
    num_nodes = adj_orig_dense_list[seq_len-1].shape[0]
    
    '''
    creating input tensors
    '''
    x_in_list = []
    for i in range(seq_len):
        x = torch.eye(num_nodes, dtype=torch.float).to(device)
        x_in_list.append(x)

    x_in = Variable(torch.stack(x_in_list)).to(device)     #(seq_len, num_node, num_node)
    
    adj_label_l = []
    for adj_train in adj_train_l:
        adj_label = torch.tensor(adj_train.toarray().astype(np.float32)).to(device)
        adj_label_l.append(adj_label)
    
    '''
    building model
    '''
    
    x_dim = num_nodes
    model = VGRNN(x_dim,
                  cfg.h_dim,
                  cfg.z_dim,
                  cfg.n_layers,
                  cfg.eps,
                  cfg.conv_type,
                  bias = True,
                  device=device)
    
    mp = MessagePassing()
    print(mp.message_args)
    print(mp.update_args)
    exit()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    '''
    training
    '''
    seq_start = 0
    seq_end = seq_len - 3
    tst_after = 0
    for k in range(cfg.epoch):
        optimizer.zero_grad()
        start_time = time.time()

        kld_loss, nll_loss = 0, 0
        # kld_loss, nll_loss, _, _, hidden_st = model(x_in[seq_start:seq_end]
        #                                         , edge_idx_list[seq_start:seq_end]
        #                                         , adj_orig_dense_list[seq_start:seq_end])

        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        nn.utils.clip_grad_norm(model.parameters(), cfg.clip)

        if k % cfg.eval_interval == 0:
            pass

        # if k>tst_after:
        #     _, _, enc_means, _, _ = model(x_in[seq_end:seq_len]
        #                                 , edge_idx_list[seq_end:seq_len]
        #                                 , adj_label_l[seq_end:seq_len]
        #                                 , hidden_st)
            
        #     auc_scores_det_val, ap_scores_det_val = get_roc_scores(val_edges_l[seq_end:seq_len]
        #                                                             , val_edges_false_l[seq_end:seq_len]
        #                                                             , adj_orig_dense_list[seq_end:seq_len]
        #                                                             , enc_means)
            
        #     auc_scores_det_test, ap_scores_det_tes = get_roc_scores(test_edges_l[seq_end:seq_len]
        #                                                             , test_edges_false_l[seq_end:seq_len]
        #                                                             , adj_orig_dense_list[seq_end:seq_len]
        #                                                             , enc_means)
            
        
        # print('epoch: ', k)
        # print('kld_loss =', kld_loss.mean().item())
        # print('nll_loss =', nll_loss.mean().item())
        # print('loss =', loss.mean().item())
        # if k>tst_after:
        #     print('----------------------------------')
        #     print('Link Detection')
        #     print('val_link_det_auc_mean', np.mean(np.array(auc_scores_det_val)))
        #     print('val_link_det_ap_mean', np.mean(np.array(ap_scores_det_val)))
        #     print('test_link_det_auc_mean', np.mean(np.array(auc_scores_det_test)))
        #     print('test_link_det_ap_mean', np.mean(np.array(ap_scores_det_tes)))
        #     print('----------------------------------')
        # print('----------------------------------')


if __name__ == "__main__":
    main()