#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import hydra
import time
import logging

from utils import loader, get_roc_scores, set_seed_everywhere, visualize
from utils import mask_edges_det, mask_edges_prd, mask_edges_prd_new
from model import VGRNN

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    seed = cfg.seed
    set_seed_everywhere(seed)

    '''
    loading data
    '''
    adj_time_list, adj_orig_dense_list = loader(cfg.datasets)

    '''
    masking edges
    '''
    outs = mask_edges_det(adj_time_list)
    train_edges_l = outs[1]

    pos_edges_l, false_edges_l = mask_edges_prd(adj_time_list)

    pos_edges_l_n, false_edges_l_n = mask_edges_prd_new(adj_time_list, adj_orig_dense_list)

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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    '''
    training
    '''
    seq_start = 0
    seq_end = seq_len - cfg.val_len
    
    kld_losses = []
    nll_losses = []
    losses = []

    auc_prd = []
    ap_prd = []
    auc_prd_new = []
    ap_prd_new = []

    for k in range(cfg.epoch):
        optimizer.zero_grad()
        start_time = time.time()

        kld_loss, nll_loss, _, _, hidden_st = model(x_in[seq_start:seq_end]
                                                , edge_idx_list[seq_start:seq_end]
                                                , adj_orig_dense_list[seq_start:seq_end])

        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        kld_losses.append(kld_loss.item())
        nll_losses.append(nll_loss.item())
        losses.append(loss.item())

        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)

        if k % cfg.eval_interval == 0:
            _, _, enc_means, pri_means, _ = model(x_in[seq_end:seq_len]
                                        , edge_idx_list[seq_end:seq_len]
                                        , adj_orig_dense_list[seq_end:seq_len]
                                        , hidden_st)

            auc_score_prd, ap_score_prd = get_roc_scores(pos_edges_l[seq_end:seq_len]
                                                        , false_edges_l[seq_end:seq_len]
                                                        , adj_orig_dense_list[seq_end:seq_len]
                                                        , pri_means)
            
            auc_score_prd_new, ap_score_prd_new = get_roc_scores(pos_edges_l_n[seq_end:seq_len]
                                                                , false_edges_l_n[seq_end:seq_len]
                                                                , adj_orig_dense_list[seq_end:seq_len]
                                                                , pri_means)
            
            auc_prd.append(np.mean(np.array(auc_score_prd)))
            ap_prd.append(np.mean(np.array(ap_score_prd)))
            auc_prd_new.append(np.mean(np.array(auc_score_prd_new)))
            ap_prd_new.append(np.mean(np.array(ap_score_prd_new)))

            visualize(kld_losses, nll_losses, losses, auc_prd, ap_prd, auc_prd_new, ap_prd_new, cfg)
            logger.info(f"Epoch: {k}, kld loss: {kld_loss.item()}, nll loss: {nll_loss.item()}, loss: {loss.item()}, time: {time.time() - start_time}s")

    logger.info(f"Best auc val: {np.max(np.array(auc_prd))}")
    logger.info(f"Best ap val: {np.max(np.array(ap_prd))}")
    logger.info(f"Best auc test: {np.max(np.array(auc_prd_new))}")
    logger.info(f"Best ap test: {np.max(np.array(ap_prd_new))}") 

if __name__ == "__main__":
    main()