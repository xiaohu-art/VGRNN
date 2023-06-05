#!/usr/bin/env python
# coding: utf-8

import torch
import hydra
import time
import logging

from utils import loader, mask_edges_prd, set_seed_everywhere

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
    adj_train_l, train_edges_l, \
        val_edges_l, val_edges_false_l, \
            test_edges_l, test_edges_false_l = mask_edges_prd(adj_time_list)

if __name__ == "__main__":
    main()