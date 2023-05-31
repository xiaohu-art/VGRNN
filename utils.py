import pickle
import numpy as np
import scipy.sparse as sp
import random
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import matplotlib.pyplot as plt

def set_seed_everywhere(seed=0):
    """
    Set seed for all randomness sources
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def loader(datasets):

    with open('data/{data}/adj_time_list.pickle'.format(data = datasets), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')

    with open('data/{data}/adj_orig_dense_list.pickle'.format(data = datasets), 'rb') as handle:
        adj_orig_dense_list = pickle.load(handle, encoding='bytes')

    return adj_time_list, adj_orig_dense_list

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_edges_det(adjs_list):
    adj_train_l, train_edges_l, val_edges_l = [], [], []
    val_edges_false_l, test_edges_l, test_edges_false_l = [], [], []
    edges_list = []

    for i in range(len(adjs_list)):
        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        
        assert np.diag(adj.todense()).sum() == 0
        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = sparse_to_tuple(adj)[0]

        num_test = int(np.floor(edges.shape[0] / 10.))
        num_val = int(np.floor(edges.shape[0] / 20.))

        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)

        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]

        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        edges_list.append(edges)

        # Checking elements in a: list are or aren't in b
        def ismember(a, b, tol=5):  
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

        assert ~ismember(test_edges_false, edges_all)
        assert ~ismember(val_edges_false, edges_all)
        assert ~ismember(val_edges, train_edges)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)

        data = np.ones(train_edges.shape[0])

        # Re-build adj matrix
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T

        adj_train_l.append(adj_train)                           # train adjacent matrixes list
        train_edges_l.append(train_edges)                       # train edges list
        val_edges_l.append(val_edges)                           # validation positive edges list
        val_edges_false_l.append(np.array(val_edges_false))     # validation negtive edges list
        test_edges_l.append(test_edges)                         # test positive edges list
        test_edges_false_l.append(np.array(test_edges_false))   # test negtive edges list

    return adj_train_l, train_edges_l, val_edges_l, val_edges_false_l, test_edges_l, test_edges_false_l

@torch.no_grad()
def get_roc_scores(edges_pos, edges_neg, adj_orig_dense_list, embs):
    def sigmoid(x):
        if x < 0:
            return np.exp(x) / (1+np.exp(x))
        else:
            return 1 / (1 + np.exp(-x))
    
    auc_scores = []
    ap_scores = []

    for i in range(len(edges_pos)):
        # Predict on test set of edges
        emb = embs[i].cpu().numpy()
        adj_rec = np.dot(emb, emb.T)
        adj_orig_t = adj_orig_dense_list[i]
        preds = []
        pos = []
        for e in edges_pos[i]:
            preds.append(adj_rec[e[0], e[1]])
            pos.append(adj_orig_t[e[0], e[1]])
            
        preds_neg = []
        neg = []
        for e in edges_neg[i]:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig_t[e[0], e[1]])
        
        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        auc_scores.append(roc_auc_score(labels_all, preds_all))
        ap_scores.append(average_precision_score(labels_all, preds_all))

    return auc_scores, ap_scores

def visualize(kld_losses, nll_losses, losses, auc_val, ap_val, auc_test, ap_test, cfg):
    
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plt.title("kld loss, nll loss and total loss")
    plt.plot(kld_losses, label="kld loss")
    plt.plot(nll_losses, label="nll loss")
    plt.plot(losses, label="loss")
    plt.legend()
    plt.xlabel("epoch")

    plt.subplot(1, 2, 2)
    plt.title("auc scores and ap scores")
    plt.plot(auc_val, label="Val set auc score")
    plt.plot(ap_val, label="Val set ap score")
    plt.plot(auc_test, label="Test set auc score")
    plt.plot(ap_test, label="Test set ap score")
    plt.legend()
    plt.xlabel("epoch / interval")

    plt.savefig(f"{cfg.conv_type}_{cfg.datasets}.png")
    plt.close()