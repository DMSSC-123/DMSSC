import numpy as np
import random
import torch
from scipy.optimize import linear_sum_assignment
from sklearn import metrics, cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)
import torch_geometric as torchgeo
from torchvision import models

# ============================
# Metrics
# ============================

nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score


# ============================
# Accuracy (Hungarian)
# ============================

def cluster_accuracy(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


# ============================
# Pretrain Model Loader (EfficientNet / DINO)
# ============================

def load_pretrain_model(name):
    if name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
    elif name == 'efficientnet_b1':
        model = models.efficientnet_b1(pretrained=True)
    elif name == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=True)
    elif name == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=True)
    elif name == 'efficientnet_b4':
        model = models.efficientnet_b4(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return torch.nn.Sequential(*list(model.children())[:-1]) 



# ============================
# Random Seed
# ============================

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ============================
# KNN Graph Builder
# ============================

def build_graph(features, K):
    sparse_adj = []
    for i in range(len(features)):
        sparse_adj.append(
            kneighbors_graph(
                features[i].cpu().numpy(),
                K,
                mode='connectivity',
                metric='euclidean'
            )
        )
    return sparse_adj


def build_pyg_data(features, sparse_adj):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datas = []
    for i in range(len(features)):
        pyg_graph = torchgeo.data.Data()
        pyg_graph.x = features[i].to(device)

        edge_index = torch.from_numpy(
            np.transpose(np.stack(sparse_adj[i].nonzero(), axis=1))
        ).to(device)

        pyg_graph.edge_index = torchgeo.utils.to_undirected(edge_index)
        pyg_graph.num_nodes = features[i].shape[0]
        datas.append(pyg_graph)
    return datas


# ============================
# Spectral Clustering (simple)
# ============================

def sklearn_spectral_clustering(C, K):
    S = np.abs(C) + np.abs(C.T)
    spectral = cluster.SpectralClustering(
        n_clusters=K,
        eigen_solver='arpack',
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=0
    )
    spectral.fit(S)
    return spectral.fit_predict(S)


# ----------------------------
# Extra Accuracy
# ----------------------------

def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row, col = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row, col)]) / y_pred.size


def err_rate(gt_s, s):
    return 1.0 - acc(gt_s, s)


# ============================
# Post-processing for Self-Expression
# ============================

def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)

        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            csum = 0
            t = 0
            while True:
                csum += S[t, i]
                if csum > alpha * cL1:
                    Cp[Ind[:t + 1, i], i] = C[Ind[:t + 1, i], i]
                    break
                t += 1
    else:
        Cp = C
    return Cp


def post_proC(C, K, d, ro):
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    C = C - np.diag(np.diag(C)) + np.eye(n, n)
    r = d * K + 1

    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)

    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)

    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)

    spectral = cluster.SpectralClustering(
        n_clusters=K,
        eigen_solver='arpack',
        affinity='precomputed',
        assign_labels='discretize'
    )
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L


def spectral_clustering(C, K, d, alpha, ro):
    C = thrC(C, alpha)
    y, _ = post_proC(C, K, d, ro)
    return y

