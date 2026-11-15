"""Full-catalog ranker and slate generation functions used in the paper."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

K = 10

def ease(X, lam=250.0):
    G = (X.T @ X).toarray().astype(np.float64); G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G); B = P / (-np.diag(P)); B[np.diag_indices_from(B)] = 0.0
    return B.astype(np.float32)

def itemknn(X, k=100, shrink=10.0):
    norms = np.sqrt(np.asarray(X.power(2).sum(0)).ravel())
    S = (X.T @ X).toarray().astype(np.float32); S /= (np.outer(norms, norms) + shrink); np.fill_diagonal(S, 0)
    idx = np.argpartition(-S, k, axis=0)[k:]; np.put_along_axis(S, idx, 0.0, axis=0)
    return S

def puresvd(X, f=64):
    U, s, Vt = svds(X.astype(np.float64), k=f, random_state=0)
    return (Vt.T @ Vt).astype(np.float32)

def topk(scores, H):
    out = np.empty((scores.shape[0], K), dtype=np.int64)
    for r in range(scores.shape[0]):
        s = scores[r].astype(np.float64); s[H[r]] = -np.inf
        top = np.argpartition(-s, K)[:K]; out[r] = top[np.lexsort((top, -s[top]))]
    return out

def slates_for(Xfit, Xhist, users, H, pop):
    res = {}
    popscore = np.log1p(pop)[None, :]
    res['pop'] = topk(np.repeat(popscore, len(users), 0), H)
    for name, W in (('knn', itemknn(Xfit)), ('svd', puresvd(Xfit)), ('ease', ease(Xfit))):
        S = np.empty((len(users), K), dtype=np.int64)
        for s in range(0, len(users), 2048):
            u = users[s:s + 2048]; S[s:s + len(u)] = topk(np.asarray(Xhist[u] @ W), H[s:s + len(u)])
        res[name] = S
    return res

def csr(pairs, nU, nI):
    r, c = zip(*pairs); X = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(nU, nI)); X.data[:] = 1; return X
