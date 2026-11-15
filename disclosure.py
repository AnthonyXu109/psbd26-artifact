"""Exact outcome disclosure from segment-level exact counts. An unlinked user i could have chosen item j
iff some completion (perfect b-matching of unlinked users to the released counts, respecting histories)
assigns j to i. With the true assignment as reference, edge (i, j) lies in some completion iff it lies on an
alternating cycle, i.e. i and j are in the same strongly connected component of the graph with arcs
user -> admissible unassigned item and item -> assigned user. Forced: exactly one possible item."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

def exact_support(Wg, pos, H):
    J = np.unique(pos[Wg]); col = {int(j): t for t, j in enumerate(J)}; u = len(Wg); m = len(J)
    adm = np.ones((u, m), bool)
    for a, i in enumerate(Wg):
        h = [col[int(j)] for j in H[i] if int(j) in col]
        if h: adm[a, h] = False
    own = np.array([col[int(pos[i])] for i in Wg]); adm[np.arange(u), own] = False
    r, c = np.nonzero(adm)
    rows = np.concatenate([r, u + own]); cols = np.concatenate([u + c, np.arange(u)])
    Gr = sp.csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(u + m, u + m))
    _, lab = connected_components(Gr, directed=True, connection='strong')
    same = lab[u + c] == lab[r]
    cnt = 1 + np.bincount(r[same], minlength=u)
    return cnt
