"""Public segment labels computed from logged history only (no evaluation outcomes)."""
import numpy as np, scipy.sparse as sp, hashlib
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans

def history_matrix(z):
    H_ptr, H_idx = z['H_ptr'], z['H_idx']; n = len(z['users']); nI = int(z['n_items'])
    rows = np.repeat(np.arange(n), np.diff(H_ptr))
    return sp.csr_matrix((np.ones(len(H_idx), np.float64), (rows, H_idx)), shape=(n, nI))

def profile_segments(z, G, f=32, seed=0):
    if G == 1:
        return np.zeros(len(z['users']), np.int64)
    X = history_matrix(z)
    U, s, Vt = svds(X, k=f, random_state=seed)
    E = X @ Vt.T; E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return KMeans(n_clusters=G, n_init=4, random_state=seed).fit_predict(E).astype(np.int64)

def random_segments(z, G):
    users = z['users']
    h = np.array([int.from_bytes(hashlib.sha256(f'seg|{u}'.encode()).digest()[:8], 'big') for u in users], dtype=np.uint64)
    return (h % np.uint64(G)).astype(np.int64)
