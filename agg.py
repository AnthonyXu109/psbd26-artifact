"""Aggregate-constrained sharp bounds for a paired top-k ranking metric.

Setting (one designated relevant item per user, r_i = 1):
  user i has a known candidate universe C_i = catalog minus its logged history H_i,
  two known slates A_i, B_i, and contribution q_i(j) = g_A(j) - g_B(j) if its relevant
  item is j (g = rank discount for NDCG@k, indicator for Hit/Recall@k).
  Linked users: relevant item known.  Unlinked users W: relevant item unknown, but a
  release gives the multiset of their relevant items, either exactly (counts c_j) or
  through a box lo_j <= c_j <= hi_j with sum_j c_j = |W|.

Sharp bounds on sum_{i in W} q_i(s_i) are the optimal values of a transportation LP,
which is integral (bipartite incidence matrix is totally unimodular).  We solve a
reduced network: explicit arcs only for pairs with q_i(j) != 0; all other choices are
routed through a zero-cost pool.  The reduced LP is a relaxation (outer bound); an
explicit completion step then certifies that the optimum is attained by a legal
assignment, in which case the endpoint is sharp.
"""
from __future__ import annotations
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse.csgraph import maximum_flow

K = 10
DISC = 1.0 / np.log2(np.arange(2, K + 2))


def contributions(SA: np.ndarray, SB: np.ndarray, metric: str = 'ndcg'):
    """Return (items, q) arrays of shape (n, 2k); duplicate items merged, q==0 padded."""
    n, k = SA.shape
    g = DISC[:k] if metric == 'ndcg' else np.ones(k)
    items = np.concatenate([SA, SB], 1)
    q = np.concatenate([np.tile(g, (n, 1)), -np.tile(g, (n, 1))], 1)
    order = np.argsort(items, axis=1, kind='stable')
    it = np.take_along_axis(items, order, 1)
    qq = np.take_along_axis(q, order, 1).copy()
    for c in range(1, it.shape[1]):
        same = it[:, c] == it[:, c - 1]
        qq[same, c - 1] += qq[same, c]
        qq[same, c] = 0.0
    qq[np.abs(qq) < 1e-12] = 0.0
    return it, qq


def realized(it, qq, pos):
    hit = it == pos[:, None]
    return (qq * hit).sum(1)


def per_user_bounds(qq):
    """Sharp bounds when unlinked users are independent (no aggregate).  With a full
    catalog every user has off-slate candidates, so 0 is always attainable."""
    return np.minimum(qq.min(1), 0.0), np.maximum(qq.max(1), 0.0)


def _build(it, qq, item_lo, item_hi, other_lo, other_hi, items_list):
    n = it.shape[0]
    J = len(items_list)
    col = {j: t for t, j in enumerate(items_list)}
    rows, cols = np.nonzero(qq != 0)
    jj = it[rows, cols]
    tj = np.fromiter((col.get(int(j), -1) for j in jj), dtype=np.int64, count=len(jj))
    keep = tj >= 0
    rows, cols, jj, tj = rows[keep], cols[keep], jj[keep], tj[keep]
    cost = qq[rows, cols]
    nx = len(rows)
    nv = nx + n + J + 1
    A_user = sp.csr_matrix((np.ones(nx + n), (np.concatenate([rows, np.arange(n)]),
                            np.concatenate([np.arange(nx), nx + np.arange(n)]))), shape=(n, nv))
    A_pool = sp.csr_matrix((np.concatenate([np.ones(n), -np.ones(J + 1)]),
                            (np.zeros(n + J + 1, dtype=np.int64), nx + np.arange(n + J + 1))), shape=(1, nv))
    A_item = sp.csr_matrix((np.ones(nx + J), (np.concatenate([tj, np.arange(J)]),
                            np.concatenate([np.arange(nx), nx + n + np.arange(J)]))), shape=(J, nv))
    Aeq = sp.vstack([A_user, A_pool]).tocsr()
    beq = np.concatenate([np.ones(n), [0.0]])
    Aub = sp.vstack([A_item, -A_item]).tocsr()
    bub = np.concatenate([item_hi, -item_lo])
    bounds = np.array([(0, 1)] * (nx + n) + [(0, None)] * J + [(other_lo, other_hi)], dtype=float)
    return dict(n=n, nx=nx, J=J, rows=rows, jj=jj, cost=cost, Aeq=Aeq, beq=beq, Aub=Aub, bub=bub, bounds=bounds)


def _lp(P, sense):
    c = np.concatenate([sense * P['cost'], np.zeros(P['n'] + P['J'] + 1)])
    r = linprog(c, A_ub=P['Aub'], b_ub=P['bub'], A_eq=P['Aeq'], b_eq=P['beq'],
                bounds=P['bounds'], method='highs-ds')
    if r.status != 0:
        raise RuntimeError(f'LP failed: {r.message}')
    return sense * r.fun, r.x


def aggregate_bounds(it, qq, lo, hi, n_items, H=None, verify=True):
    """Bounds on sum_{i in W} q_i(s_i) given catalog-length count bounds lo<=c<=hi for
    the unlinked users' relevant items.  H: list of per-user history arrays (for the
    completion certificate).  Returns dict with L, U and certification flags."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    active = np.unique(it[qq != 0])
    items_list = np.array([j for j in active if hi[j] > 0], dtype=np.int64)
    others = np.ones(n_items, bool); others[items_list] = False
    P = _build(it, qq, lo[items_list], hi[items_list], lo[others].sum(), hi[others].sum(), list(items_list))
    out = {}
    for name, sense in (('L', 1.0), ('U', -1.0)):
        val, x = _lp(P, sense)
        out[name] = val
        if verify:
            out[name + '_certified'] = certify(P, x, it, qq, lo, hi, items_list, others, H)
    return out


def certify(P, x, it, qq, lo, hi, items_list, others, H):
    """Check that the relaxed optimum is attained by a legal integral assignment.
    Pool users must receive items that are neither in their history nor active for them
    (active items would carry a nonzero contribution)."""
    n, nx, J = P['n'], P['nx'], P['J']
    xa = np.rint(x[:nx]); z = np.rint(x[nx:nx + n]); y = np.rint(x[nx + n:nx + n + J]); yo = np.rint(x[-1])
    if (np.abs(x[:nx] - xa).max(initial=0) > 1e-6 or np.abs(x[nx:] - np.rint(x[nx:])).max(initial=0) > 1e-6):
        return False  # non-integral vertex (should not happen)
    pool_users = np.flatnonzero(z > 0.5)
    if len(pool_users) == 0:
        return True
    # units that pool users must absorb: explicit items y_j, plus yo units spread over
    # 'other' items within their [lo, hi] boxes.  For 'other' items we pick counts
    # greedily inside boxes (any split is admissible).
    units = {int(items_list[t]): int(y[t]) for t in range(J) if y[t] > 0}
    other_idx = np.flatnonzero(others)
    need = int(yo)
    base = np.floor(lo[other_idx]).astype(np.int64)
    rem = need - int(base.sum())
    if rem < 0:
        return False
    extra_cap = (np.floor(hi[other_idx]) - base).astype(np.int64)
    # spread the remainder over items with the largest capacity (keeps counts small)
    order = np.argsort(-extra_cap, kind='stable')
    for t in order:
        if rem == 0:
            break
        a = min(rem, int(extra_cap[t])); base[t] += a; rem -= a
    if rem > 0:
        return False
    for j, cnt in zip(other_idx[base > 0], base[base > 0]):
        units[int(j)] = units.get(int(j), 0) + int(cnt)
    # blocked sets for pool users
    blocked = []
    for i in pool_users:
        b = set(int(j) for j in it[i][qq[i] != 0])
        if H is not None:
            b.update(int(j) for j in H[i])
        blocked.append(b)
    # greedy: most constrained users first, give item with largest remaining units
    item_keys = sorted(units, key=lambda j: -units[j])
    remaining = dict(units)
    order_u = sorted(range(len(pool_users)), key=lambda a: -len(blocked[a]))
    unassigned = []
    users_of = {}
    for a in order_u:
        for j in sorted(remaining, key=lambda j: -remaining[j]) if len(remaining) < 64 else item_keys:
            if remaining.get(j, 0) > 0 and j not in blocked[a]:
                remaining[j] -= 1; users_of.setdefault(j, []).append(a)
                break
        else:
            unassigned.append(a)
    if not unassigned:
        return True
    # one-step augmenting repair: give a an item j held by b, move b to an item with spare units
    still = []
    for a in unassigned:
        spare = [r for r, v in remaining.items() if v > 0]
        done = False
        for j in item_keys:
            if j in blocked[a] or j not in users_of:
                continue
            for pos_b, b in enumerate(users_of[j][:200]):
                for r in spare:
                    if r not in blocked[b]:
                        remaining[r] -= 1; users_of.setdefault(r, []).append(b)
                        users_of[j][pos_b] = a; done = True; break
                if done: break
            if done: break
        if not done:
            still.append(a)
    if not still:
        return True
    # exact fallback: bipartite max flow on the explicit legal graph (only when small)
    if len(pool_users) * len(units) > 5_000_000:
        return False
    keys = [j for j in units]
    kidx = {j: t for t, j in enumerate(keys)}
    U = len(pool_users); Kc = len(keys)
    src, snk = U + Kc, U + Kc + 1
    r_, c_, w_ = [], [], []
    for a in range(U):
        r_.append(src); c_.append(a); w_.append(1)
        for j in keys:
            if j not in blocked[a]:
                r_.append(a); c_.append(U + kidx[j]); w_.append(1)
    for j in keys:
        r_.append(U + kidx[j]); c_.append(snk); w_.append(units[j])
    G = sp.csr_matrix((np.array(w_, dtype=np.int32), (r_, c_)), shape=(U + Kc + 2, U + Kc + 2))
    return int(maximum_flow(G, src, snk).flow_value) == U


def dp_box(true_counts_all, linked_counts, eps, alpha, rng, bins=None):
    """Laplace release of per-item counts over the whole cohort (replace-one adjacency,
    L1 sensitivity 2), turned into a simultaneous (1-alpha) box for the unlinked counts.
    bins: boolean mask of released items (default: whole catalog)."""
    m_items = len(true_counts_all)
    if bins is None:
        bins = np.ones(m_items, bool)
    m = int(bins.sum())
    b = 2.0 / eps
    t = b * np.log(m / alpha)
    noisy = true_counts_all + rng.laplace(0.0, b, size=m_items)
    lo = np.where(bins, np.maximum(0.0, np.ceil(noisy - t - linked_counts - 1e-9)), 0.0)
    hi = np.where(bins, np.maximum(0.0, np.floor(noisy + t - linked_counts + 1e-9)), np.inf)
    return lo, hi, t


def exact_dense(it, qq, lo, hi, H, n_items, max_vars=30_000_000):
    """Full transportation LP over all legal (user, item) pairs with hi_j > 0."""
    n = it.shape[0]
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    items = np.flatnonzero(hi > 0)
    col = -np.ones(n_items, np.int64); col[items] = np.arange(len(items))
    R, C, W = [], [], []
    for i in range(n):
        ok = np.ones(n_items, bool); ok[np.asarray(H[i], dtype=np.int64)] = False; ok &= hi > 0
        js = np.flatnonzero(ok)
        w = np.zeros(n_items); act = qq[i] != 0; w[it[i][act]] = qq[i][act]
        R.append(np.full(len(js), i)); C.append(js); W.append(w[js])
    R = np.concatenate(R); C = np.concatenate(C); W = np.concatenate(W)
    if len(R) > max_vars:
        raise MemoryError('dense LP too large')
    nv = len(R)
    Aeq = sp.csr_matrix((np.ones(nv), (R, np.arange(nv))), shape=(n, nv))
    Ai = sp.csr_matrix((np.ones(nv), (col[C], np.arange(nv))), shape=(len(items), nv))
    Aub = sp.vstack([Ai, -Ai]).tocsr(); bub = np.concatenate([hi[items], -lo[items]])
    out = {}
    for name, sense in (('L', 1.0), ('U', -1.0)):
        r = linprog(sense * W, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=np.ones(n), bounds=(0, 1), method='highs-ds')
        if r.status != 0:
            raise RuntimeError(r.message)
        out[name] = sense * r.fun
    return out


# ---------------------------------------------------------------------------
# Lazy exact refinement: items on which pooled users conflict are modelled with
# explicit arcs from every admissible user (cost q_i(j), zero if inactive), and
# the pool may no longer route to them.  Repeat until the completion certifies.

def _build_S(it, qq, lo_list, hi_list, other_lo, other_hi, items_list, S, H):
    n = it.shape[0]; J = len(items_list)
    col = {int(j): t for t, j in enumerate(items_list)}
    Sset = set(int(j) for j in S)
    rows, cols = np.nonzero(qq != 0)
    jj = it[rows, cols]
    keep = np.array([(int(j) in col) and (int(j) not in Sset) for j in jj], dtype=bool)
    rows, jj = rows[keep], jj[keep]; cost = qq[rows, cols[keep]]
    # explicit arcs for S items from every user that may choose them
    sr, sj, sc = [], [], []
    qmap = [dict() for _ in range(n)]
    for i, c in zip(*np.nonzero(qq != 0)):
        qmap[i][int(it[i, c])] = qq[i, c]
    for j in Sset:
        if j not in col:
            continue
        for i in range(n):
            if H is not None and j in H[i]:
                continue
            sr.append(i); sj.append(j); sc.append(qmap[i].get(j, 0.0))
    rows = np.concatenate([rows, np.array(sr, dtype=np.int64)]); jj = np.concatenate([jj, np.array(sj, dtype=np.int64)])
    cost = np.concatenate([cost, np.array(sc, dtype=float)])
    tj = np.array([col[int(j)] for j in jj], dtype=np.int64)
    nx = len(rows); nv = nx + n + J + 1
    A_user = sp.csr_matrix((np.ones(nx + n), (np.concatenate([rows, np.arange(n)]), np.concatenate([np.arange(nx), nx + np.arange(n)]))), shape=(n, nv))
    A_pool = sp.csr_matrix((np.concatenate([np.ones(n), -np.ones(J + 1)]), (np.zeros(n + J + 1, dtype=np.int64), nx + np.arange(n + J + 1))), shape=(1, nv))
    A_item = sp.csr_matrix((np.ones(nx + J), (np.concatenate([tj, np.arange(J)]), np.concatenate([np.arange(nx), nx + n + np.arange(J)]))), shape=(J, nv))
    Aeq = sp.vstack([A_user, A_pool]).tocsr(); beq = np.concatenate([np.ones(n), [0.0]])
    Aub = sp.vstack([A_item, -A_item]).tocsr(); bub = np.concatenate([hi_list, -lo_list])
    ybounds = [(0, 0) if int(items_list[t]) in Sset else (0, None) for t in range(J)]
    bounds = np.array([(0, 1)] * (nx + n) + ybounds + [(other_lo, other_hi)], dtype=float)
    return dict(n=n, nx=nx, J=J, rows=rows, jj=jj, cost=cost, Aeq=Aeq, beq=beq, Aub=Aub, bub=bub, bounds=bounds)


def _certify_S(P, x, it, qq, lo, hi, items_list, others, H, S):
    """Like certify, but pooled users only need items outside S; returns (ok, conflict_items)."""
    ok = certify(P, x, it, qq, lo, hi, items_list, others, H)
    if ok:
        return True, set()
    n, nx, J = P['n'], P['nx'], P['J']
    z = np.rint(x[nx:nx + n]); y = np.rint(x[nx + n:nx + n + J])
    pool_users = np.flatnonzero(z > 0.5)
    act_pool = set()
    for i in pool_users:
        act_pool.update(int(j) for j in it[i][qq[i] != 0])
    conflicts = {int(items_list[t]) for t in range(J) if y[t] > 0 and int(items_list[t]) in act_pool}
    return False, conflicts - set(S)


def aggregate_bounds_exact(it, qq, lo, hi, n_items, H, max_iter=8, max_arcs=4_000_000):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    active = np.unique(it[qq != 0])
    items_list = np.array([j for j in active if hi[j] > 0], dtype=np.int64)
    others = np.ones(n_items, bool); others[items_list] = False
    out = {}
    for name, sense in (('L', 1.0), ('U', -1.0)):
        S = set(); certified = False
        for itn in range(max_iter):
            P = _build_S(it, qq, lo[items_list], hi[items_list], lo[others].sum(), hi[others].sum(), items_list, S, H)
            val, x = _lp(P, sense)
            ok, conf = _certify_S(P, x, it, qq, lo, hi, items_list, others, H, S)
            if ok:
                certified = True; break
            if not conf or len(S | conf) * it.shape[0] > max_arcs:
                break          # keep the valid outer bound from the last solve
            S |= conf
        out[name] = val; out[name + '_certified'] = certified; out[name + '_S'] = len(S); out[name + '_iter'] = itn + 1
    return out


# ---------------------------------------------------------------------------
# Grouped release: counts are released per item group (a partition of the catalog).
# Singleton groups reproduce item-level counts.  The constraint matrix stays a
# bipartite incidence matrix (users x groups), so the LP is still integral.

def aggregate_bounds_grouped(it, qq, group, lo_g, hi_g, H, max_iter=8, max_arcs=4_000_000):
    """group: catalog-length array item -> group id; lo_g/hi_g: bounds per group.
    Returns dict(L, U, L_certified, U_certified)."""
    n = it.shape[0]; Gn = int(group.max()) + 1
    gsize = np.bincount(group, minlength=Gn)
    lo_g = np.asarray(lo_g, float); hi_g = np.asarray(hi_g, float)
    rows0, cols0 = np.nonzero(qq != 0)
    jj0 = it[rows0, cols0]; c0 = qq[rows0, cols0]
    act = [set() for _ in range(n)]
    for i, j in zip(rows0, jj0):
        act[i].add(int(j))
    Hs = [set(int(j) for j in h) for h in H]
    order_g = np.argsort(group, kind='stable'); bounds_g = np.concatenate([[0], np.cumsum(gsize)])
    items_of = [order_g[bounds_g[g]:bounds_g[g + 1]] for g in range(Gn)]
    out = {}
    for name, sense in (('L', 1.0), ('U', -1.0)):
        S = set()   # singleton groups modelled exactly
        cert = False
        for _ in range(max_iter):
            # explicit arcs: active arcs to non-S groups; for S groups all admissible users
            keep = np.array([group[j] not in S for j in jj0], dtype=bool)
            rows = list(rows0[keep]); gj = list(group[jj0[keep]]); cost = list(c0[keep])
            for g in S:
                j = int(items_of[g][0])
                for i in range(n):
                    if j in Hs[i]:
                        continue
                    rows.append(i); gj.append(g); cost.append(qq[i][it[i] == j].sum() if j in act[i] else 0.0)
            rows = np.array(rows, dtype=np.int64); gj = np.array(gj, dtype=np.int64); cost = np.array(cost, float)
            nx = len(rows); nv = nx + n + Gn
            A_user = sp.csr_matrix((np.ones(nx + n), (np.concatenate([rows, np.arange(n)]), np.concatenate([np.arange(nx), nx + np.arange(n)]))), shape=(n, nv))
            A_pool = sp.csr_matrix((np.concatenate([np.ones(n), -np.ones(Gn)]), (np.zeros(n + Gn, dtype=np.int64), nx + np.arange(n + Gn))), shape=(1, nv))
            A_g = sp.csr_matrix((np.ones(nx + Gn), (np.concatenate([gj, np.arange(Gn)]), np.concatenate([np.arange(nx), nx + n + np.arange(Gn)]))), shape=(Gn, nv))
            Aeq = sp.vstack([A_user, A_pool]).tocsr(); beq = np.concatenate([np.ones(n), [0.0]])
            Aub = sp.vstack([A_g, -A_g]).tocsr(); bub = np.concatenate([hi_g, -lo_g])
            bounds = [(0, 1)] * (nx + n) + [((0, 0) if g in S else (0, None)) for g in range(Gn)]
            r = linprog(np.concatenate([sense * cost, np.zeros(n + Gn)]), A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs-ds')
            if r.status != 0:
                raise RuntimeError(r.message)
            x = r.x; val = sense * r.fun
            z = np.rint(x[nx:nx + n]); y = np.rint(x[nx + n:])
            pool = np.flatnonzero(z > 0.5)
            units = {g: int(y[g]) for g in range(Gn) if y[g] > 0}
            # blocked(i,g): every item of g is active for i or in its history
            def blocked(i, g):
                if gsize[g] > len(act[i]) + len(Hs[i]):
                    return False
                items = items_of[g]
                return all((int(j) in act[i]) or (int(j) in Hs[i]) for j in items)
            rem = dict(units); fail = []
            order = sorted(pool, key=lambda i: -(len(act[i]) + len(Hs[i])))
            for i in order:
                for g in sorted(rem, key=lambda g: -rem[g]):
                    if rem[g] > 0 and not blocked(i, g):
                        rem[g] -= 1; break
                else:
                    fail.append(i)
            if not fail:
                cert = True; break
            conf = {g for g in units if gsize[g] == 1 and any(int(items_of[g][0]) in act[i] for i in pool)} - S
            if not conf or len(S | conf) * it.shape[0] > max_arcs:
                break          # keep the valid outer bound from the last solve
            S |= conf
        out[name] = val; out[name + '_certified'] = cert
    return out


# ---------------------------------------------------------------------------
# Several relevant items per user with known count r_i (user rows sum to r_i, x_ij <= 1).

def per_user_bounds_multi(qq, r):
    lo = np.zeros(len(r)); hi = np.zeros(len(r))
    for i in range(len(r)):
        v = np.sort(qq[i]); ri = int(r[i])
        neg = v[v < 0][:ri]; pos = v[v > 0][::-1][:ri]
        lo[i] = neg.sum(); hi[i] = pos.sum()
    return lo, hi


def aggregate_bounds_multi(it, qq, r, lo, hi, n_items, H, max_iter=8, max_arcs=4_000_000):
    n = it.shape[0]; r = np.asarray(r, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    active = np.unique(it[qq != 0])
    items_list = np.array([j for j in active if hi[j] > 0], dtype=np.int64)
    others = np.ones(n_items, bool); others[items_list] = False
    col = {int(j): t for t, j in enumerate(items_list)}; J = len(items_list)
    act = [set(int(j) for j in it[i][qq[i] != 0]) for i in range(n)]
    Hs = [set(int(j) for j in h) for h in H]
    qmap = [dict() for _ in range(n)]
    for i, c in zip(*np.nonzero(qq != 0)):
        qmap[i][int(it[i, c])] = qq[i, c]
    out = {}
    for name, sense in (('L', 1.0), ('U', -1.0)):
        S = set(); cert = False
        for _ in range(max_iter):
            rows, jj, cost = [], [], []
            for i in range(n):
                for j, v in qmap[i].items():
                    if j in col and j not in S:
                        rows.append(i); jj.append(j); cost.append(v)
            for j in S:
                for i in range(n):
                    if j not in Hs[i]:
                        rows.append(i); jj.append(j); cost.append(qmap[i].get(j, 0.0))
            rows = np.array(rows, dtype=np.int64); tj = np.array([col[j] for j in jj], dtype=np.int64); cost = np.array(cost, float)
            nx = len(rows); nv = nx + n + J + 1
            A_user = sp.csr_matrix((np.ones(nx + n), (np.concatenate([rows, np.arange(n)]), np.concatenate([np.arange(nx), nx + np.arange(n)]))), shape=(n, nv))
            A_pool = sp.csr_matrix((np.concatenate([np.ones(n), -np.ones(J + 1)]), (np.zeros(n + J + 1, dtype=np.int64), nx + np.arange(n + J + 1))), shape=(1, nv))
            A_item = sp.csr_matrix((np.ones(nx + J), (np.concatenate([tj, np.arange(J)]), np.concatenate([np.arange(nx), nx + n + np.arange(J)]))), shape=(J, nv))
            Aeq = sp.vstack([A_user, A_pool]).tocsr(); beq = np.concatenate([r, [0.0]])
            Aub = sp.vstack([A_item, -A_item]).tocsr(); bub = np.concatenate([hi[items_list], -lo[items_list]])
            bnd = [(0, 1)] * nx + [(0, r[i]) for i in range(n)] + [((0, 0) if int(items_list[t]) in S else (0, None)) for t in range(J)] + [(lo[others].sum(), hi[others].sum())]
            res = linprog(np.concatenate([sense * cost, np.zeros(n + J + 1)]), A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bnd, method='highs-ds')
            if res.status != 0:
                raise RuntimeError(res.message)
            x = res.x; val = sense * res.fun
            xa = np.rint(x[:nx]); z = np.rint(x[nx:nx + n]).astype(int); y = np.rint(x[nx + n:nx + n + J]).astype(int); yo = int(np.rint(x[-1]))
            # items each user already takes directly (must stay distinct)
            taken = [set() for _ in range(n)]
            for k in np.flatnonzero(xa > 0.5):
                taken[rows[k]].add(jj[k])
            units = {int(items_list[t]): int(y[t]) for t in range(J) if y[t] > 0}
            oi = np.flatnonzero(others); base = np.floor(lo[oi]).astype(np.int64); rem_o = yo - int(base.sum())
            cap = (np.floor(np.minimum(hi[oi], 1e9)) - base).astype(np.int64)
            for t in np.argsort(-cap, kind='stable'):
                if rem_o <= 0: break
                a = min(rem_o, int(cap[t])); base[t] += a; rem_o -= a
            for j, cnt in zip(oi[base > 0], base[base > 0]):
                units[int(j)] = units.get(int(j), 0) + int(cnt)
            pool = [i for i in range(n) if z[i] > 0]
            ok = rem_o <= 0
            if ok and pool:
                # greedy first: most constrained users take distinct items with most remaining units
                remg = dict(units); gfail = False
                for i in sorted(pool, key=lambda i: -(len(act[i]) + len(Hs[i]))):
                    need = int(z[i]); bl = act[i] | Hs[i] | taken[i]
                    for j in sorted(remg, key=lambda j: -remg[j]):
                        if need == 0: break
                        if remg[j] > 0 and j not in bl:
                            remg[j] -= 1; need -= 1
                    if need > 0:
                        gfail = True; break
                if not gfail:
                    cert = True; break
                ok = False
            if ok and pool and False:
                keys = list(units); kidx = {j: t for t, j in enumerate(keys)}
                U = len(pool); K = len(keys); src, snk = U + K, U + K + 1
                r_, c_, w_ = [], [], []
                for a_, i in enumerate(pool):
                    r_.append(src); c_.append(a_); w_.append(int(z[i]))
                    bl = act[i] | Hs[i] | taken[i]
                    for j in keys:
                        if j not in bl:
                            r_.append(a_); c_.append(U + kidx[j]); w_.append(1)
                for j in keys:
                    r_.append(U + kidx[j]); c_.append(snk); w_.append(units[j])
                Gm = sp.csr_matrix((np.array(w_, dtype=np.int32), (r_, c_)), shape=(U + K + 2, U + K + 2))
                ok = int(maximum_flow(Gm, src, snk).flow_value) == int(sum(z[i] for i in pool))
            if ok:
                cert = True; break
            conf = {j for j in units if j in col and any(j in act[i] for i in pool)} - S
            if not conf or len(S | conf) * it.shape[0] > max_arcs:
                break          # keep the valid outer bound from the last solve
            S |= conf
        out[name] = val; out[name + '_certified'] = cert
    return out
