"""Same-information baseline: bounds that use the unlinked item counts but drop the one-outcome-per-user
constraint (each unit of item j goes to the most/least favourable admissible unlinked users, a user may be
reused across items). Compared with per-user bounds (drop the item rows) and the joint LP (both rows)."""
import numpy as np

def item_only(it, qq, pos, W, H, nI):
    itW, qW = it[W], qq[W]; c = np.bincount(pos[W], minlength=nI)
    hist = np.zeros(nI, np.int64)
    for i in W:
        hist[H[i]] += 1
    r, k = np.nonzero(qW != 0); jj = itW[r, k]; vv = qW[r, k]
    order = np.argsort(jj, kind='stable'); jj, vv = jj[order], vv[order]
    starts = np.searchsorted(jj, np.arange(nI)); ends = np.searchsorted(jj, np.arange(nI), side='right')
    U = L = 0.0; short = 0
    for j in np.flatnonzero(c > 0):
        v = vv[starts[j]:ends[j]]; n0 = len(W) - hist[j] - len(v)
        if n0 < 0: n0 = 0
        cj = int(c[j]); allv = np.concatenate([v, np.zeros(min(n0, cj))])
        if len(allv) < cj: short += 1; allv = np.concatenate([allv, np.zeros(cj - len(allv))])
        s = np.sort(allv); U += s[::-1][:cj].sum(); L += s[:cj].sum()
    return L, U, short
