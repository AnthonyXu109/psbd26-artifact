"""Popularity-tiered item binning: top-P items released individually, the tail merged
into T bins of roughly equal public (historical) popularity."""
import sys, json, time, hashlib, itertools, numpy as np, agg
from run_grid import load, masks
T_TAIL = 8
def make_groups(pop, P, T=T_TAIL):
    m = len(pop); order = np.argsort(-pop, kind='stable'); group = np.empty(m, np.int64)
    head = order[:P]; tail = order[P:]
    group[head] = np.arange(len(head))
    if len(tail):
        share = np.cumsum(pop[tail] + 1e-9) / (pop[tail] + 1e-9).sum()
        group[tail] = len(head) + np.minimum((share * T).astype(int), T - 1)
    _, group = np.unique(group, return_inverse=True)
    return group
def main(ds, Ps, pairsel, nmask, shard):
    z, H = load(ds); users = z['users']; pos = z['pos']; n = len(users); nI = int(z['n_items'])
    pop = np.bincount(z['H_idx'], minlength=nI).astype(float)
    pairs = list(itertools.combinations(('ease', 'knn', 'svd', 'pop'), 2))
    pairs = [pairs[i] for i in pairsel]
    out = open(f'bins_{ds}_{shard}.jsonl', 'x')
    for a, b in pairs:
        it, qq = agg.contributions(z[a], z[b], 'ndcg'); truth = agg.realized(it, qq, pos); D = truth.mean()
        for rate in (0.5, 0.7, 0.9):
            for mname, fam, vis in masks(users, z['hist_len'], truth, rate)[:nmask]:
                W = np.flatnonzero(~vis); known = truth[vis].sum(); HW = [H[i] for i in W]
                l, u = agg.per_user_bounds(qq[W])
                for P in Ps:
                    group = make_groups(pop, P); Gn = group.max() + 1
                    cW = np.bincount(group[pos[W]], minlength=Gn).astype(float)
                    call = np.bincount(group[pos], minlength=Gn).astype(float)
                    cV = call - cW
                    for eps in (None, 4.0, 1.0):
                        if eps is None:
                            lo = hi = cW; cover = True
                        else:
                            rng = np.random.default_rng([int(hashlib.sha256(f'bins{ds}{a}{b}{rate}{mname}{P}{eps}'.encode()).hexdigest()[:8], 16)])
                            bsc = 2.0 / eps; t = bsc * np.log(Gn / 0.05)
                            noisy = call + rng.laplace(0, bsc, size=Gn)
                            lo = np.maximum(0, np.ceil(noisy - t - cV - 1e-9)); hi = np.maximum(0, np.floor(noisy + t - cV + 1e-9))
                            cover = bool(np.all((cW >= lo) & (cW <= hi)))
                        t0 = time.time()
                        try:
                            r = agg.aggregate_bounds_grouped(it[W], qq[W], group, lo, hi, HW)
                            bb = [(known + r['L']) / n, (known + r['U']) / n]; cert = bool(r['L_certified'] and r['U_certified'])
                        except RuntimeError:
                            bb = None; cert = False
                        rec = dict(ds=ds, A=a, B=b, rate=rate, mask=mname, P=P, M=int(Gn), eps=eps, D=D, b=bb, cert=cert, box_covers=cover,
                                   per=[(known + l.sum()) / n, (known + u.sum()) / n], sec=time.time() - t0)
                        out.write(json.dumps(rec) + '\n'); out.flush()
        print(a, b, 'done', flush=True)
if __name__ == '__main__':
    ds = sys.argv[1]; Ps = [int(x) for x in sys.argv[2].split(',')]; pairsel = [int(x) for x in sys.argv[3].split(',')]
    main(ds, Ps, pairsel, int(sys.argv[4]), sys.argv[5])
