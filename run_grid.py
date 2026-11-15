"""Main experiment grid. Writes one JSON line per (dataset, pair, metric, rate, mask[, dp draw])."""
import sys, json, time, hashlib, itertools, numpy as np
from multiprocessing import Pool
from pathlib import Path
import os
import agg
DATA = Path(os.environ.get('PSBD_ARRAYS', 'inputs'))
RANKERS = ('ease', 'knn', 'svd', 'pop')
PAIRS = list(itertools.combinations(RANKERS, 2))
RATES = (0.5, 0.7, 0.8, 0.9)
HASH_SEEDS = 10
EPS = (0.5, 1.0, 2.0, 4.0)
DP_SEEDS = 5          # hash masks that also receive DP draws
ALPHA = 0.05

def load(ds):
    z = np.load(DATA / f'{ds}.npz', allow_pickle=False)
    H = np.split(z['H_idx'], z['H_ptr'][1:-1])
    return z, H

def masks(users, hist_len, truth, rate):
    n = len(users); nv = int(np.floor(rate * n + 0.5)); out = []
    for s in range(HASH_SEEDS):
        key = np.array([hashlib.sha256(f'{s}|{u}'.encode()).digest()[:8] for u in users])
        order = np.argsort(key, kind='stable'); v = np.zeros(n, bool); v[order[:nv]] = True
        out.append((f'hash{s}', 'hash', v))
    for tag, vals in (('active', -hist_len), ('inactive', hist_len), ('oracle_low', truth), ('oracle_high', -truth)):
        order = np.lexsort((users, vals)); v = np.zeros(n, bool); v[order[:nv]] = True
        fam = 'activity' if tag in ('active', 'inactive') else 'oracle'
        out.append((tag, fam, v))
    return out

def decide(L, U):
    return 'A' if L > 1e-12 else ('B' if U < -1e-12 else 'abstain')

def task(args):
    ds, a, b, metric = args
    z, H = load(ds); users = z['users']; pos = z['pos']; n = len(users); nI = int(z['n_items'])
    it, qq = agg.contributions(z[a], z[b], metric)
    truth = agg.realized(it, qq, pos); D = truth.mean()
    all_counts = np.bincount(pos, minlength=nI).astype(float)
    lines = []
    for rate in RATES:
        for mname, fam, vis in masks(users, z['hist_len'], truth, rate):
            W = ~vis; known = truth[vis].sum(); nw = int(W.sum())
            cc = truth[vis].mean()
            l, u = agg.per_user_bounds(qq[W])
            rec = dict(ds=ds, A=a, B=b, metric=metric, rate=rate, mask=mname, family=fam, n=n, n_unlinked=nw,
                       D=D, cc=cc, gen=[(known - nw) / n, (known + nw) / n],
                       per=[(known + l.sum()) / n, (known + u.sum()) / n])
            cW = np.bincount(pos[W], minlength=nI).astype(float)
            HW = [H[i] for i in np.flatnonzero(W)]
            t0 = time.time()
            r = agg.aggregate_bounds(it[W], qq[W], cW, cW, nI, H=HW)
            rec['agg'] = [(known + r['L']) / n, (known + r['U']) / n]
            rec['agg_cert'] = bool(r['L_certified'] and r['U_certified']); rec['agg_sec'] = time.time() - t0
            if not rec['agg_cert'] and ds == 'ml':
                d = agg.exact_dense(it[W], qq[W], cW, cW, HW, nI)
                rec['agg_dense'] = [(known + d['L']) / n, (known + d['U']) / n]
            rec['dp'] = {}
            if fam == 'hash' and int(mname[4:]) < DP_SEEDS and metric == 'ndcg':
                linked = np.bincount(pos[vis], minlength=nI).astype(float)
                for eps in EPS:
                    rng = np.random.default_rng([int(hashlib.sha256(f'{ds}{a}{b}{rate}{mname}{eps}'.encode()).hexdigest()[:8], 16)])
                    lo, hi, t = agg.dp_box(all_counts, linked, eps, ALPHA, rng)
                    covered = bool(np.all(cW >= lo) and np.all(cW <= hi))
                    try:
                        rd = agg.aggregate_bounds(it[W], qq[W], lo, hi, nI, H=HW)
                        rec['dp'][str(eps)] = dict(b=[(known + rd['L']) / n, (known + rd['U']) / n], box_covers=covered,
                                                   cert=bool(rd['L_certified'] and rd['U_certified']), t=t)
                    except RuntimeError:
                        rec['dp'][str(eps)] = dict(b=None, box_covers=covered, t=t)
            lines.append(rec)
    return lines

if __name__ == '__main__':
    ds = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    tasks = [(ds, a, b, m) for (a, b) in PAIRS for m in ('ndcg', 'hit')]
    with Pool(procs) as p, open(f'results_{ds}.jsonl', 'x') as f:
        for lines in p.imap_unordered(task, tasks):
            for rec in lines:
                f.write(json.dumps(rec) + '\n')
            f.flush(); print(lines[0]['A'], lines[0]['B'], lines[0]['metric'], 'done', flush=True)
