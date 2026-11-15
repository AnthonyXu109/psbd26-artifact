import itertools, numpy as np, agg

def brute(it, qq, H, lo, hi, m, n):
    best = [np.inf, -np.inf]
    cand = [[j for j in range(m) if j not in set(H[i])] for i in range(n)]
    for s in itertools.product(*cand):
        c = np.bincount(np.array(s), minlength=m)
        if np.all(c >= lo) and np.all(c <= hi):
            v = sum(qq[i][it[i] == s[i]].sum() for i in range(n))
            best[0] = min(best[0], v); best[1] = max(best[1], v)
    return best

def run(trials=3000, seed=0):
    rng = np.random.default_rng(seed)
    stats = dict(total=0, certified=0, mismatch=0, invalid=0, infeasible=0)
    for t in range(trials):
        m = int(rng.integers(4, 8)); n = int(rng.integers(1, 6)); k = int(rng.integers(1, 3))
        H = [rng.choice(m, size=int(rng.integers(0, 2)), replace=False) for _ in range(n)]
        SA = np.zeros((n, k), int); SB = np.zeros((n, k), int)
        for i in range(n):
            pool = [j for j in range(m) if j not in set(H[i])]
            SA[i] = rng.choice(pool, k, replace=False); SB[i] = rng.choice(pool, k, replace=False)
        metric = 'ndcg' if rng.random() < .5 else 'hit'
        it, qq = agg.contributions(SA, SB, metric)
        # true relevant items -> counts; exact or box
        s = np.array([rng.choice([j for j in range(m) if j not in set(H[i])]) for i in range(n)])
        c = np.bincount(s, minlength=m).astype(float)
        if rng.random() < .5:
            lo, hi = c.copy(), c.copy()
        else:
            w = rng.integers(0, 3, size=m)
            lo = np.maximum(0, c - rng.integers(0, 2, size=m)).astype(float); hi = (c + w).astype(float)
        bL, bU = brute(it, qq, H, lo, hi, m, n)
        stats['total'] += 1
        res = agg.aggregate_bounds(it, qq, lo, hi, m, H=H)
        if res['L'] > bL + 1e-9 or res['U'] < bU - 1e-9:
            stats['invalid'] += 1
        if res['L_certified'] and res['U_certified']:
            stats['certified'] += 1
            if abs(res['L'] - bL) > 1e-9 or abs(res['U'] - bU) > 1e-9:
                stats['mismatch'] += 1
                print('MISMATCH', t, res, bL, bU)
    return stats

if __name__ == '__main__':
    print(run())
