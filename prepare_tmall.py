"""Build local Tmall histories, held-out purchases, and four top-10 slates."""

import argparse
from pathlib import Path

import numpy as np
from rankers import csr, slates_for


def tmall(purchases, output, n_items=10000, min_hist=3):
    """Tmall IJCAI-15 purchase log (action_type = purchase). Catalog: the n_items most purchased items.
    For each user, the relevant item is one purchase on the user's last purchase day (keyed hash);
    the history is the set of catalog items purchased on earlier days."""
    import pandas as pd, hashlib
    d = pd.read_csv(purchases, header=None, names=['u', 'i', 't'], dtype={'t': str})
    d['t'] = d.t.str.zfill(4).astype(int)
    top = d.drop_duplicates(['u', 'i']).i.value_counts().index[:n_items]
    imap = {it: k for k, it in enumerate(top)}
    d = d[d.i.isin(imap)].copy(); d['j'] = d.i.map(imap)
    last = d.groupby('u').t.transform('max')
    fut = d[d.t == last]; past = d[d.t < last]
    hist = past.groupby('u').j.apply(lambda s: np.unique(s.values))
    cand = fut.groupby('u').j.apply(lambda s: np.unique(s.values))
    users, pos, H = [], [], []
    for u, h in hist.items():
        if len(h) < min_hist or u not in cand.index:
            continue
        c = np.setdiff1d(cand[u], h)
        if len(c) == 0:
            continue
        p = min(c, key=lambda j: (hashlib.sha256(f'tmall|{u}|{j}'.encode()).hexdigest(), j))
        users.append(u); pos.append(p); H.append(np.asarray(h, np.int64))
    nU = len(users); uids = list(users); users = np.arange(nU); pos = np.array(pos)
    X = csr([(u, j) for u in range(nU) for j in H[u]], nU, n_items)
    pop = np.asarray(X.sum(0)).ravel()
    S = slates_for(X, X, users, H, pop)
    np.savez(output, users=users, orig_uid=np.array(uids), pos=pos, hist_len=np.array([len(h) for h in H]), n_items=n_items,
             H_ptr=np.cumsum([0] + [len(h) for h in H]), H_idx=np.concatenate(H), **S)
    return users, pos, S

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("purchases", type=Path, help="Purchase CSV created by extract_tmall.py")
    parser.add_argument("output", type=Path, help="New NPZ destination")
    args = parser.parse_args()
    if args.output.suffix != ".npz" or args.output.exists():
        parser.error("output must be a new .npz file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    users, _, _ = tmall(args.purchases, args.output)
    print(f"Wrote {len(users):,} users to {args.output}")


if __name__ == "__main__":
    main()
