# What Unlinked Aggregates Still Prove

Research code accompanying IEEE BigData 2026 PSBD paper SP01232, *What Unlinked Aggregates Still Prove: Ranker Comparisons from Exact and Differentially Private Item Counts*.

The paper asks when aggregate item counts can determine which of two recommenders did better after some purchases can no longer be matched to the recommendations users received. The code includes the transportation bounds, certification checks, private confidence boxes and tiered item bins. It also includes the Tmall preparation recipe and a small example you can run without data or third-party Python packages.

## Reproduce the two-user example

Use Python 3.10 or later. From the repository root, run:

```sh
python3 -B demo.py
```

The script enumerates the assignments in the paper's example and reports the summed difference between rankers A and B. With no counts, the interval is `[-2, 2]`. Releasing one count for item `a` and one for `d` gives `[1, 1]` and establishes A's advantage. Releasing one count for `a` and one for `b` gives `[0, 1]` and leaves a tie possible.

To check the same two count cases with both transportation solvers, install the dependencies in a local environment and run:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -B demo.py --lp
```

## Code and experiments

| File | What it provides |
|---|---|
| `agg.py` | Paired Hit/Recall and NDCG contributions, per-user bounds, dense transportation formulation, reduced networks and certification, popularity-bin bounds, private confidence boxes, and the multi-item variant |
| `item_bounds.py` | Item-only bounds using exact counts without the constraint that each unlinked user has a single outcome |
| `segments.py` and `disclosure.py` | History-based and hash-based user grouping, with an exact check of how many outcomes are feasible for each user given group counts |
| `rankers.py` and `prepare_tmall.py` | Full-catalog Pop, ItemKNN, PureSVD, and EASE slates in the Tmall setup |
| `run_grid.py` and `run_bins.py` | Main-grid and tiered-release evaluation code, including the SHA-256 mask and noise-seed recipes used by these runners |
| `test_agg.py` | Synthetic cross-check of the reduced solver against exhaustive assignment |

The bounds operate on differences in summed contributions over the unlinked users. To compare a fixed cohort, add the known difference from linked users to each endpoint, then divide both endpoints by the cohort size. Two endpoints on the same side of zero establish the direction under exact feasible counts. With private counts, the direction is covered by the corresponding simultaneous confidence guarantee over the release noise. An uncertified reduced endpoint is an outer bound. The dense solver and exhaustive example are meant for small instances.

## Obtain the datasets from their providers

- **Beibei.** Doyeon Kim, *Beibei Dataset*, release 2. [Figshare dataset and splits](https://doi.org/10.6084/m9.figshare.25844296.v2). The provider distributes `beibei.zip` under CC BY 4.0. The paper uses the supplied splits. In the evaluation cohort, users had a training history and a test purchase. Histories use training and validation items and the rankers fit on the training split. A fixed keyed rule selected one designated test purchase per user. The historical Beibei cohort-builder relied on an upstream split reader and seed helper outside this release. To use the supplied runners for Beibei, provide locally reconstructed cohort arrays with the matching split and held-out rule. The archive from Figshare is the data source and does not contain those ready-made arrays.
- **Tmall.** *IJCAI-15 Repeat Buyers Prediction Dataset*. Obtain `data_format1.zip` from [Tianchi](https://tianchi.aliyun.com/dataset/42) under its access terms. To extract purchases, check the paper input signature and build local histories and slates, run:

  ```sh
  python3 -B extract_tmall.py /path/to/data_format1.zip inputs/tmall --verify-paper-input
  python3 -B prepare_tmall.py inputs/tmall/tmall_buy.csv.gz inputs/tmall.npz
  ```

  The first command requires the historical purchase signature. The reference extraction has 3,292,144 purchase rows and MD5 `0dffb6ca97c6aa8ff340afe5951a0447` for the uncompressed purchase CSV. Tmall preparation fits full-catalog rankers and can use substantial memory. The commands write local files that may contain user and item identifiers.

No raw data, derived user-level arrays, model caches, or frozen experiment files are distributed here. The publication tables describe the frozen runs. The code release does not recreate every table or all of the historical grid orchestration from a single command.

## Run targeted evaluations on prepared arrays

Supply the local `<dataset>.npz` in `inputs/`, or set `PSBD_ARRAYS` to its parent directory. The runners expect `users`, `pos` (one zero-based held-out item per row), `hist_len`, `n_items`, ragged histories split by `H_ptr` and `H_idx`, and the zero-based slate matrices `ease`, `knn`, `svd`, and `pop`. The arrays must follow the same user order.

For example, the following evaluates EASE against PureSVD with the top 100 items released separately and the remaining items binned by logged-history popularity, using the first two hash masks and linked fractions 0.5, 0.7, and 0.9. It computes exact-bin results and noisy-bin results at privacy budgets 1 and 4.

```sh
python3 -B run_bins.py beibei 100 1 2 ease_svd
```

The command writes `bins_beibei_ease_svd.jsonl`. Use a fresh output name for each invocation. To examine the 90% linked, privacy budget 4 case, select the two rows with `rate=0.9`, `eps=4.0`, `P=100`, and masks `hash0` and `hash1`. A positive lower endpoint in the `b` field indicates EASE is ahead for that evaluated cohort under the stated confidence guarantee.

`run_grid.py beibei 1` runs the first-pass exact and private main grid on prepared Beibei arrays using one worker. The runner does not apply historical refinement and correction passes. The paper also reports additional Tmall linkage masks, segment and repeated-release experiments, timing, and robustness checks. The released code provides the underlying methods rather than every launcher for those analyses.
