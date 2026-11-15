"""Reconstruct the two-user example with exhaustive search or the released solver."""

import argparse
from collections import Counter
from itertools import product

CATALOG = "abcd"
SLATE_A = ("a", "a")
SLATE_B = ("b", "c")


def difference(choices):
    return sum(
        int(chosen == a) - int(chosen == b)
        for chosen, a, b in zip(choices, SLATE_A, SLATE_B)
    )


def enumerate_bounds(counts=None):
    values = [
        difference(choices)
        for choices in product(CATALOG, repeat=len(SLATE_A))
        if counts is None or Counter(choices) == counts
    ]
    return min(values), max(values)


def check_lp(counts, expected):
    import numpy as np

    import agg

    a = np.array([[CATALOG.index(item)] for item in SLATE_A])
    b = np.array([[CATALOG.index(item)] for item in SLATE_B])
    items, gains = agg.contributions(a, b, "hit")
    totals = np.array([counts[item] for item in CATALOG])
    histories = [np.array([], dtype=int) for _ in SLATE_A]
    for label, solve in (("dense", agg.exact_dense), ("reduced", agg.aggregate_bounds_exact)):
        if label == "dense":
            result = solve(items, gains, totals, totals, histories, len(CATALOG))
        else:
            result = solve(items, gains, totals, totals, len(CATALOG), histories)
            if not (result["L_certified"] and result["U_certified"]):
                raise RuntimeError("The reduced solver did not certify both toy endpoints")
        observed = (result["L"], result["U"])
        if not np.allclose(observed, expected, atol=1e-9, rtol=0):
            raise RuntimeError(f"{label} solver returned {observed}, expected {expected}")
        print(f"  {label} solver agrees with enumeration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lp", action="store_true", help="Check each count case against both LP implementations")
    args = parser.parse_args()
    for name, counts, expected in (
        ("No counts", None, (-2, 2)),
        ("One a and one d", Counter("ad"), (1, 1)),
        ("One a and one b", Counter("ab"), (0, 1)),
    ):
        observed = enumerate_bounds(counts)
        if observed != expected:
            raise RuntimeError(f"{name} returned {observed}, expected {expected}")
        print(f"{name}: [{observed[0]}, {observed[1]}]")
        if args.lp and counts is not None:
            check_lp(counts, expected)


if __name__ == "__main__":
    main()
