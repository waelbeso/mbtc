#!/usr/bin/env python3
# Print-only, universal CSV stats for any headered CSV.
# Python 3.8/3.9 compatible. No files are written.


'''
# try full load first
python print_csv_stats.py --csv cve_custom_merged_descriptions.csv

# if your CSV is very large or still malformed, use chunked mode (recommended):
python print_csv_stats.py --csv cve_custom_merged_descriptions.csv --chunksize 200000

'''
#!/usr/bin/env python3
# Print-only, universal CSV stats for any headered CSV.
# Python 3.8/3.9 compatible. No files are written.

import argparse, sys
from collections import Counter
import pandas as pd
import numpy as np

EMPTY = {"", "nan", "none", "<na>", "<NA>"}

def is_empty_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.lower()
    return s.isna() | s2.isin({x.lower() for x in EMPTY})

def robust_read_csv(path: str, chunksize=None):
    """Try C engine first; fall back to Python engine w/ bad-line skipping (new & old pandas)."""
    # 1) fast path
    try:
        return pd.read_csv(path, dtype=str, low_memory=False, chunksize=chunksize)
    except Exception:
        pass
    # 2) python engine (new pandas)
    try:
        return pd.read_csv(path, dtype=str, engine="python",
                           on_bad_lines="skip", chunksize=chunksize)
    except TypeError:
        # 3) python engine (older pandas)
        return pd.read_csv(path, dtype=str, engine="python",
                           error_bad_lines=False, warn_bad_lines=True,
                           chunksize=chunksize)

def infer_col_type(series: pd.Series) -> str:
    # datetime?
    dt = pd.to_datetime(series, errors="coerce", utc=True, infer_datetime_format=True)
    if dt.notna().sum() >= max(5, int(0.05 * len(series))):
        return "datetime"
    # numeric?
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().sum() >= max(5, int(0.05 * len(series))):
        return "numeric"
    return "text"

def print_full(df: pd.DataFrame, top_k: int):
    total_rows = len(df)
    cols = list(df.columns)

    print(f"Total rows: {total_rows}")
    print(f"Total columns: {len(cols)}")
    print("Columns:", cols, "\n")

    print("== Emptiness per column ==")
    for c in cols:
        empty_count = int(is_empty_series(df[c]).sum())
        non_empty = total_rows - empty_count
        pct = (empty_count / total_rows * 100) if total_rows else 0.0
        print(f"{c:>20}  empty={empty_count:>8}  non_empty={non_empty:>8}  empty%={pct:6.2f}")
    print()

    print("== Distinct values per column ==")
    for c in cols:
        uniq = df[c].nunique(dropna=True)
        print(f"{c:>20}  unique={uniq}")
    print()
'''
    print("== Type-aware stats ==")
    for c in cols:
        s = df[c]
        ctype = infer_col_type(s)
        print(f"[{c}] inferred_type={ctype}")
        if ctype == "numeric":
            num = pd.to_numeric(s, errors="coerce")
            nn = num.dropna()
            if len(nn):
                print(f"  count={len(nn)}  min={nn.min()}  max={nn.max()}  "
                      f"mean={nn.mean():.4f}  median={nn.median():.4f}  std={nn.std():.4f}")
            else:
                print("  (no numeric values)")
        elif ctype == "datetime":
            dt = pd.to_datetime(s, errors="coerce", utc=True, infer_datetime_format=True)
            nd = dt.dropna()
            if len(nd):
                print(f"  count={len(nd)}  min={nd.min()}  max={nd.max()}")
            else:
                print("  (no datetime-like values)")
        else:
            lens = s.fillna("").astype(str).str.len()
            if len(lens):
                print(f"  mean_len={lens.mean():.2f}  median_len={lens.median():.2f}  "
                      f"p90_len={lens.quantile(0.90):.0f}  p99_len={lens.quantile(0.99):.0f}  max_len={lens.max()}")
            else:
                print("  (no text)")
    print()

    print("== Top values per column ==")
    for c in cols:
        vc = df[c].astype(str).str.strip()
        vc = vc[~is_empty_series(df[c])]
        top = vc.value_counts().head(top_k)
        print(f"[{c}] top {min(top_k, len(top))}:")
        if total_rows:
            for val, cnt in top.items():
                pct = cnt / total_rows * 100
                show = (val[:80] + "…") if len(val) > 80 else val
                print(f"  {cnt:>8} ({pct:5.2f}%)  {show}")
        else:
            print("  (no rows)")
        print()
'''
def print_chunked(path: str, top_k: int, chunksize: int):
    it = robust_read_csv(path, chunksize=chunksize)
    it_iter = iter(it)
    first = next(it_iter)
    cols = list(first.columns)
    total_rows = 0
    empty_counts = {c: 0 for c in cols}
    top_counters = {c: Counter() for c in cols}
    DISTINCT_CAP = 2_000_000
    distinct_sets = {c: set() for c in cols}
    overflowed = {c: False for c in cols}

    def process(df: pd.DataFrame):
        nonlocal total_rows
        n = len(df)
        total_rows += n
        for c in cols:
            em = int(is_empty_series(df[c]).sum())
            empty_counts[c] += em
            vals = df[c].dropna().astype(str).str.strip()
            vals = vals[~is_empty_series(df[c])]
            top_counters[c].update(vals)
            if not overflowed[c]:
                s = distinct_sets[c]
                for v in vals:
                    s.add(v)
                    if len(s) >= DISTINCT_CAP:
                        overflowed[c] = True

    process(first)
    for chunk in it_iter:
        process(chunk)

    print(f"Total rows: {total_rows}")
    print(f"Total columns: {len(cols)}")
    print("Columns:", cols, "\n")

    print("== Emptiness per column ==")
    for c in cols:
        empty_count = empty_counts[c]
        non_empty = total_rows - empty_count
        pct = (empty_count / total_rows * 100) if total_rows else 0.0
        print(f"{c:>20}  empty={empty_count:>8}  non_empty={non_empty:>8}  empty%={pct:6.2f}")
    print()

    print("== Distinct values per column (approx if capped) ==")
    for c in cols:
        cnt = len(distinct_sets[c])
        suffix = " (capped)" if overflowed[c] else ""
        print(f"{c:>20}  unique={cnt}{suffix}")
    print()

    print("== Top values per column ==")
    for c in cols:
        items = top_counters[c].most_common(top_k)
        print(f"[{c}] top {min(top_k, len(items))}:")
        for val, cnt in items:
            pct = cnt / total_rows * 100 if total_rows else 0.0
            show = (val[:80] + "…") if len(val) > 80 else val
            print(f"  {cnt:>8} ({pct:5.2f}%)  {show}")
        print()

def main():
    ap = argparse.ArgumentParser(description="Print stats for ANY CSV (no files written).")
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--top-k", type=int, default=30, help="Top-K values per column")
    ap.add_argument("--chunksize", type=int, default=0, help="If >0, process in chunks (memory-safe)")
    args = ap.parse_args()

    if args.chunksize and args.chunksize > 0:
        print_chunked(args.csv, top_k=args.top_k, chunksize=args.chunksize)
    else:
        df_or_iter = robust_read_csv(args.csv, chunksize=None)
        # normalize iterator if returned
        if isinstance(df_or_iter, pd.io.parsers.TextFileReader):
            df = next(iter(df_or_iter))
        else:
            df = df_or_iter
        print_full(df, top_k=args.top_k)

if __name__ == "__main__":
    main()
