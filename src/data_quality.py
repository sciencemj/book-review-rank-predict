"""Detect data-quality issues and (optionally) gate on them (Stages 4 & 6).

Stage 4 (데이터 이해 · profiling): run WITHOUT --strict for a readable report —
per-column nulls / cardinality / numeric outliers / datetime ranges, plus WARN flags
(constant, all-null, high-cardinality columns).

Stage 6 (데이터 준비 · Data Preparation): run WITH --strict as a gate. HARD checks
(high null rate, full-row dups, key dups, constant/all-null columns, optional outlier
fraction) make it exit(2) on any violation, exit(0) when clean. Usage/IO errors exit(1).

Robust across shapes — numeric / categorical / datetime / boolean / object / mixed —
and degenerate frames (empty, single-row, all-null columns) never crash.

Usage:
    python data_quality.py data.csv                       # report mode (profiling)
    python data_quality.py data.csv --strict              # gate mode (exit 2 on fail)
    python data_quality.py data.parquet --key id,date --strict
    python data_quality.py data.csv --max-null 0.3 --fail-outliers 0.05 --json out.json

CSV encoding is tried utf-8 → cp949 → euc-kr (Korean public-data lore) and the working
one is reported; .parquet/.pq are read with pandas.read_parquet.

Requires: pandas, numpy  (`uv add pandas numpy`).
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

pdt = pd.api.types


def load(path):
    """Read CSV/Parquet. Returns (df, source). Raises on failure (-> exit 1)."""
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path), "parquet"
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError("could not decode as utf-8 / cp949 / euc-kr")


def numeric_stats(s, iqr_mult, z_thresh):
    """Summary stats + IQR/robust-z outlier counts for a numeric Series, or None."""
    try:
        a = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
        n = a.size
        if n == 0:
            return None
        q1, med, q3 = (float(np.percentile(a, p)) for p in (25, 50, 75))
        iqr = q3 - q1
        lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        iqr_out = int(np.count_nonzero((a < lo) | (a > hi)))
        mad = float(np.median(np.abs(a - med)))  # guard MAD==0 -> no z-outliers
        z_out = int(np.count_nonzero(np.abs(a - med) / (1.4826 * mad) > z_thresh)) if mad > 0 else 0
        return {
            "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "median": med, "std": float(a.std(ddof=1)) if n > 1 else 0.0, "q1": q1, "q3": q3,
            "iqr_outliers": iqr_out, "iqr_outlier_frac": iqr_out / n,
            "z_outliers": z_out, "z_outlier_frac": z_out / n,
        }
    except Exception:  # noqa: BLE001 - exotic numeric dtypes degrade to "no stats"
        return None


def datetime_info(s):
    """Best-effort min/max/span + trivial daily-gap note, or None. Never raises."""
    try:
        if pdt.is_datetime64_any_dtype(s):
            d = s.dropna()
        else:  # only attempt object cols that *look* date-ish on a cheap sample
            sample = s.dropna().astype(str).head(50)
            if sample.empty or sample.str.contains(r"[-/:]").mean() < 0.8:
                return None
            d = pd.to_datetime(s, errors="coerce")
            if d.notna().mean() < 0.8:
                return None
            d = d.dropna()
        if d.empty:
            return None
        span = int((d.max() - d.min()).days)
        info = {"min": str(d.min()), "max": str(d.max()), "span_days": span}
        norm = d.dt.normalize()  # gap check only for pure-date (midnight) series
        if (norm == d).all():
            present, expected = int(norm.nunique()), span + 1
            if 1 < present < expected:
                info["gap"] = f"{present}/{expected} daily points present"
        return info
    except Exception:  # noqa: BLE001 - parsing is opportunistic, never fatal
        return None


def profile(df, args):
    """Return (dataset dict, columns dict, key list). Raises KeyError on bad --key."""
    n_rows, n_cols = df.shape
    keys = [k.strip() for k in args.key.split(",") if k.strip()] if args.key else []
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"--key column(s) not found: {', '.join(missing)}")
    dataset = {
        "rows": int(n_rows), "cols": int(n_cols),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "full_row_duplicates": int(df.duplicated().sum()),
        "key": keys,
        "key_duplicates": int(df.duplicated(subset=keys).sum()) if keys else None,
    }
    cols = {}
    for name in df.columns:
        s = df[name]
        non_null = int(s.notna().sum())
        n_unique = int(s.nunique(dropna=True))
        is_num = pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s)
        kind = ("numeric" if is_num else "boolean" if pdt.is_bool_dtype(s)
                else "datetime" if pdt.is_datetime64_any_dtype(s)
                else "categorical" if isinstance(s.dtype, pd.CategoricalDtype) else "other")
        prof = {
            "dtype": str(s.dtype), "kind": kind, "non_null": non_null,
            "null_count": int(n_rows - non_null),
            "null_rate": (n_rows - non_null) / n_rows if n_rows else 0.0,
            "n_unique": n_unique, "unique_ratio": n_unique / n_rows if n_rows else 0.0,
            "examples": [str(v)[:30] for v in s.dropna().unique()[:3]],
        }
        if is_num:
            prof["numeric"] = numeric_stats(s, args.iqr, args.z)
        elif kind in ("datetime", "categorical", "other"):
            dt = datetime_info(s)
            if dt:
                prof["datetime"] = dt
                if kind == "other":
                    prof["kind"] = "datetime?"
        cols[str(name)] = prof
    return dataset, cols, keys


def _out_frac(nm, method):
    if method == "iqr":
        return nm["iqr_outlier_frac"]
    if method == "zscore":
        return nm["z_outlier_frac"]
    return max(nm["iqr_outlier_frac"], nm["z_outlier_frac"])


def evaluate(dataset, cols, keys, args):
    """Run HARD + WARN checks. Returns (checks list, verdict 'pass'|'fail')."""
    checks = []

    def add(name, level, passed, detail):
        checks.append({"name": name, "level": level, "passed": bool(passed),
                       "detail": detail or "ok"})

    bad_null = sorted((c, p["null_rate"]) for c, p in cols.items() if p["null_rate"] > args.max_null)
    add(f"null_rate <= {args.max_null:.2f}", "hard", not bad_null,
        "; ".join(f"{c} {r:.1%}" for c, r in bad_null))

    fd = dataset["full_row_duplicates"]
    add("no full-row duplicates", "hard", fd == 0, f"{fd} duplicate rows")

    if keys:
        kd = dataset["key_duplicates"]
        add(f"no duplicate keys ({','.join(keys)})", "hard", kd == 0, f"{kd} duplicate keys")

    consts = sorted(c for c, p in cols.items() if p["n_unique"] <= 1)  # <=1 covers all-null (0)
    add("no constant / all-null columns", "hard", not consts, "; ".join(consts))

    flagged = sorted((c, _out_frac(p["numeric"], args.outlier_method))
                     for c, p in cols.items() if p.get("numeric"))
    flagged = [(c, f) for c, f in flagged if f > 0]
    if args.fail_outliers is not None:  # HARD only when an explicit threshold is given
        bad = [(c, f) for c, f in flagged if f > args.fail_outliers]
        add(f"outlier fraction <= {args.fail_outliers:.2f}", "hard", not bad,
            "; ".join(f"{c} {f:.1%}" for c, f in bad))
    else:
        add("numeric outliers (warn)", "warn", not flagged,
            "; ".join(f"{c} {f:.1%}" for c, f in flagged))

    hicard = sorted(c for c, p in cols.items()
                    if p["n_unique"] > 1 and p["unique_ratio"] > 0.9 and c not in keys)
    add("high-cardinality columns (warn)", "warn", not hicard,
        "; ".join(f"{c} {cols[c]['unique_ratio']:.1%}" for c in hicard))

    gaps = sorted((c, p["datetime"]["gap"]) for c, p in cols.items()
                  if p.get("datetime", {}).get("gap"))
    add("datetime gaps (warn)", "warn", not gaps, "; ".join(f"{c}: {g}" for c, g in gaps))

    verdict = "fail" if any(c["level"] == "hard" and not c["passed"] for c in checks) else "pass"
    return checks, verdict


def fmt_bytes(n):
    x = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:.1f} {u}"
        x /= 1024
    return f"{x:.1f} TB"


def table(headers, rows):
    """Render an aligned, color-free text table (deterministic)."""
    w = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))
    out = ["  ".join(h.ljust(w[i]) for i, h in enumerate(headers)),
           "  ".join("-" * w[i] for i in range(len(headers)))]
    out += ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join(out)


def print_report(path, dataset, cols, checks, verdict):
    print(f"DATA QUALITY  {path}")
    print(f"source: {dataset['encoding']} | rows: {dataset['rows']} | "
          f"cols: {dataset['cols']} | memory: {fmt_bytes(dataset['memory_bytes'])}")
    dl = f"full-row duplicates: {dataset['full_row_duplicates']}"
    if dataset["key_duplicates"] is not None:
        dl += f" | key duplicates ({','.join(dataset['key'])}): {dataset['key_duplicates']}"
    print(dl + "\n")

    names = sorted(cols)
    print("COLUMNS")
    print(table(["column", "dtype", "non_null", "null%", "n_uniq", "uniq%", "examples"],
                [[n[:28], cols[n]["dtype"], str(cols[n]["non_null"]),
                  f"{cols[n]['null_rate']:.1%}", str(cols[n]["n_unique"]),
                  f"{cols[n]['unique_ratio']:.1%}", ", ".join(cols[n]["examples"])[:40]]
                 for n in names]))

    nrows = [[n[:28], f"{m['min']:.4g}", f"{m['max']:.4g}", f"{m['mean']:.4g}",
              f"{m['median']:.4g}", f"{m['std']:.4g}",
              f"{m['iqr_outliers']} ({m['iqr_outlier_frac']:.1%})",
              f"{m['z_outliers']} ({m['z_outlier_frac']:.1%})"]
             for n in names for m in [cols[n].get("numeric")] if m]
    if nrows:
        print("\nNUMERIC")
        print(table(["column", "min", "max", "mean", "median", "std", "iqr_out", "z_out"], nrows))

    drows = [[n[:28], d["min"][:19], d["max"][:19], str(d["span_days"])]
             for n in names for d in [cols[n].get("datetime")] if d]
    if drows:
        print("\nDATETIME")
        print(table(["column", "min", "max", "span_days"], drows))

    print("\nFLAGS / WARN")
    issues = [c for c in checks if not c["passed"]]
    for c in issues:
        print(f"  [{'HARD' if c['level'] == 'hard' else 'WARN'}] {c['name']}: {c['detail']}")
    if not issues:
        print("  (none)")
    print(f"\nVERDICT: {verdict.upper()}")


def print_strict(checks, verdict):
    print("STRICT GATE (HARD checks)")
    for c in checks:
        if c["level"] == "hard":
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}  ({c['detail']})")
    print(f"VERDICT: {verdict.upper()}")


def jsonable(obj):
    """Recursively coerce numpy scalars to native types and NaN/inf to None."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    ap = argparse.ArgumentParser(
        description="Data-quality profiling (Stage 4) and strict gate (Stage 6).")
    ap.add_argument("path", help="CSV or Parquet (.parquet/.pq) file")
    ap.add_argument("--strict", action="store_true",
                    help="gate mode: exit 2 on any HARD violation, else exit 0")
    ap.add_argument("--key", help="key column(s) for duplicate-key check, comma-separated")
    ap.add_argument("--max-null", type=float, default=0.40,
                    help="max per-column null rate; above this FAILS in strict (default 0.40)")
    ap.add_argument("--outlier-method", choices=("iqr", "zscore", "both"), default="both")
    ap.add_argument("--iqr", type=float, default=1.5, help="IQR multiplier (default 1.5)")
    ap.add_argument("--z", type=float, default=3.5, help="robust-z (MAD) threshold (default 3.5)")
    ap.add_argument("--fail-outliers", type=float, default=None,
                    help="strict: FAIL a numeric col if its outlier fraction exceeds this")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="also write the structured report as JSON to this path")
    args = ap.parse_args()

    try:
        df, enc = load(args.path)
    except FileNotFoundError:
        print(f"file not found: {args.path}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - IO / parse errors map to usage exit
        print(f"could not read {args.path}: {type(e).__name__}: {e}")
        sys.exit(1)

    try:
        dataset, cols, keys = profile(df, args)
    except KeyError as e:
        print(e.args[0])
        sys.exit(1)

    dataset["encoding"] = enc
    checks, verdict = evaluate(dataset, cols, keys, args)

    if args.json_path:
        report = {"dataset": dataset, "columns": cols, "checks": checks, "verdict": verdict}
        try:
            with open(args.json_path, "w", encoding="utf-8") as f:
                json.dump(jsonable(report), f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"could not write json {args.json_path}: {e}")
            sys.exit(1)

    if args.strict:
        print_strict(checks, verdict)
        sys.exit(0 if verdict == "pass" else 2)
    print_report(args.path, dataset, cols, checks, verdict)
    sys.exit(0)


if __name__ == "__main__":
    main()
