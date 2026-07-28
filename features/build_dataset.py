import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone

FEATURES = [
    "total_commits",
    "commits_last_30d",
    "commits_last_90d",
    "days_since_last_change",
    "unique_authors",
    "top_owner_pct",
    "avg_lines_added",
    "avg_lines_deleted",
    "max_lines_changed",
    "file_age_days",
    "avg_nloc",
]

# NEW: only count bugs within this many days of the reference date
BUG_LOOKBACK_DAYS = 90


def build_dataset(
    data_dir: str = "data",
    output_path: str = "data/features.csv",
    single_repo_csv: str = None,
    reference_date: pd.Timestamp = None,
) -> pd.DataFrame:
    # ── 1. discover files ──
    files_to_load = []
    if single_repo_csv:
        labeled = single_repo_csv.replace("_commits.csv", "_labeled.csv")
        if os.path.exists(labeled):
            files_to_load.append(labeled)
        else:
            files_to_load.append(single_repo_csv)
    else:
        for f in sorted(os.listdir(data_dir)):
            if f.endswith("_labeled.csv"):
                files_to_load.append(os.path.join(data_dir, f))
        if not files_to_load:
            print("WARNING: No *_labeled.csv found. Falling back to *_commits.csv")
            for f in sorted(os.listdir(data_dir)):
                if f.endswith("_commits.csv"):
                    files_to_load.append(os.path.join(data_dir, f))

    if not files_to_load:
        raise FileNotFoundError(f"No commit CSVs found in {data_dir}")

    # ── 2. load & concat ──
    all_dfs = []
    for path in files_to_load:
        df = pd.read_csv(path)
        repo_name = os.path.basename(path).replace("_labeled.csv", "").replace("_commits.csv", "")
        df["repo"] = repo_name
        all_dfs.append(df)
        print(f"  Loaded {len(df):,} rows from {os.path.basename(path)}")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(df):,}")

    # ── 3. normalize columns ──
    date_col = "author_date" if "author_date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], format="ISO8601", utc=True, errors="coerce")

    bad_dates = df[date_col].isna().sum()
    if bad_dates:
        print(f"  Dropping {bad_dates:,} row(s) with bad dates")
        df = df.dropna(subset=[date_col])

    if "is_buggy" not in df.columns:
        print("  WARNING: 'is_buggy' not found — falling back to 'is_bug_fix'")
        df["is_buggy"] = df.get("is_bug_fix", 0)

    for col in ["lines_added", "lines_deleted", "lines_changed", "nloc"]:
        if col not in df.columns:
            df[col] = 0

    # ── 4. reference date ──
    if reference_date is None:
        reference_date = df[date_col].max()
    print(f"  Reference date: {reference_date}")
    print(f"  Bug lookback  : {BUG_LOOKBACK_DAYS} days")

    # ── 5. pre-compute days before ref ──
    df["days_before_ref"] = (reference_date - df[date_col]).dt.days

    # ── 6. aggregate per file ──
    print("  Aggregating per file ...")
    g = df.groupby(["repo", "file_path"])

    # Basic counts
    agg = g[date_col].agg(["min", "max", "count"]).rename(columns={
        "min": "first_commit_date",
        "max": "last_commit_date",
        "count": "total_commits",
    })

    # Recency windows
    last_date_per_file = g[date_col].max().reset_index(name="last_date")
    df_merged = df.merge(last_date_per_file, on=["repo", "file_path"])
    df_merged["days_since_last"] = (reference_date - df_merged["last_date"]).dt.days

    commits_7d = df_merged[df_merged["days_before_ref"] <= 7].groupby(["repo", "file_path"]).size()
    commits_30d = df_merged[df_merged["days_before_ref"] <= 30].groupby(["repo", "file_path"]).size()
    commits_90d = df_merged[df_merged["days_before_ref"] <= 90].groupby(["repo", "file_path"]).size()

    # Authors
    author_counts = g["author"].nunique().rename("unique_authors")
    top_owner = g["author"].apply(
        lambda s: s.value_counts().iloc[0] / len(s) if len(s) > 0 else 0
    ).rename("top_owner_pct")

    # Line stats
    line_stats = g[["lines_added", "lines_deleted", "lines_changed"]].agg({
        "lines_added": ["mean", "std"],
        "lines_deleted": "mean",
        "lines_changed": ["mean", "max", "std"],
    })
    line_stats.columns = [
        "avg_lines_added", "std_lines_added", "avg_lines_deleted",
        "avg_lines_changed", "max_lines_changed", "std_lines_changed"
    ]
    line_stats["std_lines_added"] = line_stats["std_lines_added"].fillna(0)
    line_stats["std_lines_changed"] = line_stats["std_lines_changed"].fillna(0)

    # NLOC
    nloc_stats = g["nloc"].mean().rename("avg_nloc")

    # ── TARGET FIX: only bugs within lookback window ──
    recent_df = df[df["days_before_ref"] <= BUG_LOOKBACK_DAYS]
    is_buggy = recent_df.groupby(["repo", "file_path"])["is_buggy"].max()
    # Reindex to include all files (files with no recent bugs → 0)
    is_buggy = is_buggy.reindex(g.groups.keys(), fill_value=0).rename("is_buggy")

    # Bug-fix stats (all-time, for context)
    bug_fix_count = g["is_bug_fix"].sum().rename("bug_fix_count") if "is_bug_fix" in df.columns else pd.Series(0, index=g.groups.keys(), name="bug_fix_count")

    # ── 7. merge everything ──
    result = agg.copy()
    result = result.join(commits_7d.rename("commits_last_7d")).fillna({"commits_last_7d": 0})
    result = result.join(commits_30d.rename("commits_last_30d")).fillna({"commits_last_30d": 0})
    result = result.join(commits_90d.rename("commits_last_90d")).fillna({"commits_last_90d": 0})
    result = result.join(author_counts)
    result = result.join(top_owner)
    result = result.join(line_stats)
    result = result.join(nloc_stats)
    result = result.join(is_buggy)
    result = result.join(bug_fix_count)

    result["days_since_last_change"] = (reference_date - result["last_commit_date"]).dt.days
    result["file_age_days"] = (result["last_commit_date"] - result["first_commit_date"]).dt.days
    result["bug_fix_rate"] = result["bug_fix_count"] / result["total_commits"]

    bug_fix_dates = df[df.get("is_bug_fix", 0) == 1].groupby(["repo", "file_path"])[date_col].max()
    result["days_since_last_bugfix"] = (reference_date - bug_fix_dates).dt.days
    result["days_since_last_bugfix"] = result["days_since_last_bugfix"].fillna(9999)

    result = result.reset_index()

    # ── 8. per-repo stats ──
    print("\nPer-repo bug rates (last {} days):".format(BUG_LOOKBACK_DAYS))
    for repo, group in result.groupby("repo"):
        rate = group["is_buggy"].mean()
        print(f"  {repo:20s} : {rate:.2%} ({group['is_buggy'].sum():,}/{len(group):,})")

    # ── 9. save ──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"\nDataset built → {output_path}")
    print(f"  Total files : {len(result):,}")
    print(f"  Buggy files : {result['is_buggy'].sum():,} ({result['is_buggy'].mean():.2%})")
    print(f"  Avg commits/file: {result['total_commits'].mean():.1f}")
    print(f"  Features    : {len(result.columns)} columns")

    return result


if __name__ == "__main__":
    build_dataset()