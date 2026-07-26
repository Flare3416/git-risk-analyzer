import pandas as pd
import os

def build_dataset(data_dir: str = "data", output_path: str = "data/features.csv", single_repo_csv: str = None) -> pd.DataFrame:
    all_dfs = []

    if single_repo_csv:
        df = pd.read_csv(single_repo_csv)
        df["repo"] = os.path.basename(single_repo_csv).replace("_commits.csv", "")
        all_dfs.append(df)
    else:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith("_commits.csv")]
        for csv_file in csv_files:
            df = pd.read_csv(f"{data_dir}/{csv_file}")
            df["repo"] = csv_file.replace("_commits.csv", "")
            all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date")

    grouped = df.groupby(["repo", "file_path"])

    records = []
    for (repo, file_path), group in grouped:
        bug_commits   = group["is_bug_fix"].sum()
        total_commits = len(group)
        last_date     = group["date"].max()
        first_date    = group["date"].min()

        records.append({
            "repo":                   repo,
            "file_path":              file_path,
            "total_commits":          total_commits,
            "commits_last_30d":       len(group[group["date"] >= last_date - pd.Timedelta(days=30)]),
            "commits_last_90d":       len(group[group["date"] >= last_date - pd.Timedelta(days=90)]),
            "days_since_last_change": (pd.Timestamp.now(tz="UTC") - last_date).days,
            "unique_authors":         group["author"].nunique(),
            "top_owner_pct":          group["author"].value_counts().iloc[0] / total_commits,
            "avg_lines_added":        group["lines_added"].mean(),
            "avg_lines_deleted":      group["lines_deleted"].mean(),
            "max_lines_changed":      (group["lines_added"] + group["lines_deleted"]).max(),
            "bug_fix_count":          bug_commits,
            "bug_fix_rate":           bug_commits / total_commits,
            "days_since_last_bugfix": (pd.Timestamp.now(tz="UTC") - group[group["is_bug_fix"] == 1]["date"].max()).days if bug_commits > 0 else 9999,
            "file_age_days":          (last_date - first_date).days,
            "avg_nloc":               group["nloc"].mean(),
            "is_buggy":               int(bug_commits > 0),
        })

    result = pd.DataFrame(records)
    result.to_csv(output_path, index=False)
    print(f"Dataset built → {output_path}")
    print(f"Total files: {len(result)}")
    print(f"Buggy files: {result['is_buggy'].sum()} ({result['is_buggy'].mean():.2%})")
    return result


if __name__ == "__main__":
    build_dataset()