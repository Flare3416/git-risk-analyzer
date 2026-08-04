import pandas as pd
import os

BUG_KEYWORDS = [
    "fix", "bug", "error", "issue", "crash", "defect",
    "patch", "fault", "failure", "resolve", "broken",
    "wrong", "incorrect", "problem", "corrupt",
    "prevent", "handle", "correct", "avoid", "address",
    "repair", "hotfix", "regression", "race condition",
    "null pointer", "overflow", "deadlock", "leak",
]

FALSE_POSITIVES = [
    "fix typo", "fix typos", "fix test", "fix tests", "fix testing",
    "fix formatting", "fix lint", "fix style",
    "fix docs", "fix readme", "fix doc", "fix documentation",
    "fix ci", "fix build", "fix import", "fix imports",
    "fix whitespace", "fix indentation", "fix docstring",
    "merge", "revert", "bump version", "update changelog",
    "fix requirements", "fix setup", "fix config", "fix .gitignore",
    "fix merge", "fix conflict", "fix rebase",
]


def is_real_bug_fix(msg: str) -> bool:
    msg = str(msg).lower()
    has_bug = any(kw in msg for kw in BUG_KEYWORDS)
    is_fp = any(fp in msg for fp in FALSE_POSITIVES)
    return has_bug and not is_fp


def label_commits(input_path: str, output_path: str = None, bug_lookback_commits: int = 1) -> pd.DataFrame:
    print(f"Labeling {input_path} ...")
    df = pd.read_csv(input_path)

    date_col = "date" if "date" in df.columns else "author_date"
    df[date_col] = pd.to_datetime(df[date_col], format="ISO8601", utc=True, errors="coerce")

    bad_dates = df[date_col].isna().sum()
    if bad_dates > 0:
        print(f"  ⚠️ Dropping {bad_dates} row(s) with bad dates")
        df = df.dropna(subset=[date_col])

    df = df.sort_values(date_col).reset_index(drop=True)
    df["is_bug_fix"] = df["msg"].apply(is_real_bug_fix).astype(int)
    df["is_buggy"] = 0

    import numpy as np
    buggy_indices = []
    path_col = "file_path" if "file_path" in df.columns else "file"
    # df is sorted by date_col, so group subsets will also be chronological
    for file_p, group in df.groupby(path_col):
        if pd.isna(file_p):
            continue
        # Find local indices in the group where it is a bug fix
        fix_positions = np.where(group["is_bug_fix"] == 1)[0]
        for pos in fix_positions:
            # We want to label bug_lookback_commits before the fix commit
            start_pos = max(0, pos - bug_lookback_commits)
            # Fetch the original dataframe indices corresponding to these group elements
            buggy_indices.extend(group.index[start_pos:pos].tolist())

    df.loc[buggy_indices, "is_buggy"] = 1

    total = len(df)
    bug_fixes = df["is_bug_fix"].sum()
    buggy = df["is_buggy"].sum()

    print(f"  Total rows        : {total:,}")
    print(f"  Bug-fix commits   : {bug_fixes:,}  ({bug_fixes/total:.2%})")
    print(f"  Bug-inducing rows : {buggy:,}  ({buggy/total:.2%})")

    if output_path is None:
        output_path = input_path.replace("_commits.csv", "_labeled.csv")
        if output_path == input_path:
            output_path = input_path.replace(".csv", "_labeled.csv")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved → {output_path}\n")
    return df


if __name__ == "__main__":
    csv_files = sorted(f for f in os.listdir("data") if f.endswith("_commits.csv"))
    if not csv_files:
        print("No *_commits.csv files found in data/")
    else:
        print(f"Found {len(csv_files)} commit CSV(s) to label\n")
        for csv_file in csv_files:
            label_commits(f"data/{csv_file}")