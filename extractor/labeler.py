import pandas as pd
import os

BUG_KEYWORDS = [
    "fix", "bug", "error", "issue", "crash", "defect",
    "patch", "fault", "failure", "resolve", "broken",
    "wrong", "incorrect", "problem", "corrupt"
]

def label_commits(input_path: str, output_path: str = None) -> pd.DataFrame:
    print(f"Labeling {input_path} ...")
    df = pd.read_csv(input_path)

    # label each commit as bug fix or not
    df["is_bug_fix"] = df["msg"].str.lower().apply(
        lambda msg: int(any(kw in str(msg) for kw in BUG_KEYWORDS))
    )

    print(f"Total rows: {len(df)}")
    print(f"Bug fix commits: {df['is_bug_fix'].sum()}")
    print(f"Bug fix rate: {df['is_bug_fix'].mean():.2%}")

    if output_path is None:
        output_path = input_path  # overwrite in place

    df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}\n")
    return df


if __name__ == "__main__":
    # label all repo CSVs
    csv_files = [f for f in os.listdir("data") if f.endswith("_commits.csv")]
    
    for csv_file in csv_files:
        label_commits(f"data/{csv_file}")