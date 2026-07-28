import os
import shutil
import subprocess
import pandas as pd
from tqdm import tqdm

SKIP_EXTS = {
    ".md", ".rst", ".txt", ".yaml", ".yml", ".json",
    ".toml", ".cfg", ".ini", ".lock", ".gitignore",
    ".gitattributes", ".dockerignore", ".editorconfig",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".pdf", ".ipynb", ".csv", ".xml", ".html", ".css",
}

def _should_skip(filename: str | None) -> bool:
    if not filename:
        return True
    ext = os.path.splitext(filename)[1].lower()
    return ext in SKIP_EXTS


def mine_commits(
    repo_path: str,
    output_path: str = "data/raw_commits.csv",
    keep_repo: bool = True,
) -> pd.DataFrame:
    print(f"Fast-mining commits from {repo_path} ...")

    count_cmd = ["git", "-C", repo_path, "rev-list", "--all", "--count"]
    count_res = subprocess.run(count_cmd, capture_output=True, text=True, check=True)
    expected_commits = int(count_res.stdout.strip())
    print(f"  Repo has {expected_commits:,} commits")

    # Use null-byte (%x00) delimiter — impossible to appear in real data
    cmd = [
        "git", "--no-pager", "-C", repo_path,
        "log", "--all", "--numstat", "--no-renames",
        "--pretty=tformat:COMMIT%n%H%x00%an%x00%cn%x00%ad%x00%cd%x00%P%x00%s",
        "--date=iso-strict",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")

    if result.returncode != 0:
        raise RuntimeError(f"git log failed: {result.stderr[:500]}")

    raw = result.stdout.strip().split("\n")
    print(f"  Git returned {len(raw):,} lines")

    records = []
    commit_meta = None
    parsed_commits = set()

    for line in tqdm(raw, desc="Parsing", unit="line"):
        if line == "COMMIT":
            continue

        # Metadata line: split on null byte (chr(0))
        if "\x00" in line and not line.startswith(("\t", " ")):
            parts = line.split("\x00", 6)
            if len(parts) == 7 and len(parts[0]) == 40:
                parents = parts[5].strip().split()
                commit_meta = {
                    "commit_hash":    parts[0],
                    "author":         parts[1],
                    "committer":      parts[2],
                    "author_date":    parts[3],
                    "committer_date": parts[4],
                    "msg":            parts[6],
                    "is_merge":       len(parents) > 1,
                    "parents_count":  len(parents),
                }
                parsed_commits.add(parts[0])
                continue

        # Numstat line: added\tdeleted\tfilepath
        if "\t" in line and commit_meta is not None:
            added_str, deleted_str, filepath = line.split("\t", 2)

            if added_str == "-" or deleted_str == "-":
                continue
            if _should_skip(filepath):
                continue

            records.append({
                **commit_meta,
                "file":          os.path.basename(filepath),
                "file_path":     filepath,
                "lines_added":   int(added_str),
                "lines_deleted": int(deleted_str),
                "lines_changed": int(added_str) + int(deleted_str),
            })

    df = pd.DataFrame(records)

    actual_commits = len(parsed_commits)
    print(f"  Parsed {actual_commits:,} commits with metadata")
    print(f"  Extracted {len(df):,} file-change rows")
    print(f"  Unique files: {df['file_path'].nunique():,}")

    if actual_commits < expected_commits * 0.5:
        print(f"  ⚠️ WARNING: Only parsed {actual_commits}/{expected_commits} commits.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")

    if not keep_repo:
        shutil.rmtree(repo_path)
        print(f"Deleted repo folder: {repo_path}")

    return df


if __name__ == "__main__":
    mine_commits("data/repo")