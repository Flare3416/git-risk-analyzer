import os
import shutil
import pandas as pd
from pydriller import Repository

def mine_commits(repo_path: str, output_path: str = "data/raw_commits.csv") -> pd.DataFrame:
    records = []

    print(f"Mining commits from {repo_path} ...")

    for commit in Repository(repo_path).traverse_commits():
        for modified_file in commit.modified_files:
            records.append({
                "commit_hash":    commit.hash,
                "author":         commit.author.name,
                "date":           commit.author_date,
                "msg":            commit.msg,
                "file":           modified_file.filename,
                "file_path":      modified_file.new_path,
                "lines_added":    modified_file.added_lines,
                "lines_deleted":  modified_file.deleted_lines,
                "complexity":     modified_file.complexity,
                "nloc":           modified_file.nloc,  # lines of code
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Done. {len(df)} rows extracted → {output_path}")
    
    # cleanup repo folder
    shutil.rmtree(repo_path)
    print(f"Deleted repo folder: {repo_path}")
    
    return df

if __name__ == "__main__":
    mine_commits("data/repo")