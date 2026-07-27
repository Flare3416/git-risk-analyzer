import os
import sys
sys.path.append(os.path.abspath("."))

from extractor.clone_repo import clone_repo
from extractor.commit_miner import mine_commits

REPOS = [
    # large
    "https://github.com/pallets/flask",
    "https://github.com/django/django",
    "https://github.com/psf/requests",
    "https://github.com/scikit-learn/scikit-learn",
    "https://github.com/ansible/ansible",          
    "https://github.com/home-assistant/core",       
    "https://github.com/keras-team/keras",          
    "https://github.com/apache/airflow",            
    "https://github.com/fastapi/fastapi", 
    "https://github.com/pallets/click",          
    "https://github.com/pytest-dev/pytest",       
    "https://github.com/httpie/cli",               
    "https://github.com/Textualize/rich",            
]

if __name__ == "__main__":
    for url in REPOS:
        repo_name = url.rstrip("/").split("/")[-1]
        clone_dir = f"data/repos/{repo_name}"
        output_csv = f"data/{repo_name}_commits.csv"

        # skip if already mined
        if os.path.exists(output_csv):
            print(f"Skipping {repo_name} — already mined")
            continue

        clone_repo(url, clone_dir)
        mine_commits(clone_dir, output_csv)
        print(f"✓ {repo_name} done\n")