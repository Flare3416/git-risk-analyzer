import os
import shutil
from git import Repo

def clone_repo(github_url: str, clone_dir: str = "data/repo") -> str:
    # If repo already cloned, delete and reclone fresh
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    
    print(f"Cloning {github_url} ...")
    Repo.clone_from(github_url, clone_dir)
    print(f"Cloned successfully to {clone_dir}")
    
    return clone_dir


if __name__ == "__main__":
    url = input("Enter GitHub repo URL: ")
    clone_repo(url)