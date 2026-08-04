import os
import shutil
from git import Repo, GitCommandError

def clone_repo(
    github_url: str,
    clone_dir: str = "data/repo",
    force: bool = False,
) -> str:
    if os.path.exists(clone_dir):
        if force:
            shutil.rmtree(clone_dir)
            print(f"Removed existing repo at {clone_dir}")
        else:
            print(f"Repo already exists at {clone_dir} — skipping clone")
            return clone_dir

    print(f"Cloning {github_url} ...")
    try:
        # Full clone with optimizations (single branch, no tags) to support git log --numstat without remote fetches
        Repo.clone_from(github_url, clone_dir, single_branch=True, no_tags=True)
        print(f"Cloned → {clone_dir}")
    except GitCommandError as e:
        raise RuntimeError(f"Failed to clone {github_url}: {e}")

    return clone_dir


if __name__ == "__main__":
    url = input("Enter GitHub repo URL: ")
    clone_repo(url)