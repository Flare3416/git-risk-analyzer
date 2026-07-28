# extractor/run_all.py
import os
import sys
import traceback
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath("."))

from extractor.clone_repo import clone_repo
from extractor.commit_miner import mine_commits
from extractor.labeler import label_commits

REPOS = [
    "https://github.com/django/django",
    "https://github.com/pallets/flask",
    "https://github.com/fastapi/fastapi",
    "https://github.com/psf/requests",
    "https://github.com/encode/httpx",
    "https://github.com/pandas-dev/pandas",
    "https://github.com/scikit-learn/scikit-learn",
    "https://github.com/numpy/numpy",
    "https://github.com/matplotlib/matplotlib",
    "https://github.com/pallets/click",
    "https://github.com/Textualize/rich",
    "https://github.com/pytest-dev/pytest",
    "https://github.com/ansible/ansible",
    "https://github.com/celery/celery",
    "https://github.com/sqlalchemy/sqlalchemy",
]

MAX_WORKERS = 3


def process_repo(url: str, force: bool = False) -> dict:
    repo_name = url.rstrip("/").split("/")[-1]
    clone_dir = f"data/repos/{repo_name}"
    commits_csv = f"data/{repo_name}_commits.csv"
    labeled_csv = f"data/{repo_name}_labeled.csv"

    result = {
        "repo": repo_name,
        "status": "skipped",
        "mined_rows": 0,
        "error": None,
    }

    if os.path.exists(labeled_csv) and not force:
        result["status"] = "skipped (labeled csv exists)"
        return result

    try:
        if not os.path.exists(clone_dir):
            clone_repo(url, clone_dir, force=force)  # ← no shallow arg
        else:
            print(f"  [{repo_name}] repo already cloned")

        if not os.path.exists(commits_csv) or force:
            df = mine_commits(clone_dir, commits_csv, keep_repo=True)
            result["mined_rows"] = len(df)
        else:
            print(f"  [{repo_name}] raw csv exists, skipping mine")

        if not os.path.exists(labeled_csv) or force:
            label_commits(commits_csv, labeled_csv)

        result["status"] = "success"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Clone, mine, and label repos")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--repo", type=str, default=None)
    args = parser.parse_args()

    urls = [u for u in REPOS if not args.repo or args.repo.lower() in u.lower()]
    if not urls:
        print(f"No repo matching '{args.repo}'")
        return

    print(f"Processing {len(urls)} repo(s) | workers={args.workers} | force={args.force}\n")

    results = []
    if args.workers == 1:
        for url in urls:
            res = process_repo(url, force=args.force)
            results.append(res)
            if res["status"] == "failed":
                print(f"✗ {res['repo']} FAILED:\n{res['error']}")
            else:
                print(f"{'✓' if res['status'] == 'success' else '→'} {res['repo']}: {res['status']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_repo, url, args.force): url for url in urls}
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                if res["status"] == "failed":
                    print(f"✗ {res['repo']} FAILED:\n{res['error']}")
                else:
                    print(f"{'✓' if res['status'] == 'success' else '→'} {res['repo']}: {res['status']}")

    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if "skipped" in r["status"]]
    failed  = [r for r in results if r["status"] == "failed"]

    print("\n" + "=" * 50)
    print(f"Success : {len(success)}")
    print(f"Skipped : {len(skipped)}")
    print(f"Failed  : {len(failed)}")


if __name__ == "__main__":
    main()