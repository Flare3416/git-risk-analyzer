# backend/tasks.py
import os
import sys
import shutil
import tempfile
import traceback
import pickle
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extractor.clone_repo import clone_repo
from extractor.commit_miner import mine_commits
from extractor.labeler import label_commits
from features.build_dataset import build_dataset

# In-memory job state store
JOBS = {}

def load_model_bundle():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/saved_model.pkl"))
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def run_predictions_on_df(features_df, bundle):
    model = bundle["model"]
    scaler = bundle.get("scaler")
    feature_cols = bundle.get("features", [])

    df = features_df.copy()
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols]

    if scaler is not None:
        X = scaler.transform(X)

    proba = model.predict_proba(X)[:, 1]
    df["risk_score"] = (proba * 100).round(1)
    df["risk_label"] = pd.cut(
        df["risk_score"],
        bins=[0, 40, 70, 100],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    
    df["confidence"] = pd.cut(
        proba,
        bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    ).astype(str)

    # Sort: riskiest first
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df

def sanitize_error_message(e: Exception) -> str:
    err_str = str(e)
    if "Failed to clone" in err_str or "git clone" in err_str or "exit code(128)" in err_str or "could not read Username" in err_str:
        return "Repository not found or is private. Please verify the URL and ensure the repository is public."
    if "git log" in err_str:
        return "Failed to analyze repository commit history. The repository might be empty or corrupted."
    return err_str

def analyze_repository_task(job_id: str, github_url: str):
    repo_name = github_url.rstrip("/").split("/")[-1]
    temp_dir = tempfile.mkdtemp(prefix=f"gitrisk_job_{job_id}_")
    
    clone_dir = os.path.join(temp_dir, repo_name)
    commits_csv = os.path.join(temp_dir, f"{repo_name}_commits.csv")
    labeled_csv = os.path.join(temp_dir, f"{repo_name}_labeled.csv")
    features_csv = os.path.join(temp_dir, f"{repo_name}_features.csv")

    try:
        # Step 1: Clone
        JOBS[job_id].update({"status": "cloning", "progress": 10})
        clone_repo(github_url, clone_dir)
        
        # Step 2: Mine
        JOBS[job_id].update({"status": "mining", "progress": 30})
        mine_commits(clone_dir, commits_csv, keep_repo=False)
        
        # Step 3: Label
        JOBS[job_id].update({"status": "labeling", "progress": 50})
        label_commits(commits_csv, labeled_csv)
        
        # Step 4: Features
        JOBS[job_id].update({"status": "building_features", "progress": 70})
        build_dataset(
            data_dir=temp_dir,
            output_path=features_csv,
            single_repo_csv=labeled_csv,
        )
        
        # Step 5: Predict
        JOBS[job_id].update({"status": "predicting", "progress": 90})
        bundle = load_model_bundle()
        if bundle is None:
            raise RuntimeError("ML Model not found on server. Run training first.")
        
        features_df = pd.read_csv(features_csv)
        predictions_df = run_predictions_on_df(features_df, bundle)
        
        # Format results for API payload
        records = predictions_df.to_dict(orient="records")
        
        # Compute summary metrics
        high_count = sum(1 for r in records if r["risk_label"] == "High")
        medium_count = sum(1 for r in records if r["risk_label"] == "Medium")
        low_count = sum(1 for r in records if r["risk_label"] == "Low")
        avg_risk = float(predictions_df["risk_score"].mean()) if len(predictions_df) > 0 else 0.0

        JOBS[job_id].update({
            "status": "success",
            "progress": 100,
            "results": {
                "repo_name": repo_name,
                "github_url": github_url,
                "total_files": len(records),
                "high_risk_count": high_count,
                "medium_risk_count": medium_count,
                "low_risk_count": low_count,
                "average_risk_score": round(avg_risk, 2),
                "files": records
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        JOBS[job_id].update({
            "status": "failed",
            "progress": 100,
            "error": sanitize_error_message(e),
            "traceback": traceback.format_exc()
        })
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
