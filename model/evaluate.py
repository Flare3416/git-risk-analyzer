import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

FEATURES = [
    "total_commits",
    "commits_last_30d",
    "commits_last_90d",
    "days_since_last_change",
    "unique_authors",
    "top_owner_pct",
    "avg_lines_added",
    "avg_lines_deleted",
    "max_lines_changed",
    "file_age_days",
    "avg_nloc",
]


def temporal_split(df, date_col="last_commit_date", train_frac=0.7, val_frac=0.15):
    """Same logic as train.py — test = newest files."""
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    val_end = int(n * (train_frac + val_frac))
    return df.index[val_end:]


def evaluate(features_path: str = "data/features.csv"):
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Load bundle
    with open("model/saved_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    raw_model = bundle.get("raw_model", model)
    best_name = bundle.get("best_model_name", "Model")
    features = bundle.get("features", FEATURES)

    print(f"Model: {best_name} (calibrated)")
    print(f"Features: {features}\n")

    # Load data
    df = pd.read_csv(features_path)
    df = df.dropna(subset=features + ["is_buggy"])

    date_col = "last_commit_date" if "last_commit_date" in df.columns else None
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    X = df[features]
    y = df["is_buggy"]

    # Temporal test split (matches train.py)
    if date_col and date_col in df.columns:
        test_idx = temporal_split(df, date_col=date_col)
    else:
        test_idx = df.index[int(len(df) * 0.85):]

    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    print(f"Test set: {len(y_test):,} samples | Bug rate: {y_test.mean():.2%}\n")

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Classification Report ──
    print("=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Clean", "Buggy"]))

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  TN (Clean→Clean) : {cm[0][0]:,}")
    print(f"  FP (Clean→Buggy) : {cm[0][1]:,}")
    print(f"  FN (Buggy→Clean) : {cm[1][0]:,}")
    print(f"  TP (Buggy→Buggy) : {cm[1][1]:,}")

    # ── Scores ──
    print("\n" + "=" * 60)
    print("Probabilistic Metrics")
    print("=" * 60)
    print(f"  ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  PR-AUC  : {average_precision_score(y_test, y_prob):.4f}")
    print(f"  Brier   : {brier_score_loss(y_test, y_prob):.4f}")

    os.makedirs("model", exist_ok=True)

    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color="#58A6FF", linewidth=2,
             label=f"{best_name} (AUC = {roc_auc_score(y_test, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("model/roc_curve.png", dpi=150)
    plt.close()
    print("\nSaved → model/roc_curve.png")

    # ── PR Curve ──
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, color="#F85149", linewidth=2,
             label=f"{best_name} (AP = {average_precision_score(y_test, y_prob):.3f})")
    plt.axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.3,
                label=f"Baseline ({y_test.mean():.2%})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("model/pr_curve.png", dpi=150)
    plt.close()
    print("Saved → model/pr_curve.png")

    # ── Confusion Matrix Heatmap ──
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Clean", "Pred Buggy"],
                yticklabels=["Actual Clean", "Actual Buggy"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved → model/confusion_matrix.png")

    # ── SHAP Summary ──
    if SHAP_AVAILABLE:
        print("\n" + "=" * 60)
        print("SHAP Explainability")
        print("=" * 60)

        sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
        try:
            explainer = shap.TreeExplainer(raw_model)
            shap_values = explainer.shap_values(sample)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance — {best_name}")
            plt.tight_layout()
            plt.savefig("model/shap_importance.png", dpi=150)
            plt.close()
            print("Saved → model/shap_importance.png")
            print("\nSHAP shows WHICH features drive predictions.")
            print("Next: we'll add per-file SHAP to the dashboard.")
        except Exception as e:
            print(f"SHAP skipped: {e}")
    else:
        print("\nTip: pip install shap  (for feature importance plots)")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()