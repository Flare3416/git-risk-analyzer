import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

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

def evaluate(features_path: str = "data/features.csv"):
    print("Loading model...")
    with open("model/saved_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model    = bundle["model"]
    features = bundle["features"]

    print("Loading dataset...")
    df = pd.read_csv(features_path)
    df = df.dropna(subset=FEATURES + ["is_buggy"])

    X = df[FEATURES]
    y = df["is_buggy"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    # classification report
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_test, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  True Negative  (Clean → Clean):  {cm[0][0]}")
    print(f"  False Positive (Clean → Buggy):  {cm[0][1]}")
    print(f"  False Negative (Buggy → Clean):  {cm[1][0]}")
    print(f"  True Positive  (Buggy → Buggy):  {cm[1][1]}")

    # ROC AUC
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC Score: {auc:.4f}")

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.4f})", color="darkorange")
    plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Bug Prediction Model")
    plt.legend()
    plt.tight_layout()
    os.makedirs("model", exist_ok=True)
    plt.savefig("model/roc_curve.png")
    print("ROC curve saved → model/roc_curve.png")


if __name__ == "__main__":
    evaluate()