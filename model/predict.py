import pandas as pd
import pickle
import os

def load_model(model_path: str = "model/saved_model.pkl"):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["features"]


def predict(features_path: str = "data/features.csv", output_path: str = "data/predictions.csv"):
    print("Loading model...")
    model, scaler, feature_cols = load_model()

    print("Loading features...")
    df = pd.read_csv(features_path)
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]

    # predict probability and label
    df["risk_score"]  = (model.predict_proba(X)[:, 1] * 100).round(1)
    df["risk_label"]  = pd.cut(
        df["risk_score"],
        bins=[0, 40, 70, 100],
        labels=["🟢 Low", "🟡 Medium", "🔴 High"]
    )

    # sort by risk
    df = df.sort_values("risk_score", ascending=False)

    # save
    output_cols = ["repo", "file_path", "risk_score", "risk_label",
                   "total_commits", "unique_authors", "bug_fix_rate",
                   "file_age_days", "avg_nloc"]
    
    df[output_cols].to_csv(output_path, index=False)
    print(f"Predictions saved → {output_path}")
    print(f"\nTop 10 riskiest files:")
    print(df[output_cols].head(10).to_string(index=False))
    return df


if __name__ == "__main__":
    predict()