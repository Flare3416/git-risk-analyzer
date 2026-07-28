import os
import pickle
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()
TOP_K = 10
DEBUG = False


def load_model(model_path: str = "model/saved_model.pkl"):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return (
        bundle["model"],
        bundle.get("scaler", None),
        bundle.get("features", []),
        bundle.get("best_model_name", "Model"),
        bundle.get("metrics", {}),
    )


def _risk_style(score: float):
    if score >= 70:
        return "HIGH", "red"
    elif score >= 40:
        return "MEDIUM", "yellow"
    return "LOW", "green"


def confidence_label(prob: float):
    if prob >= 0.90:
        return "Very High"
    elif prob >= 0.70:
        return "High"
    elif prob >= 0.50:
        return "Medium"
    elif prob >= 0.30:
        return "Low"
    return "Very Low"


def predict(features_path: str = "data/features.csv", output_path: str = "data/predictions.csv"):
    console.print("=" * 60, style="bold blue")
    console.print("Git Risk Analyzer — Inference", style="bold blue")
    console.print("=" * 60, style="bold blue")

    model, scaler, feature_cols, model_name, metrics = load_model()
    console.print(f"\nModel: [cyan]{model_name}[/cyan] (calibrated)")
    console.print(f"Features: [dim]{len(feature_cols)}[/dim]")
    if metrics:
        console.print(f"ROC-AUC: [green]{metrics.get('roc_auc', 0):.4f}[/green]")

    df = pd.read_csv(features_path)
    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    if dropped:
        console.print(f"[yellow]Dropped {dropped} rows with missing features.[/yellow]")

    X = df[feature_cols].copy()
    if scaler is not None:
        X = scaler.transform(X)

    probability = model.predict_proba(X)[:, 1]
    prediction = model.predict(X)

    df["bug_probability"] = probability
    df["prediction"] = prediction
    df["risk_score"] = (probability * 100).round(1)
    df["confidence"] = df["bug_probability"].apply(confidence_label)
    df["risk_label"] = df["risk_score"].apply(lambda x: _risk_style(x)[0])

    sort_columns = ["bug_probability"]
    ascending = [False]
    if "total_commits" in df.columns:
        sort_columns.append("total_commits")
        ascending.append(False)
    if "unique_authors" in df.columns:
        sort_columns.append("unique_authors")
        ascending.append(False)

    df = df.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)

    if DEBUG:
        console.print("\nProbability Diagnostics")
        console.print(f"Min : {df['bug_probability'].min():.6f}")
        console.print(f"Max : {df['bug_probability'].max():.6f}")
        console.print(df["bug_probability"].head(10).to_list())

    # Save Predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_columns = [
        "repo", "file_path", "prediction", "bug_probability", "risk_score",
        "risk_label", "confidence", "total_commits", "commits_last_30d",
        "commits_last_90d", "unique_authors", "top_owner_pct", "avg_lines_added",
        "avg_lines_deleted", "max_lines_changed", "file_age_days", "avg_nloc", "bug_fix_rate",
    ]
    export_columns = [c for c in export_columns if c in df.columns]
    df[export_columns].to_csv(output_path, index=False)
    console.print(f"\n[green]Predictions saved → {output_path}[/green]")

    # Top K Table
    console.print(f"\n[bold underline]Top {TOP_K} Riskiest Files[/bold underline]\n")
    table = Table(
        show_header=True,
        header_style="bold white on blue",
        box=box.SIMPLE_HEAVY,
        row_styles=["none", "dim"],
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("File", min_width=42, no_wrap=False)
    table.add_column("Risk", justify="center", width=8)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Confidence", justify="center", width=12)
    if "total_commits" in df.columns:
        table.add_column("Commits", justify="right", width=8)
    if "unique_authors" in df.columns:
        table.add_column("Authors", justify="right", width=8)

    for rank, (_, row) in enumerate(df.head(TOP_K).iterrows(), start=1):
        label, color = _risk_style(row["risk_score"])
        path = row["file_path"]
        if len(path) > 55:
            path = "..." + path[-52:]
        values = [
            str(rank),
            path,
            Text(label, style=f"bold {color}"),
            f"{row['bug_probability'] * 100:.2f}%",
            row["confidence"],
        ]
        if "total_commits" in df.columns:
            values.append(str(int(row["total_commits"])))
        if "unique_authors" in df.columns:
            values.append(str(int(row["unique_authors"])))
        table.add_row(*values)

    console.print(table)

    # Summary
    total_files = len(df)
    predicted_buggy = int(df["prediction"].sum())
    predicted_clean = total_files - predicted_buggy
    avg_probability = df["bug_probability"].mean() * 100
    high = (df["risk_label"] == "HIGH").sum()
    medium = (df["risk_label"] == "MEDIUM").sum()
    low = (df["risk_label"] == "LOW").sum()

    summary = Table(title="Prediction Summary", box=box.ROUNDED, show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Total Files", f"{total_files:,}")
    summary.add_row("Predicted Buggy", f"{predicted_buggy:,}")
    summary.add_row("Predicted Clean", f"{predicted_clean:,}")
    summary.add_row("Average Risk", f"{avg_probability:.2f}%")
    summary.add_row("HIGH Risk", str(high))
    summary.add_row("MEDIUM Risk", str(medium))
    summary.add_row("LOW Risk", str(low))
    console.print()
    console.print(summary)

    if DEBUG:
        console.print("\n[bold]Debug Statistics[/bold]")
        console.print(f"Minimum Probability : {df['bug_probability'].min():.6f}")
        console.print(f"Maximum Probability : {df['bug_probability'].max():.6f}")
        console.print(f"Mean Probability    : {df['bug_probability'].mean():.6f}")
        console.print(f"Median Probability  : {df['bug_probability'].median():.6f}")
        console.print("\nTop 10 Raw Probabilities")
        console.print(df["bug_probability"].head(10).round(6).to_list())

    console.print()
    console.print("[bold green]Prediction completed successfully.[/bold green]")
    return df


if __name__ == "__main__":
    predict()