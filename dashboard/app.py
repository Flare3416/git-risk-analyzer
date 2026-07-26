import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import sys
sys.path.append(os.path.abspath("."))

from extractor.clone_repo import clone_repo
from extractor.commit_miner import mine_commits
from extractor.labeler import label_commits
from features.build_dataset import build_dataset


st.set_page_config(
    page_title="Git Risk Analyzer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* page background */
.stApp { background-color: #0D1117; }

/* sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #21262D;
}

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* metric cards */
[data-testid="stMetric"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #E6EDF3 !important; font-size: 28px !important; font-weight: 600 !important; }

/* inputs */
.stTextInput input {
    background: #0D1117 !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
    color: #E6EDF3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
.stTextInput input:focus { border-color: #58A6FF !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important; }

/* button */
.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: 1px solid #2EA043 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    width: 100% !important;
    padding: 0.6rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #2EA043 !important; }

/* dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #21262D;
    border-radius: 8px;
    overflow: hidden;
}

/* multiselect */
.stMultiSelect [data-baseweb="tag"] { background: #1F6FEB !important; }

/* divider */
hr { border-color: #21262D !important; }

/* status */
[data-testid="stStatus"] {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 8px !important;
}

/* risk badges */
.badge-high   { background:#3D1A1A; color:#F85149; border:1px solid #F85149; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:500; }
.badge-medium { background:#2D1F00; color:#D29922; border:1px solid #D29922; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:500; }
.badge-low    { background:#0D2818; color:#3FB950; border:1px solid #3FB950; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:500; }

/* section headers */
.section-label {
    font-size: 11px;
    font-weight: 600;
    color: #8B949E;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


def load_model():
    with open("model/saved_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["features"]


def predict_repo(features_df):
    model, scaler, feature_cols = load_model()
    df = features_df.copy()
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols]
    df["risk_score"] = (model.predict_proba(X)[:, 1] * 100).round(1)
    df["risk_label"] = pd.cut(
        df["risk_score"],
        bins=[0, 40, 70, 100],
        labels=["Low", "Medium", "High"]
    )
    return df.sort_values("risk_score", ascending=False)


PLOT_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "font_color":    "#8B949E",
    "font_family":   "Inter",
}

# ── sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px'>
        <div style='font-family:JetBrains Mono;font-size:15px;font-weight:500;color:#E6EDF3;letter-spacing:0.02em;'>
            git-risk-analyzer
        </div>
        <div style='font-size:11px;color:#8B949E;margin-top:3px;letter-spacing:0.05em;text-transform:uppercase;'>
            ML-powered bug prediction
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Repository</div>', unsafe_allow_html=True)
    github_url  = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo", label_visibility="collapsed")
    analyze_btn = st.button("Analyze Repository")

    st.divider()
    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px;color:#8B949E;line-height:1.7;'>
    Analyzes git commit history to predict which files are most likely to contain bugs using XGBoost trained on 320k+ commits.
    <br><br>
    <b style='color:#E6EDF3;'>Features used</b><br>
    Change frequency · Developer churn · Commit size · File age · Code volume
    </div>
    """, unsafe_allow_html=True)

# ── main ─────────────────────────────────────────────
if not analyze_btn or not github_url:
    st.markdown("""
    <div style='padding: 100px 0 48px; text-align:center;'>
        <div style='font-family:JetBrains Mono;font-size:11px;color:#30363D;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:20px;'>
            ML-POWERED BUG PREDICTION
        </div>
        <h1 style='color:#E6EDF3;font-size:44px;font-weight:600;margin:0 0 16px;letter-spacing:-1px;line-height:1.2;'>
            Find bugs before<br>they find you.
        </h1>
        <p style='color:#8B949E;font-size:15px;max-width:420px;margin:0 auto 48px;line-height:1.7;'>
            Paste any public GitHub repo and get an ML analysis of which files carry the highest bug risk — based on real commit history.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, title, desc in [
        (c1, "320k+ commits analyzed", "Trained on Flask, Django, Scikit-learn, Pytest and more"),
        (c2, "86% accuracy", "XGBoost model with ROC AUC of 0.92"),
        (c3, "11 features", "Change frequency, dev churn, commit size and more"),
    ]:
        col.markdown(f"""
        <div style='background:#161B22;border:1px solid #21262D;border-radius:8px;padding:20px;text-align:center;'>
            <div style='font-size:20px;font-weight:600;color:#58A6FF;margin-bottom:6px;'>{title}</div>
            <div style='font-size:13px;color:#8B949E;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── analysis ─────────────────────────────────────────
repo_name    = github_url.rstrip("/").split("/")[-1]
clone_dir    = f"data/repos/{repo_name}"
commits_csv  = f"data/{repo_name}_commits.csv"
features_csv = f"data/{repo_name}_features.csv"

with st.status(f"Analyzing **{repo_name}**...", expanded=True) as status:
    st.write("Cloning repository...")
    clone_repo(github_url, clone_dir)
    st.write("Mining commit history...")
    mine_commits(clone_dir, commits_csv)
    st.write("Labeling bug commits...")
    label_commits(commits_csv)
    st.write("Building feature set...")
    features_df = build_dataset(data_dir="data", output_path=features_csv, single_repo_csv=commits_csv)
    st.write("Running predictions...")
    results = predict_repo(features_df)
    st.write("Prediction complete")

    status.update(label="Analysis complete", state="complete")

high   = results[results["risk_label"] == "High"]
medium = results[results["risk_label"] == "Medium"]
low    = results[results["risk_label"] == "Low"]

# ── metrics ──────────────────────────────────────────
st.markdown(f"""
<div style='padding: 24px 0 8px;'>
    <span style='font-family:JetBrains Mono;font-size:13px;color:#8B949E;'>analysis /</span>
    <span style='font-family:JetBrains Mono;font-size:13px;color:#58A6FF;'> {repo_name}</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Files", len(results))
c2.metric("High Risk",   len(high))
c3.metric("Medium Risk", len(medium))
c4.metric("Low Risk",    len(low))
st.divider()

# ── charts ───────────────────────────────────────────
col_a, col_b = st.columns([1, 2])

with col_a:
    st.markdown('<div class="section-label">Risk breakdown</div>', unsafe_allow_html=True)
    dist = pd.DataFrame({
        "Level": ["High", "Medium", "Low"],
        "Count": [len(high), len(medium), len(low)]
    })
    fig_pie = go.Figure(go.Pie(
        labels=dist["Level"],
        values=dist["Count"],
        hole=0.65,
        marker=dict(colors=["#F85149", "#D29922", "#3FB950"]),
        textinfo="percent",
        textfont=dict(color="#E6EDF3", size=13),
    ))
    fig_pie.update_layout(
        **PLOT_THEME,
        margin=dict(t=0, b=0, l=0, r=0),
        height=240,
        showlegend=True,
        legend=dict(font=dict(color="#8B949E", size=12), bgcolor="rgba(0,0,0,0)")
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.markdown('<div class="section-label">Top 15 riskiest files</div>', unsafe_allow_html=True)
    top15 = results.head(15).copy()
    top15["short_path"] = top15["file_path"].apply(lambda x: x if len(x) <= 45 else "..." + x[-42:])
    color_map = {"High": "#F85149", "Medium": "#D29922", "Low": "#3FB950"}
    top15["color"] = top15["risk_label"].map(color_map)

    fig_bar = go.Figure(go.Bar(
        x=top15["risk_score"],
        y=top15["short_path"],
        orientation="h",
        marker=dict(color=top15["color"], opacity=0.85),
        text=top15["risk_score"].astype(str) + "%",
        textposition="outside",
        textfont=dict(color="#8B949E", size=11),
    ))
    fig_bar.update_layout(
        **PLOT_THEME,
        margin=dict(t=0, b=0, l=0, r=60),
        height=360,
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 120]),
        yaxis=dict(autorange="reversed", tickfont=dict(
            family="JetBrains Mono", size=11, color="#E6EDF3"
        )),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── table ────────────────────────────────────────────
st.markdown('<div class="section-label">All files</div>', unsafe_allow_html=True)

col_f1, col_f2, _ = st.columns([1, 1, 2])

with col_f1:
    risk_filter = st.multiselect(
        "Risk level",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
        label_visibility="collapsed"
    )

with col_f2:
    search = st.text_input(
        "Search file",
        placeholder="filter by filename...",
        label_visibility="collapsed"
    )

filtered = results[results["risk_label"].isin(risk_filter)]

if search:
    filtered = filtered[
        filtered["file_path"].str.contains(search, case=False)
    ]

filtered = filtered.reset_index(drop=True)

def risk_color(label):
    return {"High": "#F85149", "Medium": "#D29922", "Low": "#3FB950"}.get(str(label), "#8B949E")

def risk_bg(label):
    return {"High": "#3D1A1A", "Medium": "#2D1F00", "Low": "#0D2818"}.get(str(label), "#161B22")

rows = ""
for _, row in filtered.iterrows():
    color = risk_color(row["risk_label"])
    bg    = risk_bg(row["risk_label"])
    score = row["risk_score"]
    rows += f"""
    <tr>
        <td style='padding:10px 12px;color:#E6EDF3;font-family:JetBrains Mono;font-size:12px;'>{row['file_path']}</td>
        <td style='padding:10px 12px;min-width:160px;'>
            <div style='display:flex;align-items:center;gap:8px;'>
                <div style='flex:1;background:#21262D;border-radius:4px;height:6px;'>
                    <div style='width:{score}%;background:{color};border-radius:4px;height:6px;'></div>
                </div>
                <span style='font-size:12px;color:{color};font-weight:500;min-width:38px;'>{score}%</span>
            </div>
        </td>
        <td style='padding:10px 12px;'>
            <span style='background:{bg};color:{color};border:1px solid {color};
                         padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500;'>
                {row['risk_label']}
            </span>
        </td>
        <td style='padding:10px 12px;color:#8B949E;font-size:13px;text-align:right;'>{int(row['total_commits'])}</td>
        <td style='padding:10px 12px;color:#8B949E;font-size:13px;text-align:right;'>{int(row['unique_authors'])}</td>
        <td style='padding:10px 12px;color:#8B949E;font-size:13px;text-align:right;'>{int(row['file_age_days'])}</td>
        <td style='padding:10px 12px;color:#8B949E;font-size:13px;text-align:right;'>{round(row['avg_nloc'], 1)}</td>
    </tr>
    """

st.markdown(f"""
<div style='border:1px solid #21262D;border-radius:8px;overflow:hidden;'>
<table style='width:100%;border-collapse:collapse;'>
    <thead>
        <tr style='background:#161B22;border-bottom:1px solid #21262D;'>
            <th style='padding:10px 12px;text-align:left;font-size:11px;color:#8B949E;font-weight:500;letter-spacing:0.05em;'>FILE</th>
            <th style='padding:10px 12px;text-align:left;font-size:11px;color:#8B949E;font-weight:500;letter-spacing:0.05em;'>RISK SCORE</th>
            <th style='padding:10px 12px;text-align:left;font-size:11px;color:#8B949E;font-weight:500;letter-spacing:0.05em;'>LEVEL</th>
            <th style='padding:10px 12px;text-align:right;font-size:11px;color:#8B949E;font-weight:500;letter-spacing:0.05em;'>COMMITS</th>
            <th style='padding:10px 12px;text-align:right;font-size:11px;color:#8B949E;font-weight:500;letter-spacing:0.05em;'>AUTHORS</th>
            <th style='padding:10px 12px;text-align:right;font-size:11px;color:#8B949E;font-weight:500;letter-spacing:0.05em;'>AGE (DAYS)</th>
            <th style='padding:10px 12px;text-align:right;font-size:11px;color:#8B949E;font-weight:500;letter-spacing:0.05em;'>AVG LOC</th>
        </tr>
    </thead>
    <tbody>
        {rows}
    </tbody>
</table>
</div>
""", unsafe_allow_html=True)