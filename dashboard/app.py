# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extractor.clone_repo import clone_repo
from extractor.commit_miner import mine_commits
from extractor.labeler import label_commits
from features.build_dataset import build_dataset

st.set_page_config(
    page_title="Git Risk Analyzer",
    page_icon="https://raw.githubusercontent.com/github/explore/main/topics/git/git.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0D1117; }

[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #21262D;
}
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden !important; }
.stDeployButton { display: none !important; }

[data-testid="stMetric"] {
    background: linear-gradient(180deg, #161B22 0%, #0D1117 100%);
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset;
}
[data-testid="stMetricLabel"] {
    color: #8B949E !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #E6EDF3 !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

.stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background: #0D1117 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    color: #E6EDF3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
.stTextInput input:focus {
    border-color: #58A6FF !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #2EA043 0%, #238636 100%) !important;
    color: #fff !important;
    border: 1px solid rgba(46,160,67,0.4) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    width: 100% !important;
    padding: 0.65rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(180deg, #3FB950 0%, #2EA043 100%) !important;
    box-shadow: 0 0 12px rgba(46,160,67,0.25) !important;
}
.stButton > button[kind="secondary"] {
    background: #21262D !important;
    border: 1px solid #30363D !important;
    color: #C9D1D9 !important;
    border-radius: 8px !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #30363D !important;
    border-color: #8B949E !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #21262D;
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: #161B22 !important;
    color: #E6EDF3 !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stDataFrame"] td {
    font-size: 13px !important;
    color: #C9D1D9 !important;
}

.stMultiSelect [data-baseweb="tag"] {
    background: #1F6FEB !important;
    color: #fff !important;
    border-radius: 4px !important;
}

[data-testid="stStatus"] {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] {
    border: 1px solid #21262D;
    border-radius: 10px;
    overflow: hidden;
}

.section-label {
    font-size: 11px;
    font-weight: 700;
    color: #8B949E;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.spotlight-card {
    background: linear-gradient(135deg, #161B22 0%, #111820 100%);
    border: 1px solid #30363D;
    border-radius: 14px;
    padding: 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.spotlight-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #F85149, #D29922, #3FB950);
    opacity: 0.6;
}
.risk-pill {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid;
}
.pill-high   { background: rgba(248,81,73,0.12); color: #F85149; border-color: rgba(248,81,73,0.35); }
.pill-medium { background: rgba(210,153,34,0.12); color: #D29922; border-color: rgba(210,153,34,0.35); }
.pill-low    { background: rgba(63,185,80,0.12); color: #3FB950; border-color: rgba(63,185,80,0.35); }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

h1 a, h2 a, h3 a, [data-testid="StyledLinkIconContainer"] { display: none !important; }
small[data-testid="InputInstructions"], [data-testid="InputInstructions"] { display: none !important; }

.stDownloadButton > button {
    background: #1F6FEB !important;
    border-color: #388BFD !important;
    color: #fff !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

PLOT_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font_color": "#8B949E",
    "font_family": "Inter",
}

@st.cache_resource(show_spinner=False)
def load_model_bundle():
    path = "model/saved_model.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def get_model_status():
    bundle = load_model_bundle()
    if bundle is None:
        return False, None, None, None, None, "Not trained"
    name = bundle.get("best_model_name", "Model")
    return True, bundle.get("model"), bundle.get("scaler"), bundle.get("features", []), bundle.get("metrics", {}), name

@st.cache_data(show_spinner=False)
def get_training_stats():
    stats = {
        "repos": 15, "commits": "873K+", "files": "55K+",
        "roc_auc": "0.89", "f1": "0.80", "features": 11,
    }
    try:
        if os.path.exists("data/features.csv"):
            df = pd.read_csv("data/features.csv")
            stats["files"] = f"{len(df):,}"
            stats["features"] = len([
                c for c in df.columns
                if c not in ["repo", "file_path", "is_buggy", "first_commit_date", "last_commit_date"]
            ])
        _, _, _, _, metrics, _ = get_model_status()
        if metrics:
            if "roc_auc" in metrics:
                stats["roc_auc"] = f"{metrics['roc_auc']:.2f}"
            if "weighted_f1" in metrics:
                stats["f1"] = f"{metrics['weighted_f1']:.2f}"
    except Exception:
        pass
    return stats

def predict_repo(features_df):
    ok, model, scaler, feature_cols, _, _ = get_model_status()
    if not ok:
        st.error("Model not found. Run `python model/train.py` first.")
        st.stop()

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
    )
    df["confidence"] = pd.cut(
        proba,
        bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)

with st.sidebar:
    st.markdown("""
    <div style='padding: 0 0 24px'>
        <div style='display:flex;align-items:center;gap:12px;'>
            <div style='width:36px;height:36px;background:linear-gradient(135deg,#58A6FF,#1F6FEB);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:20px;color:#fff;font-weight:700;'>⬡</div>
            <div>
                <div style='font-family:JetBrains Mono;font-size:15px;font-weight:700;color:#E6EDF3;letter-spacing:-0.3px;'>
                    git-risk-analyzer
                </div>
                <div style='font-size:10px;color:#8B949E;letter-spacing:0.08em;text-transform:uppercase;margin-top:2px;'>
                    ML Bug Prediction
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Analyze Repository</div>', unsafe_allow_html=True)

    github_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repo",
        label_visibility="collapsed",
    )

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        analyze_btn = st.button("Analyze", use_container_width=True, type="primary")
    with col_btn2:
        clear_btn = st.button("Clear", use_container_width=True, type="secondary")

    if clear_btn and "results" in st.session_state:
        for k in ["results", "repo_name", "last_repo", "features_csv"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)

    model_ok, _, _, _, _, model_name = get_model_status()
    if model_ok:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
            <div style='width:8px;height:8px;background:#3FB950;border-radius:50%;box-shadow:0 0 6px rgba(63,185,80,0.5);'></div>
            <span style='font-size:12px;color:#3FB950;font-weight:600;'>Model Ready</span>
        </div>
        <div style='font-family:JetBrains Mono;font-size:11px;color:#8B949E;padding-left:16px;'>
            {model_name.replace("_", " ").title()}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='display:flex;align-items:center;gap:8px;'>
            <div style='width:8px;height:8px;background:#F85149;border-radius:50%;'></div>
            <span style='font-size:12px;color:#F85149;font-weight:600;'>Model Missing</span>
        </div>
        <div style='font-size:11px;color:#8B949E;margin-top:6px;line-height:1.5;'>
            Run <code>python model/train.py</code> to build the model.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px;color:#8B949E;line-height:1.8;'>
    Predicts which files are most likely to contain bugs using
    <b style='color:#E6EDF3;'>ML</b> trained on
    <b style='color:#E6EDF3;'>873K+ commits</b> from 15 major open-source repos.
    <br><br>
    <span style='color:#58A6FF;'>Pipeline:</span><br>
    Clone → Mine → Label → Features → Predict
    </div>
    """, unsafe_allow_html=True)

if analyze_btn and github_url:
    repo_name = github_url.rstrip("/").split("/")[-1]
    clone_dir = f"data/repos/{repo_name}"
    commits_csv = f"data/{repo_name}_commits.csv"
    labeled_csv = f"data/{repo_name}_labeled.csv"
    features_csv = f"data/{repo_name}_features.csv"

    if st.session_state.get("last_repo") == repo_name and "results" in st.session_state:
        st.toast(f"Showing cached results for {repo_name}", icon="✅")
    else:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        steps = [
            ("🔍 Cloning repository...", 0.05),
            ("⛏️ Mining commit history...", 0.25),
            ("🏷️ Labeling bug-inducing commits...", 0.45),
            ("🧬 Building feature set...", 0.65),
            ("🧠 Running ML predictions...", 0.85),
            ("✅ Finalizing results...", 0.95),
        ]

        try:
            for msg, pct in steps:
                progress_text.markdown(f"<span style='color:#8B949E;font-size:13px;'>{msg}</span>", unsafe_allow_html=True)
                progress_bar.progress(int(pct * 100))

                if pct == 0.05:
                    clone_repo(github_url, clone_dir)
                elif pct == 0.25:
                    mine_commits(clone_dir, commits_csv, keep_repo=True)
                elif pct == 0.45:
                    label_commits(commits_csv, labeled_csv)
                elif pct == 0.65:
                    build_dataset(data_dir="data", output_path=features_csv, single_repo_csv=labeled_csv)
                elif pct == 0.85:
                    feat_df = pd.read_csv(features_csv)
                    results = predict_repo(feat_df)
                    st.session_state["results"] = results
                    st.session_state["repo_name"] = repo_name
                    st.session_state["last_repo"] = repo_name
                    st.session_state["features_csv"] = features_csv

            progress_text.empty()
            progress_bar.empty()
            st.toast(f"Analysis complete — {len(st.session_state['results'])} files", icon="🎯")

        except Exception as e:
            progress_text.empty()
            progress_bar.empty()
            with st.expander("❌ Error Details", expanded=True):
                st.code(str(e))
            st.stop()

if "results" not in st.session_state:
    stats = get_training_stats()
    _, _, _, _, _, model_name = get_model_status()
    model_display = model_name.replace("_", " ").title() if model_name else "ML"

    st.markdown(f"""
    <div style='text-align:center;padding:80px 20px 60px;'>
        <div style='display:inline-block;padding:16px 24px;background:rgba(88,166,255,0.08);
                    border:1px solid rgba(88,166,255,0.2);border-radius:50px;margin-bottom:32px;'>
            <span style='font-family:JetBrains Mono;font-size:12px;color:#58A6FF;letter-spacing:0.1em;'>
                v2.0 · TEMPORAL VALIDATION · CALIBRATED {model_display.upper()}
            </span>
        </div>
        <div style='font-size:56px;font-weight:800;color:#E6EDF3;letter-spacing:-2px;line-height:1.1;margin-bottom:20px;'>
            Find bugs before<br><span style='color:#58A6FF;'>they find you</span>
        </div>
        <p style='color:#8B949E;font-size:16px;max-width:480px;margin:0 auto 60px;line-height:1.7;'>
            Paste any public GitHub repository URL and get an ML-powered risk analysis
            of every file in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, stats["roc_auc"], "ROC-AUC", "Temporal validation"),
        (c2, stats["f1"], "F1 Score", "Weighted harmonic mean"),
        (c3, stats["files"], "Training Files", "From 15 major repos"),
        (c4, str(stats["features"]), "Features", "Commit & code metrics"),
    ]
    for col, value, title, desc in cards:
        col.markdown(f"""
        <div style='background:linear-gradient(180deg,#161B22 0%,#111820 100%);
                    border:1px solid #21262D;border-radius:14px;padding:28px 20px;
                    text-align:center;height:100%;transition:transform 0.2s;'>
            <div style='font-size:28px;font-weight:800;color:#E6EDF3;margin-bottom:6px;'>{value}</div>
            <div style='font-size:13px;font-weight:700;color:#E6EDF3;margin-bottom:8px;'>{title}</div>
            <div style='font-size:11px;color:#8B949E;line-height:1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label" style="text-align:center;">Trusted by data from</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;color:#8B949E;font-family:JetBrains Mono;font-size:13px;letter-spacing:0.04em;margin-bottom:80px;'>
        Django · Flask · FastAPI · Pandas · NumPy · Scikit-learn · Matplotlib · Pytest · SQLAlchemy · Celery · Ansible
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="text-align:center;margin-bottom:24px;">How it works</div>', unsafe_allow_html=True)
    h1, h2, h3, h4, h5 = st.columns(5)
    steps = [
        (h1, "1", "Clone", "Fetch full git history"),
        (h2, "2", "Mine", "Extract commit metadata"),
        (h3, "3", "Label", "SZZ temporal shift"),
        (h4, "4", "Features", "Aggregate per file"),
        (h5, "5", "Predict", f"Calibrated {model_display}"),
    ]
    for col, num, title, desc in steps:
        col.markdown(f"""
        <div style='text-align:center;padding:20px 12px;'>
            <div style='width:36px;height:36px;background:#21262D;border:1px solid #30363D;
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        margin:0 auto 12px;font-family:JetBrains Mono;font-size:14px;font-weight:700;color:#58A6FF;'>{num}</div>
            <div style='font-size:14px;font-weight:600;color:#E6EDF3;margin-bottom:4px;'>{title}</div>
            <div style='font-size:11px;color:#8B949E;line-height:1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

results = st.session_state["results"]
repo_name = st.session_state["repo_name"]
features_csv = st.session_state.get("features_csv")
_, _, _, _, _, model_name = get_model_status()

high = results[results["risk_label"] == "High"]
medium = results[results["risk_label"] == "Medium"]
low = results[results["risk_label"] == "Low"]

st.markdown(f"""
<div style='padding: 8px 0 16px;'>
    <span style='font-family:JetBrains Mono;font-size:12px;color:#8B949E;'>ANALYSIS /</span>
    <span style='font-family:JetBrains Mono;font-size:12px;color:#58A6FF;font-weight:600;'> {repo_name}</span>
</div>
""", unsafe_allow_html=True)

if len(results) > 0:
    top = results.iloc[0]
    pill_class = f"pill-{top['risk_label'].lower()}"
    st.markdown(f"""
    <div class="spotlight-card">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">
            <div style="flex:1;min-width:0;">
                <div style="font-size:11px;color:#8B949E;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:10px;">
                    Highest Risk File
                </div>
                <div style="font-family:JetBrains Mono;font-size:17px;color:#E6EDF3;font-weight:500;word-break:break-all;line-height:1.4;">
                    {top['file_path']}
                </div>
            </div>
            <div style="flex-shrink:0;text-align:right;">
                <span class="risk-pill {pill_class}">{top['risk_label']} · {top['risk_score']}%</span>
                <div style="font-size:11px;color:#8B949E;margin-top:8px;">Confidence: {top['confidence']}</div>
            </div>
        </div>
        <div style="display:flex;gap:40px;margin-top:24px;flex-wrap:wrap;">
            <div><div style="font-size:11px;color:#8B949E;margin-bottom:4px;">Total Commits</div>
                 <div style="font-size:20px;color:#E6EDF3;font-weight:700;">{int(top['total_commits'])}</div></div>
            <div><div style="font-size:11px;color:#8B949E;margin-bottom:4px;">Authors</div>
                 <div style="font-size:20px;color:#E6EDF3;font-weight:700;">{int(top['unique_authors'])}</div></div>
            <div><div style="font-size:11px;color:#8B949E;margin-bottom:4px;">File Age</div>
                 <div style="font-size:20px;color:#E6EDF3;font-weight:700;">{int(top['file_age_days'])}d</div></div>
            <div><div style="font-size:11px;color:#8B949E;margin-bottom:4px;">Avg LOC</div>
                 <div style="font-size:20px;color:#E6EDF3;font-weight:700;">{int(top['avg_nloc'])}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Files", len(results))
m2.metric("High Risk", len(high), delta=f"{len(high)/len(results)*100:.1f}%", delta_color="inverse")
m3.metric("Medium", len(medium), delta=f"{len(medium)/len(results)*100:.1f}%", delta_color="off")
m4.metric("Low Risk", len(low), delta=f"{len(low)/len(results)*100:.1f}%", delta_color="normal")
m5.metric("Avg Risk", f"{results['risk_score'].mean():.1f}%")
m6.metric("Buggy Predicted", int(results["risk_label"].isin(["High","Medium"]).sum()))

if features_csv and os.path.exists(features_csv):
    with open(features_csv, "r") as f:
        st.download_button(
            label="Download CSV",
            data=f.read(),
            file_name=f"{repo_name}_risk_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()

col_a, col_b = st.columns([1, 2])

with col_a:
    st.markdown('<div class="section-label">Risk Distribution</div>', unsafe_allow_html=True)
    fig_pie = go.Figure(go.Pie(
        labels=["High", "Medium", "Low"],
        values=[len(high), len(medium), len(low)],
        hole=0.68,
        marker=dict(colors=["#F85149", "#D29922", "#3FB950"], line=dict(color="#0D1117", width=2)),
        textinfo="percent+label",
        textfont=dict(color="#E6EDF3", size=12),
        insidetextorientation="horizontal",
        hovertemplate="<b>%{label}</b><br>%{value} files<br>%{percent}<extra></extra>",
    ))
    fig_pie.update_layout(
        **PLOT_THEME, margin=dict(t=0,b=0,l=0,r=0), height=300,
        showlegend=False, annotations=[dict(text=f"{len(results)}<br>files", x=0.5, y=0.5, font_size=14, font_color="#E6EDF3", showarrow=False)]
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

with col_b:
    st.markdown('<div class="section-label">Top 15 Riskiest Files</div>', unsafe_allow_html=True)
    top15 = results.head(15).copy()
    top15["short_path"] = top15["file_path"].apply(lambda x: x if len(x) <= 48 else "..." + x[-45:])
    cmap = {"High": "#F85149", "Medium": "#D29922", "Low": "#3FB950"}
    top15["bar_color"] = top15["risk_label"].map(cmap)

    fig_bar = go.Figure(go.Bar(
        x=top15["risk_score"], y=top15["short_path"], orientation="h",
        marker=dict(color=top15["bar_color"], opacity=0.9, line=dict(width=0)),
        text=top15["risk_score"].astype(str) + "%", textposition="outside",
        textfont=dict(color="#8B949E", size=11),
        hovertemplate="<b>%{y}</b><br>Risk: %{x:.1f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        **PLOT_THEME, margin=dict(t=0,b=0,l=0,r=50), height=420,
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 108]),
        yaxis=dict(autorange="reversed", tickfont=dict(family="JetBrains Mono", size=11, color="#C9D1D9")),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

st.divider()

col_c, col_d = st.columns(2)

with col_c:
    st.markdown('<div class="section-label">Risk Score Histogram</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        results, x="risk_score", nbins=24, color="risk_label",
        color_discrete_map={"High":"#F85149","Medium":"#D29922","Low":"#3FB950"},
        template="plotly_dark", opacity=0.85,
    )
    fig_hist.update_layout(
        **PLOT_THEME, margin=dict(t=0,b=0,l=0,r=0), height=300,
        xaxis_title="Risk Score (%)", yaxis_title="Files",
        showlegend=False, bargap=0.12,
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

with col_d:
    st.markdown('<div class="section-label">Commits vs Risk</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        results, x="total_commits", y="risk_score", color="risk_label",
        color_discrete_map={"High":"#F85149","Medium":"#D29922","Low":"#3FB950"},
        opacity=0.6, template="plotly_dark", log_x=True,
        hover_data={"file_path": True, "total_commits": False, "risk_score": False},
        labels={"total_commits": "Commits (log)", "risk_score": "Risk (%)"},
    )
    fig_scatter.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig_scatter.update_layout(
        **PLOT_THEME, margin=dict(t=0,b=0,l=0,r=0), height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

st.divider()

st.markdown('<div class="section-label">All Files</div>', unsafe_allow_html=True)

f1, f2, f3 = st.columns([2, 2, 2])
with f1:
    risk_filter = st.multiselect("Risk", ["High","Medium","Low"], default=["High","Medium","Low"], label_visibility="collapsed")
with f2:
    search = st.text_input("Search", placeholder="Filter by filename...", label_visibility="collapsed")
with f3:
    sort_by = st.selectbox("Sort", ["Risk Score","Commits","Authors","Age","LOC"], label_visibility="collapsed")

sort_map = {"Risk Score":"risk_score","Commits":"total_commits","Authors":"unique_authors","Age":"file_age_days","LOC":"avg_nloc"}
filtered = results[results["risk_label"].isin(risk_filter)].copy()
if search:
    filtered = filtered[filtered["file_path"].str.contains(search, case=False, na=False)]
filtered = filtered.sort_values(sort_map.get(sort_by, "risk_score"), ascending=False)

display_cols = ["file_path","risk_score","risk_label","confidence","total_commits","unique_authors","file_age_days","avg_nloc"]
display_cols = [c for c in display_cols if c in filtered.columns]
display_df = filtered[display_cols].rename(columns={
    "file_path": "File", "risk_score": "Risk Score", "risk_label": "Level",
    "confidence": "Confidence", "total_commits": "Commits",
    "unique_authors": "Authors", "file_age_days": "Age (days)", "avg_nloc": "Avg LOC",
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Risk Score": st.column_config.ProgressColumn(
            "Risk Score", min_value=0, max_value=100, format="%.1f%%",
            help="Calibrated probability of containing a bug",
        ),
        "Level": st.column_config.TextColumn("Level", help="Risk classification"),
        "Confidence": st.column_config.TextColumn("Confidence", help="Model confidence tier"),
        "File": st.column_config.TextColumn("File", help="Repository file path", width="large"),
    },
)

st.markdown(f"""
<div style='text-align:center;padding:40px 0 20px;'>
    <div style='font-size:11px;color:#3D444D;font-family:JetBrains Mono;letter-spacing:0.06em;'>
        git-risk-analyzer · calibrated {model_name.replace("_", " ")} · temporal validation · sigmoid calibration
    </div>
</div>
""", unsafe_allow_html=True)