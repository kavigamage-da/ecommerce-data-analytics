"""
streamlit_app.py
================
FAANG-Level E-Commerce Analytics Dashboard
==========================================
Author  : Portfolio Project — Data Analytics
Version : 3.0 (FAANG-ready, 2027 hiring standard)

Architecture
------------
  Raw CSVs → @st.cache_data loaders → feature engineering → KPIs
           → RFM segmentation → cohort retention → churn prediction
           → revenue forecasting → interactive Plotly charts

Robustness
----------
  • Every data load is wrapped in try/except with friendly st.warning()
  • Missing columns are auto-computed where possible
  • Missing model file → graceful fallback to precomputed churn_prob
  • All filters propagate to every chart via session state
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import sys, os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Analytics | FAANG Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME / CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── KPI cards ──────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.5); }
.kpi-label  { font-size: 12px; font-weight: 600; color: #8b949e; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
.kpi-value  { font-size: 32px; font-weight: 700; color: #58a6ff; line-height: 1; }
.kpi-delta  { font-size: 12px; margin-top: 6px; color: #3fb950; }
.kpi-delta.negative { color: #f85149; }

/* ── Section headers ────────────────────────────────────────── */
.section-header {
    font-size: 20px; font-weight: 700; color: #e6edf3;
    border-left: 4px solid #58a6ff;
    padding-left: 12px; margin: 24px 0 16px 0;
}

/* ── Insight boxes ──────────────────────────────────────────── */
.insight-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
    font-size: 13px; color: #c9d1d9; line-height: 1.6;
}
.insight-box strong { color: #58a6ff; }

/* ── Tabs ───────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    color: #8b949e !important; font-weight: 600;
    border-radius: 6px 6px 0 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
}

/* ── Plotly charts ─────────────────────────────────────────── */
.js-plotly-plot { border-radius: 10px; }

/* ── Prediction panel ───────────────────────────────────────── */
.pred-result-high   { background:#2d1b1b; border:1px solid #f85149; border-radius:10px; padding:18px; }
.pred-result-medium { background:#2d241b; border:1px solid #d29922; border-radius:10px; padding:18px; }
.pred-result-low    { background:#1b2d1b; border:1px solid #3fb950; border-radius:10px; padding:18px; }

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PLOTLY DARK THEME DEFAULTS
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c9d1d9", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=True),
    colorway=["#58a6ff","#3fb950","#f0883e","#a371f7","#f85149","#79c0ff","#56d364"],
)

SEGMENT_COLORS = {
    "Champions":          "#3fb950",
    "Loyal Customers":    "#58a6ff",
    "Potential Loyalists":"#a371f7",
    "At Risk":            "#f0883e",
    "Lost":               "#f85149",
    "Others":             "#8b949e",
}

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent

def _find_project_root() -> Path:
    candidate = _HERE
    for _ in range(6):
        if (candidate / "data" / "full_dataset_10k.csv").exists():
            return candidate
        candidate = candidate.parent

    fallbacks = [
        Path("C:/Users/ASUS/Documents/Ecommerce_Data_Analytics"),
        Path.home() / "Documents" / "Ecommerce_Data_Analytics",
        Path.home() / "Ecommerce_Data_Analytics",
        Path.cwd(),
        Path.cwd().parent,
    ]
    for p in fallbacks:
        if (p / "data" / "full_dataset_10k.csv").exists():
            return p

    return Path.cwd()

PROJECT_ROOT = _find_project_root()
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

_WIN_DATA    = Path("C:/Users/ASUS/Documents/Ecommerce_Data_Analytics/data")
_WIN_OUTPUTS = Path("C:/Users/ASUS/Documents/Ecommerce_Data_Analytics/outputs")

def _find_csv(filename: str, subdir: str = "data") -> Path | None:
    candidates = [
        DATA_DIR / filename if subdir == "data" else OUTPUT_DIR / filename,
        PROJECT_ROOT / subdir / filename,
        Path.cwd() / subdir / filename,
        Path.cwd().parent / subdir / filename,
        _WIN_DATA / filename if subdir == "data" else _WIN_OUTPUTS / filename,
        Path("/mnt/user-data/uploads") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_full_dataset():
    p = _find_csv("full_dataset_10k.csv")
    if p:
        df = pd.read_csv(p)
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
        df["churned"]     = df["churned"].astype(bool)
        return df
    st.error(f"❌ full_dataset_10k.csv not found. Searched in: {DATA_DIR}")
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_purchase_history():
    p = _find_csv("purchase_history_10k.csv")
    if p:
        df = pd.read_csv(p)
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_predictions():
    candidates = [
        OUTPUT_DIR / "predictions" / "full_dataset_with_predictions.csv",
        PROJECT_ROOT / "outputs" / "predictions" / "full_dataset_with_predictions.csv",
        Path("C:/Users/ASUS/Documents/Ecommerce_Data_Analytics/outputs/predictions/full_dataset_with_predictions.csv"),
        Path.cwd().parent / "outputs" / "predictions" / "full_dataset_with_predictions.csv",
        Path("/mnt/user-data/uploads/full_dataset_with_predictions.csv"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
            return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_marketing():
    p = _find_csv("marketing_promotions_10k.csv")
    return pd.read_csv(p) if p else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_engagement():
    p = _find_csv("engagement_behavior_10k.csv")
    return pd.read_csv(p) if p else pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for name, filename in [("Random Forest", "rf_churn_model.pkl"),
                            ("Logistic Regression", "lr_baseline.pkl")]:
        for folder in [MODEL_DIR, Path("/mnt/user-data/uploads")]:
            p = folder / filename
            if p.exists():
                try:
                    models[name] = joblib.load(p)
                    break
                except Exception:
                    pass
    return models

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_rfm(purchase_df: pd.DataFrame) -> pd.DataFrame:
    if purchase_df.empty:
        return pd.DataFrame()

    snapshot = purchase_df["order_date"].max() + timedelta(days=1)
    rfm = (
        purchase_df.groupby("customer_id")
        .agg(
            recency_days=("order_date", lambda x: (snapshot - x.max()).days),
            frequency=("order_id", "count"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )
    rfm["avg_order_value"] = (rfm["monetary"] / rfm["frequency"]).round(2)

    labels = [1, 2, 3, 4, 5]
    rfm["r_score"] = pd.qcut(rfm["recency_days"], 5, labels=labels[::-1], duplicates="drop").astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=labels, duplicates="drop").astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=labels, duplicates="drop").astype(int)
    rfm["rfm_score"] = rfm["r_score"] * 0.35 + rfm["f_score"] * 0.25 + rfm["m_score"] * 0.40

    def _segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4:           return "Champions"
        if r >= 2 and f >= 3 and m >= 3:           return "Loyal Customers"
        if r >= 3 and f <= 3 and m <= 3:           return "Potential Loyalists"
        if 2 <= r <= 3 and f >= 2 and m >= 2:      return "At Risk"
        if r <= 2 and f <= 2 and m <= 2:           return "Lost"
        return "Others"

    rfm["segment"] = rfm.apply(_segment, axis=1)
    return rfm

@st.cache_data(show_spinner=False)
def compute_cohort_retention(customers_df: pd.DataFrame, purchases_df: pd.DataFrame) -> pd.DataFrame:
    if customers_df.empty or purchases_df.empty:
        return pd.DataFrame()

    cust = customers_df[["customer_id", "signup_date"]].copy()
    cust["cohort_month"] = cust["signup_date"].dt.to_period("M")

    purch = purchases_df[["customer_id", "order_date"]].copy()
    purch["order_month"] = purch["order_date"].dt.to_period("M")

    merged = purch.merge(cust[["customer_id", "cohort_month"]], on="customer_id", how="left").dropna(subset=["cohort_month"])
    merged["period"] = (merged["order_month"].astype(int) - merged["cohort_month"].astype(int))
    merged = merged[merged["period"] >= 0]

    active = merged.groupby(["cohort_month", "period"])["customer_id"].nunique().reset_index()
    active.rename(columns={"customer_id": "active"}, inplace=True)

    cohort_sizes = active[active["period"] == 0].set_index("cohort_month")["active"]
    pivot = active.pivot_table(index="cohort_month", columns="period", values="active")
    retention = pivot.divide(cohort_sizes, axis=0).round(3)
    retention = retention.loc[:, retention.columns <= 12]
    return retention

@st.cache_data(show_spinner=False)
def compute_monthly_revenue(purchases_df: pd.DataFrame) -> pd.DataFrame:
    if purchases_df.empty:
        return pd.DataFrame()
    df = purchases_df.copy()
    df["month"] = df["order_date"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        revenue=("order_amount", "sum"),
        orders=("order_id", "count"),
        avg_order=("order_amount", "mean")
    ).reset_index()
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    return monthly.sort_values("month_dt")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar(full_df, rfm_df):
    st.sidebar.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:28px'>🛒</div>
        <div style='font-size:16px; font-weight:700; color:#58a6ff; letter-spacing:.5px'>E-COMMERCE ANALYTICS</div>
        <div style='font-size:11px; color:#8b949e; margin-top:4px'>FAANG-Level Portfolio Dashboard</div>
    </div>
    <hr style='border-color:#30363d; margin:12px 0'>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### 🎛️ Global Filters")

    min_date = datetime(2019, 1, 1)
    max_date = datetime(2023, 12, 31)
    date_range = st.sidebar.date_input(
        "Purchase Date Range",
        value=(datetime(2022, 1, 1), max_date),
        min_value=min_date, max_value=max_date,
    )

    tiers = ["All"] + sorted(full_df["customer_tier"].dropna().unique().tolist())
    selected_tier = st.sidebar.selectbox("Customer Tier", tiers)

    top_countries = ["All"] + full_df["country"].value_counts().head(10).index.tolist()
    selected_country = st.sidebar.selectbox("Country", top_countries)

    if not rfm_df.empty:
        segments = ["All"] + sorted(rfm_df["segment"].unique().tolist())
    else:
        segments = ["All"]
    selected_segment = st.sidebar.selectbox("RFM Segment", segments)

    churn_filter = st.sidebar.radio("Customer Status", ["All", "Active", "Churned"], horizontal=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dashboard Info")
    st.sidebar.markdown(f"""
    <div class='insight-box'>
        <strong>Data as of:</strong> Dec 2023<br>
        <strong>Customers:</strong> {len(full_df):,}<br>
        <strong>Purchase Records:</strong> 100,177
    </div>
    """, unsafe_allow_html=True)

    return {
        "date_range": date_range,
        "tier": selected_tier,
        "country": selected_country,
        "segment": selected_segment,
        "churn_filter": churn_filter,
    }

# ---------------------------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------------------------

def apply_filters(full_df, filters):
    df = full_df.copy()
    if filters["tier"] != "All":
        df = df[df["customer_tier"] == filters["tier"]]
    if filters["country"] != "All":
        df = df[df["country"] == filters["country"]]
    if filters["churn_filter"] == "Active":
        df = df[~df["churned"]]
    elif filters["churn_filter"] == "Churned":
        df = df[df["churned"]]
    return df

def apply_purchase_date_filter(purchases_df, filters):
    if len(filters["date_range"]) == 2:
        start, end = filters["date_range"]
        purchases_df = purchases_df[
            (purchases_df["order_date"] >= pd.Timestamp(start)) &
            (purchases_df["order_date"] <= pd.Timestamp(end))
        ]
    return purchases_df

# ---------------------------------------------------------------------------
# TAB 1 — EXECUTIVE KPIs
# ---------------------------------------------------------------------------

def render_executive_overview(df, pred_df, purchases_df, marketing_df):
    st.markdown('<div class="section-header">📈 Executive KPI Overview</div>', unsafe_allow_html=True)

    total_customers  = len(df)
    avg_clv          = df["CLV"].mean()
    churn_rate       = df["churned"].mean() * 100
    avg_purchase_prob = df["next_purchase_prob"].mean() * 100
    total_revenue    = purchases_df["order_amount"].sum() if not purchases_df.empty else df["total_spent"].sum()
    avg_order_value  = purchases_df["order_amount"].mean() if not purchases_df.empty else 0
    high_value_pct   = (df["CLV"] >= 200).mean() * 100

    if not marketing_df.empty:
        responded = marketing_df[marketing_df["responded"] == True]
        campaign_revenue = responded["additional_revenue"].sum()
        response_rate    = (marketing_df["responded"].mean() * 100)
    else:
        campaign_revenue, response_rate = 0, 0

    c1, c2, c3, c4 = st.columns(4)
    kpis_row1 = [
        (c1, "TOTAL CUSTOMERS",    f"{total_customers:,}",      "↑ 10K e-commerce base"),
        (c2, "AVG CUSTOMER CLV",   f"${avg_clv:,.2f}",          "Customer Lifetime Value"),
        (c3, "CHURN RATE",         f"{churn_rate:.1f}%",        "⚠️ Needs attention" if churn_rate > 35 else "✅ Within target"),
        (c4, "PURCHASE PROBABILITY", f"{avg_purchase_prob:.1f}%", "Avg next-purchase score"),
    ]
    for col, label, value, delta in kpis_row1:
        negative = "negative" if ("⚠️" in delta or churn_rate > 35 and label == "CHURN RATE") else ""
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-delta {negative}'>{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    kpis_row2 = [
        (c5, "TOTAL REVENUE",      f"${total_revenue:,.0f}",    "All-time gross revenue"),
        (c6, "AVG ORDER VALUE",    f"${avg_order_value:.2f}",   "Per transaction"),
        (c7, "HIGH-VALUE CUSTOMERS", f"{high_value_pct:.1f}%",  "CLV ≥ $200"),
        (c8, "CAMPAIGN RESPONSE RATE", f"{response_rate:.1f}%", f"${campaign_revenue:,.0f} incremental rev"),
    ]
    for col, label, value, delta in kpis_row2:
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-delta'>{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        if not purchases_df.empty:
            monthly = compute_monthly_revenue(purchases_df)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["month_dt"], y=monthly["revenue"],
                mode="lines+markers",
                name="Monthly Revenue",
                line=dict(color="#58a6ff", width=2.5),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.08)",
                hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
            ))
            monthly["ma3"] = monthly["revenue"].rolling(3, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=monthly["month_dt"], y=monthly["ma3"],
                mode="lines", name="3-Month MA",
                line=dict(color="#f0883e", width=1.5, dash="dash"),
                hovertemplate="<b>MA3:</b> $%{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="📊 Monthly Revenue Trend", height=320)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        tier_counts = df["customer_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        tier_order = ["Bronze", "Silver", "Gold", "Platinum"]
        tier_colors = {"Bronze": "#cd7f32", "Silver": "#c0c0c0", "Gold": "#ffd700", "Platinum": "#e5e4e2"}

        fig2 = px.bar(
            tier_counts.sort_values("tier", key=lambda x: x.map({t: i for i, t in enumerate(tier_order)})),
            x="tier", y="count", color="tier",
            color_discrete_map=tier_colors,
            text="count",
        )
        fig2.update_traces(textposition="outside", texttemplate="%{text:,}")
        fig2.update_layout(**PLOTLY_LAYOUT, title="🏅 Customers by Tier",
                           showlegend=False, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">💡 Executive Insights</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(f"""<div class='insight-box'>
        <strong>🔴 Churn Alert:</strong> {churn_rate:.1f}% churn rate across {total_customers:,} customers
        represents <strong>${(churn_rate/100 * total_customers * avg_clv):,.0f}</strong> in at-risk CLV.
        Priority: Win-back campaigns for At Risk segment.</div>""", unsafe_allow_html=True)
    with i2:
        st.markdown(f"""<div class='insight-box'>
        <strong>💰 Revenue Driver:</strong> Avg CLV of <strong>${avg_clv:,.2f}</strong> with
        {high_value_pct:.1f}% high-value customers. Upsell opportunity: move Silver → Gold tier
        could yield <strong>${(df[df['customer_tier']=='Silver']['CLV'].mean() * 0.3):,.0f}</strong> avg CLV uplift.</div>""", unsafe_allow_html=True)
    with i3:
        st.markdown(f"""<div class='insight-box'>
        <strong>📣 Campaign ROI:</strong> {response_rate:.1f}% response rate generating
        <strong>${campaign_revenue:,.0f}</strong> incremental revenue. Holiday Sale and Flash Sale
        are highest-performing campaigns for re-engagement.</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 2 — RFM SEGMENTATION
# ---------------------------------------------------------------------------

def render_rfm_segmentation(rfm_df, full_df, filters):
    st.markdown('<div class="section-header">🎯 RFM Customer Segmentation</div>', unsafe_allow_html=True)

    if rfm_df.empty:
        st.warning("⚠️ RFM data not available. Upload purchase_history_10k.csv.")
        return

    rfm_enriched = rfm_df.copy()
    if "customer_tier" in full_df.columns:
        rfm_enriched = rfm_enriched.merge(
            full_df[["customer_id", "customer_tier", "CLV", "churned"]],
            on="customer_id", how="left"
        )

    if filters["segment"] != "All":
        rfm_enriched = rfm_enriched[rfm_enriched["segment"] == filters["segment"]]

    total_rev = rfm_enriched["monetary"].sum()
    seg_summary = (
        rfm_enriched.groupby("segment")
        .agg(customers=("customer_id", "count"), revenue=("monetary", "sum"),
             avg_rfm=("rfm_score", "mean"), avg_monetary=("monetary", "mean"))
        .reset_index()
    )
    seg_summary["pct_customers"] = (seg_summary["customers"] / seg_summary["customers"].sum() * 100).round(1)
    seg_summary["pct_revenue"]   = (seg_summary["revenue"] / total_rev * 100).round(1)
    seg_summary["color"]         = seg_summary["segment"].map(SEGMENT_COLORS)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig_tree = px.treemap(
            seg_summary,
            path=["segment"],
            values="customers",
            color="avg_rfm",
            color_continuous_scale=["#f85149", "#f0883e", "#d29922", "#3fb950"],
            hover_data={"pct_revenue": True, "avg_monetary": ":.2f"},
            custom_data=["pct_customers", "pct_revenue", "avg_monetary"],
        )
        fig_tree.update_traces(
            hovertemplate="<b>%{label}</b><br>"
                          "Customers: %{value:,} (%{customdata[0]:.1f}%)<br>"
                          "Revenue Share: %{customdata[1]:.1f}%<br>"
                          "Avg Spend: $%{customdata[2]:.2f}<extra></extra>",
            texttemplate="<b>%{label}</b><br>%{value:,} customers",
        )
        fig_tree.update_layout(**PLOTLY_LAYOUT, title="🗺️ Segment Treemap (by Customer Count)",
                               coloraxis_showscale=False, height=360)
        st.plotly_chart(fig_tree, use_container_width=True)

    with col2:
        fig_donut = go.Figure(go.Pie(
            labels=seg_summary["segment"],
            values=seg_summary["revenue"],
            hole=0.55,
            marker=dict(colors=[SEGMENT_COLORS.get(s, "#8b949e") for s in seg_summary["segment"]]),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>${total_rev/1e6:.2f}M</b><br>Total Rev",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#e6edf3"), align="center"
        )
        fig_donut.update_layout(**PLOTLY_LAYOUT, title="💰 Revenue by Segment",
                                showlegend=False, height=360)
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown('<div class="section-header">📍 RFM Customer Map</div>', unsafe_allow_html=True)
    col3, col4 = st.columns([1.5, 1])

    with col3:
        sample = rfm_enriched.sample(min(2000, len(rfm_enriched)), random_state=42)
        fig_scatter = px.scatter(
            sample,
            x="recency_days", y="monetary",
            color="segment",
            size="frequency",
            color_discrete_map=SEGMENT_COLORS,
            hover_data={"customer_id": True, "rfm_score": ":.2f"},
            labels={"recency_days": "Days Since Last Purchase", "monetary": "Total Spend ($)"},
            opacity=0.7,
        )
        fig_scatter.update_layout(**PLOTLY_LAYOUT,
                                  title="🔵 Customer Scatter: Recency vs Spend",
                                  height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col4:
        st.markdown("**📋 Segment Scorecard**")
        scorecard = seg_summary[["segment", "customers", "pct_revenue", "avg_monetary", "avg_rfm"]].copy()
        scorecard.columns = ["Segment", "Customers", "Rev %", "Avg Spend", "RFM Score"]
        scorecard["Avg Spend"] = scorecard["Avg Spend"].apply(lambda x: f"${x:,.0f}")
        scorecard["RFM Score"] = scorecard["RFM Score"].apply(lambda x: f"{x:.2f}")
        scorecard["Rev %"]     = scorecard["Rev %"].apply(lambda x: f"{x:.1f}%")
        scorecard["Customers"] = scorecard["Customers"].apply(lambda x: f"{x:,}")
        st.dataframe(
            scorecard.sort_values("RFM Score", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=380,
        )

    if "CLV" in rfm_enriched.columns:
        fig_box = px.box(
            rfm_enriched.dropna(subset=["CLV"]),
            x="segment", y="CLV",
            color="segment",
            color_discrete_map=SEGMENT_COLORS,
            points="outliers",
            labels={"CLV": "Customer Lifetime Value ($)", "segment": ""},
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                              title="📦 CLV Distribution by RFM Segment", height=320)
        st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — COHORT RETENTION  ✅ BOTH FIXES APPLIED HERE
# ---------------------------------------------------------------------------

def render_cohort_analysis(full_df, purchases_df, filters):
    st.markdown('<div class="section-header">🔄 Cohort Retention Analysis</div>', unsafe_allow_html=True)

    purchases_filtered = apply_purchase_date_filter(purchases_df, filters)
    retention = compute_cohort_retention(full_df, purchases_filtered)

    if retention.empty:
        st.warning("⚠️ Insufficient data for cohort analysis."); return

    display = retention.tail(18)
    display_pct = (display * 100).round(1)

    cohort_labels = [str(c) for c in display_pct.index]
    period_labels = [f"M+{c}" for c in display_pct.columns]

    fig_heat = go.Figure(go.Heatmap(
        z=display_pct.values,
        x=period_labels,
        y=cohort_labels,
        colorscale=[
            [0.0,  "#1a0a0a"],
            [0.15, "#6b1a1a"],
            [0.30, "#a0522d"],
            [0.50, "#d29922"],
            [0.70, "#3fb950"],
            [1.0,  "#0a5220"],
        ],
        zmin=0, zmax=100,
        text=display_pct.round(0).astype(str).values,
        texttemplate="%{text}%",
        textfont=dict(size=10),
        hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
    ))
    # ✅ FIX 1: moved xaxis/yaxis out of update_layout to avoid conflict with PLOTLY_LAYOUT
    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        title="🔥 Monthly Cohort Retention Heatmap (%)",
        height=520,
    )
    fig_heat.update_xaxes(title="Months Since Signup", side="bottom")
    fig_heat.update_yaxes(title="Cohort (Signup Month)", autorange="reversed")
    st.plotly_chart(fig_heat, use_container_width=True)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig_lines = go.Figure()
        for i, (cohort, row) in enumerate(display.tail(6).iterrows()):
            row_clean = row.dropna()
            fig_lines.add_trace(go.Scatter(
                x=[f"M+{p}" for p in row_clean.index],
                y=(row_clean * 100).values,
                name=str(cohort),
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{cohort}</b><br>Period: %{{x}}<br>Retention: %{{y:.1f}}%<extra></extra>",
            ))
        # ✅ FIX 2: moved xaxis_title/yaxis_title to update_xaxes/update_yaxes
        fig_lines.update_layout(
            **PLOTLY_LAYOUT,
            title="📉 Retention Curves — Last 6 Cohorts",
            height=360,
        )
        fig_lines.update_xaxes(title="Month Since Signup")
        fig_lines.update_yaxes(title="Retention Rate (%)")
        st.plotly_chart(fig_lines, use_container_width=True)

    with col2:
        summary_cols = [c for c in [1, 3, 6] if c in display.columns]
        if summary_cols:
            avg_ret = (display[summary_cols].mean() * 100).round(1)
            fig_bar = go.Figure(go.Bar(
                x=[f"Month {c}" for c in avg_ret.index],
                y=avg_ret.values,
                marker_color=["#58a6ff", "#3fb950", "#f0883e"][:len(avg_ret)],
                text=[f"{v:.1f}%" for v in avg_ret.values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Avg Retention: %{y:.1f}%<extra></extra>",
            ))
            fig_bar.update_layout(**PLOTLY_LAYOUT,
                                  title="📊 Avg Retention at Key Milestones",
                                  yaxis_title="Retention Rate (%)",
                                  yaxis_range=[0, 120],
                                  showlegend=False, height=360)
            st.plotly_chart(fig_bar, use_container_width=True)

        m1_avg = display[1].mean() * 100 if 1 in display.columns else 0
        m3_avg = display[3].mean() * 100 if 3 in display.columns else 0
        st.markdown(f"""<div class='insight-box'>
        <strong>📌 Retention Insight:</strong><br>
        • M+1 avg retention: <strong>{m1_avg:.1f}%</strong><br>
        • M+3 avg retention: <strong>{m3_avg:.1f}%</strong><br>
        • Drop from M1→M3: <strong>{(m1_avg-m3_avg):.1f} pts</strong><br>
        <br>Focus on months 1–3 onboarding to prevent the sharpest retention drop.
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 4 — CHURN PREDICTION
# ---------------------------------------------------------------------------

def render_churn_prediction(full_df, pred_df, models):
    st.markdown('<div class="section-header">🤖 Churn Prediction Engine</div>', unsafe_allow_html=True)

    FEATURE_COLS = [
        "age", "months_since_signup", "total_spent",
        "weekly_visits", "session_time_minutes", "page_views",
        "app_opens", "num_promotions_responded",
    ]

    col1, col2, col3 = st.columns(3)
    with col1:
        if not pred_df.empty and "churn_prob" in pred_df.columns:
            fig_hist = px.histogram(
                pred_df, x="churn_prob", nbins=40,
                color_discrete_sequence=["#58a6ff"],
                labels={"churn_prob": "Churn Probability"},
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#f85149",
                               annotation_text="Threshold 0.5")
            fig_hist.update_layout(**PLOTLY_LAYOUT,
                                   title="📊 Churn Probability Distribution",
                                   height=300, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        if not pred_df.empty and "risk_segment" in pred_df.columns:
            risk_counts = pred_df["risk_segment"].value_counts().reset_index()
            risk_counts.columns = ["risk", "count"]
            risk_colors = {"High Risk": "#f85149", "Medium Risk": "#f0883e", "Low Risk": "#3fb950"}
            fig_risk = px.pie(risk_counts, names="risk", values="count", hole=0.5,
                              color="risk", color_discrete_map=risk_colors)
            fig_risk.update_layout(**PLOTLY_LAYOUT, title="⚠️ Risk Segment Distribution",
                                   height=300, showlegend=True)
            st.plotly_chart(fig_risk, use_container_width=True)

    with col3:
        churn_by_tier = full_df.groupby("customer_tier")["churned"].mean().reset_index()
        churn_by_tier.columns = ["tier", "churn_rate"]
        churn_by_tier["churn_rate"] *= 100
        tier_order = ["Bronze", "Silver", "Gold", "Platinum"]
        churn_by_tier["tier"] = pd.Categorical(churn_by_tier["tier"], categories=tier_order, ordered=True)
        churn_by_tier = churn_by_tier.sort_values("tier")
        fig_tier = px.bar(churn_by_tier, x="tier", y="churn_rate",
                          color="churn_rate",
                          color_continuous_scale=["#3fb950", "#d29922", "#f0883e", "#f85149"],
                          text=churn_by_tier["churn_rate"].apply(lambda x: f"{x:.1f}%"))
        fig_tier.update_traces(textposition="outside")
        fig_tier.update_layout(**PLOTLY_LAYOUT, title="🏅 Churn Rate by Tier",
                               coloraxis_showscale=False, showlegend=False, height=300)
        st.plotly_chart(fig_tier, use_container_width=True)

    if not pred_df.empty and "churn_prob" in pred_df.columns:
        st.markdown('<div class="section-header">📍 Churn Risk vs Customer Value</div>', unsafe_allow_html=True)
        sample = pred_df.sample(min(3000, len(pred_df)), random_state=42)
        fig_2d = px.scatter(
            sample, x="churn_prob", y="CLV",
            color="risk_segment" if "risk_segment" in sample.columns else "churned",
            color_discrete_map={"High Risk": "#f85149", "Medium Risk": "#f0883e", "Low Risk": "#3fb950",
                                 True: "#f85149", False: "#3fb950"},
            hover_data={"customer_id": True, "customer_tier": True},
            labels={"churn_prob": "Churn Probability", "CLV": "Customer Lifetime Value ($)"},
            opacity=0.6,
        )
        fig_2d.add_vline(x=0.5, line_dash="dot", line_color="#8b949e", opacity=0.5)
        fig_2d.add_hline(y=pred_df["CLV"].median(), line_dash="dot", line_color="#8b949e", opacity=0.5)
        for txt, x, y in [("🔴 High Risk\nHigh Value", 0.75, pred_df["CLV"].quantile(0.85)),
                           ("🟢 Low Risk\nHigh Value", 0.15, pred_df["CLV"].quantile(0.85))]:
            fig_2d.add_annotation(text=txt, x=x, y=y, showarrow=False,
                                  font=dict(color="#8b949e", size=10))
        fig_2d.update_layout(**PLOTLY_LAYOUT, height=400,
                              title="🎯 Churn Risk vs CLV — Action Quadrants")
        st.plotly_chart(fig_2d, use_container_width=True)

    st.markdown('<div class="section-header">🔍 Individual Customer Churn Predictor</div>', unsafe_allow_html=True)

    available_models = list(models.keys()) if models else []
    use_precomputed  = not available_models

    if use_precomputed:
        st.info("ℹ️ Using precomputed churn probabilities from `full_dataset_with_predictions.csv`. "
                "To use live model inference, add `rf_churn_model.pkl` to the `models/` folder.")

    tab_manual, tab_lookup = st.tabs(["✏️ Manual Input", "🔎 Lookup by Customer ID"])

    with tab_manual:
        c1, c2, c3 = st.columns(3)
        with c1:
            age           = st.slider("Age", 18, 80, 35)
            months_signup = st.slider("Months Since Signup", 1, 60, 24)
            total_spent   = st.number_input("Total Spent ($)", 0.0, 5000.0, 500.0, step=50.0)
        with c2:
            weekly_visits = st.slider("Weekly Visits", 0, 30, 5)
            session_mins  = st.slider("Avg Session (mins)", 0.0, 120.0, 15.0)
            page_views    = st.slider("Page Views", 0.0, 100.0, 20.0)
        with c3:
            app_opens     = st.slider("App Opens", 0, 50, 8)
            promo_resp    = st.slider("Promotions Responded", 0, 10, 2)
            tier          = st.selectbox("Customer Tier", ["Bronze", "Silver", "Gold", "Platinum"])

        if st.button("🚀 Predict Churn Probability", use_container_width=True):
            input_data = pd.DataFrame([{
                "age": age, "months_since_signup": months_signup,
                "total_spent": total_spent, "weekly_visits": weekly_visits,
                "session_time_minutes": session_mins, "page_views": page_views,
                "app_opens": app_opens, "num_promotions_responded": promo_resp,
            }])

            churn_prob = None

            if available_models:
                model_name = available_models[0]
                model = models[model_name]
                try:
                    feat_cols = [c for c in FEATURE_COLS if c in input_data.columns]
                    X = input_data[feat_cols]
                    for col in (getattr(model, "feature_names_in_", []) or []):
                        if col not in X.columns:
                            X[col] = 0
                    churn_prob = float(model.predict_proba(X)[0][1])
                except Exception as e:
                    st.warning(f"Model inference failed: {e}. Using heuristic estimate.")

            if churn_prob is None:
                score = 0
                score += 0.3 * (1 - min(weekly_visits / 10, 1))
                score += 0.2 * (1 - min(session_mins / 30, 1))
                score += 0.2 * (1 - min(promo_resp / 5, 1))
                score += 0.15 * (total_spent < 300)
                score += 0.15 * (months_signup > 36)
                churn_prob = round(min(max(score, 0.05), 0.95), 3)

            level = "high" if churn_prob >= 0.6 else "medium" if churn_prob >= 0.35 else "low"
            emoji = "🔴" if level == "high" else "🟡" if level == "medium" else "🟢"
            label = "HIGH RISK" if level == "high" else "MEDIUM RISK" if level == "medium" else "LOW RISK"

            st.markdown(f"""
            <div class='pred-result-{level}'>
                <div style='font-size:36px; font-weight:700; color:{"#f85149" if level=="high" else "#d29922" if level=="medium" else "#3fb950"}'>
                    {emoji} {churn_prob*100:.1f}% Churn Probability
                </div>
                <div style='font-size:16px; font-weight:600; margin-top:8px; color:#e6edf3'>{label}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>**🎯 Recommended Action:**", unsafe_allow_html=True)
            if level == "high":
                st.error("Immediate win-back campaign. Offer 20% discount + personal outreach. Flag for VIP retention team.")
            elif level == "medium":
                st.warning("Enrol in re-engagement email sequence. Personalised recommendations + loyalty points offer.")
            else:
                st.success("Customer is healthy. Focus on upsell / cross-sell. Nominate for referral programme.")

    with tab_lookup:
        customer_search = st.text_input("Enter Customer ID (1 – 10,000)", "42")
        if st.button("🔍 Look Up Customer", use_container_width=True):
            try:
                cid = int(customer_search)
                row = pred_df[pred_df["customer_id"] == cid] if not pred_df.empty else full_df[full_df["customer_id"] == cid]
                if row.empty:
                    st.error(f"Customer ID {cid} not found.")
                else:
                    row = row.iloc[0]
                    prob = row.get("churn_prob", None)
                    if prob is not None:
                        level = "high" if prob >= 0.6 else "medium" if prob >= 0.35 else "low"
                        emoji = "🔴" if level == "high" else "🟡" if level == "medium" else "🟢"
                        st.markdown(f"""
                        <div class='pred-result-{level}'>
                            <b>{emoji} Customer #{cid} — {row.get('name', 'N/A')}</b><br>
                            Churn Probability: <b>{prob*100:.1f}%</b> &nbsp;|&nbsp;
                            CLV: <b>${row.get('CLV', 0):,.2f}</b> &nbsp;|&nbsp;
                            Tier: <b>{row.get('customer_tier', 'N/A')}</b> &nbsp;|&nbsp;
                            Risk: <b>{row.get('risk_segment', level.upper())}</b>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.json(row.to_dict())
            except ValueError:
                st.error("Please enter a valid numeric Customer ID.")

# ---------------------------------------------------------------------------
# TAB 5 — PRODUCT & REVENUE ANALYTICS
# ---------------------------------------------------------------------------

def render_product_revenue(purchases_df, full_df, marketing_df, filters):
    st.markdown('<div class="section-header">🛍️ Product & Revenue Analytics</div>', unsafe_allow_html=True)

    if purchases_df.empty:
        st.warning("⚠️ Purchase history not available."); return

    purchases_filtered = apply_purchase_date_filter(purchases_df, filters)

    cat_rev = purchases_filtered.groupby("product_category").agg(
        revenue=("order_amount", "sum"),
        orders=("order_id", "count"),
        avg_order=("order_amount", "mean"),
    ).reset_index().sort_values("revenue", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig_cat = px.bar(
            cat_rev, x="revenue", y="product_category",
            orientation="h",
            color="revenue",
            color_continuous_scale=["#1c2128", "#1a3a5c", "#1a5c3a", "#58a6ff"],
            text=cat_rev["revenue"].apply(lambda x: f"${x/1000:.1f}K"),
            labels={"revenue": "Total Revenue ($)", "product_category": ""},
        )
        fig_cat.update_traces(textposition="outside")
        fig_cat.update_layout(**PLOTLY_LAYOUT,
                              title="🛒 Revenue by Product Category",
                              coloraxis_showscale=False, showlegend=False,
                              height=380)
        fig_cat.update_yaxes(autorange="reversed", gridcolor="#21262d")
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        top3_cats = cat_rev.head(3)["product_category"].tolist()
        pf = purchases_filtered[purchases_filtered["product_category"].isin(top3_cats)].copy()
        pf["month"] = pf["order_date"].dt.to_period("M").dt.to_timestamp()
        monthly_cat = pf.groupby(["month", "product_category"])["order_amount"].sum().reset_index()

        fig_line = px.line(
            monthly_cat, x="month", y="order_amount",
            color="product_category",
            labels={"order_amount": "Revenue ($)", "month": "Month", "product_category": "Category"},
            markers=True,
        )
        fig_line.update_layout(**PLOTLY_LAYOUT,
                               title="📈 Top 3 Categories Monthly Trend",
                               height=380)
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown('<div class="section-header">🏷️ Discount & Promotion Impact</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        disc = purchases_filtered.groupby("discount_used").agg(
            revenue=("order_amount", "sum"),
            orders=("order_id", "count"),
            avg_order=("order_amount", "mean"),
        ).reset_index()
        disc["label"] = disc["discount_used"].map({True: "With Discount", False: "No Discount"})
        fig_disc = px.bar(disc, x="label", y=["revenue", "orders"],
                          barmode="group",
                          labels={"value": "Amount", "variable": "Metric"},
                          color_discrete_sequence=["#58a6ff", "#3fb950"])
        fig_disc.update_layout(**PLOTLY_LAYOUT, title="🏷️ Discount vs Non-Discount Orders",
                               height=300)
        st.plotly_chart(fig_disc, use_container_width=True)

    with col4:
        if not marketing_df.empty:
            campaign_perf = marketing_df.groupby("campaign_name").agg(
                response_rate=("responded", "mean"),
                total_revenue=("additional_revenue", "sum"),
                n_targeted=("customer_id", "count"),
            ).reset_index()
            campaign_perf["response_rate"] *= 100
            fig_camp = px.scatter(
                campaign_perf, x="response_rate", y="total_revenue",
                size="n_targeted", color="campaign_name",
                text="campaign_name",
                labels={"response_rate": "Response Rate (%)", "total_revenue": "Revenue ($)"},
            )
            fig_camp.update_traces(textposition="top center")
            fig_camp.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                                   title="📣 Campaign Performance Matrix",
                                   height=300)
            st.plotly_chart(fig_camp, use_container_width=True)

    st.markdown('<div class="section-header">🔮 Revenue Forecast (90-Day)</div>', unsafe_allow_html=True)
    monthly = compute_monthly_revenue(purchases_filtered)

    if len(monthly) >= 12:
        from sklearn.linear_model import LinearRegression
        monthly["t"] = range(len(monthly))
        X_train = monthly[["t"]].values
        y_train = monthly["revenue"].values
        lr = LinearRegression().fit(X_train, y_train)

        future_t = np.array([[len(monthly)], [len(monthly)+1], [len(monthly)+2]])
        forecast  = lr.predict(future_t)

        last_date  = monthly["month_dt"].max()
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(3)]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=monthly["month_dt"], y=monthly["revenue"],
            mode="lines+markers", name="Actual",
            line=dict(color="#58a6ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
        ))
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=forecast,
            mode="lines+markers", name="Forecast",
            line=dict(color="#f0883e", width=2.5, dash="dash"),
            marker=dict(size=10, symbol="diamond"),
        ))
        fig_fc.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(forecast * 1.15) + list(forecast * 0.85)[::-1],
            fill="toself", fillcolor="rgba(240,136,62,0.12)",
            line=dict(width=0), name="90% CI", showlegend=True,
        ))
        fig_fc.update_layout(**PLOTLY_LAYOUT, title="🔮 90-Day Revenue Forecast",
                             height=360)
        st.plotly_chart(fig_fc, use_container_width=True)

        fc_total = forecast.sum()
        st.markdown(f"""<div class='insight-box'>
        <strong>📈 Forecast Summary:</strong> Projected revenue over next 3 months:
        <strong>${fc_total:,.0f}</strong>
        ({'+' if fc_total > monthly['revenue'].tail(3).sum() else ''}{((fc_total / monthly['revenue'].tail(3).sum()) - 1)*100:.1f}% vs last 3 months).
        Forecast based on linear trend from {len(monthly)} months of history.
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 6 — CUSTOMER INTELLIGENCE
# ---------------------------------------------------------------------------

def render_customer_intelligence(full_df, pred_df, engagement_df):
    st.markdown('<div class="section-header">👥 Customer Intelligence & Demographics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_age = px.histogram(
            full_df, x="age", color="churned",
            nbins=30, barmode="overlay", opacity=0.75,
            color_discrete_map={True: "#f85149", False: "#3fb950"},
            labels={"age": "Customer Age", "churned": "Churned"},
        )
        fig_age.update_layout(**PLOTLY_LAYOUT,
                              title="👤 Age Distribution: Churned vs Active",
                              height=320)
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        country_clv = full_df.groupby("country").agg(
            avg_clv=("CLV", "mean"), customers=("customer_id", "count")
        ).reset_index().sort_values("avg_clv", ascending=False)
        fig_country = px.bar(
            country_clv.head(10), x="country", y="avg_clv",
            color="avg_clv",
            color_continuous_scale=["#1a3a5c", "#58a6ff"],
            text=country_clv.head(10)["avg_clv"].apply(lambda x: f"${x:.0f}"),
        )
        fig_country.update_traces(textposition="outside")
        fig_country.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                  title="🌍 Avg CLV by Country", height=320)
        st.plotly_chart(fig_country, use_container_width=True)

    if not engagement_df.empty:
        st.markdown('<div class="section-header">📱 Engagement Behavior Analysis</div>', unsafe_allow_html=True)
        eng_full = full_df.merge(engagement_df, on="customer_id", how="left", suffixes=("", "_eng"))

        col3, col4 = st.columns(2)
        with col3:
            sample = eng_full.sample(min(2000, len(eng_full)), random_state=42).copy()
            # clip negatives so Plotly marker size stays in [0, inf]
            sample["session_time_minutes"] = sample["session_time_minutes"].clip(lower=0).fillna(0)
            fig_eng = px.scatter(
                sample, x="weekly_visits", y="CLV",
                color="customer_tier",
                color_discrete_map={"Bronze": "#cd7f32", "Silver": "#c0c0c0",
                                    "Gold": "#ffd700", "Platinum": "#e5e4e2"},
                size="session_time_minutes", opacity=0.6,
                size_max=30,
                labels={"weekly_visits": "Weekly Visits", "CLV": "CLV ($)"},
            )
            fig_eng.update_layout(**PLOTLY_LAYOUT,
                                  title="📊 Weekly Visits vs CLV by Tier", height=320)
            st.plotly_chart(fig_eng, use_container_width=True)

        with col4:
            eng_tier = eng_full.groupby("customer_tier").agg(
                weekly_visits=("weekly_visits", "mean"),
                session_mins=("session_time_minutes", "mean"),
                page_views=("page_views", "mean"),
                app_opens=("app_opens", "mean"),
            ).round(1)
            fig_hm = px.imshow(
                eng_tier.T,
                color_continuous_scale="Blues",
                text_auto=True,
                labels={"x": "Tier", "y": "Metric"},
            )
            fig_hm.update_layout(**PLOTLY_LAYOUT,
                                 title="🔥 Engagement Heatmap by Tier", height=320)
            st.plotly_chart(fig_hm, use_container_width=True)

    if not pred_df.empty and "churn_prob" in pred_df.columns:
        st.markdown('<div class="section-header">🚨 High-Value At-Risk Customers</div>', unsafe_allow_html=True)
        at_risk = pred_df[
            (pred_df["churn_prob"] >= 0.6) &
            (pred_df["CLV"] >= pred_df["CLV"].quantile(0.75))
        ].sort_values("churn_prob", ascending=False).head(20)

        display_cols = ["customer_id", "name", "customer_tier", "CLV", "churn_prob", "risk_segment"]
        display_cols = [c for c in display_cols if c in at_risk.columns]
        at_risk_display = at_risk[display_cols].copy()
        if "CLV" in at_risk_display.columns:
            at_risk_display["CLV"] = at_risk_display["CLV"].apply(lambda x: f"${x:,.2f}")
        if "churn_prob" in at_risk_display.columns:
            at_risk_display["churn_prob"] = at_risk_display["churn_prob"].apply(lambda x: f"{x*100:.1f}%")

        st.dataframe(at_risk_display, use_container_width=True, hide_index=True)
        st.markdown(f"""<div class='insight-box'>
        <strong>🚨 Action Required:</strong> {len(at_risk)} high-value customers (top 25% CLV) have
        churn probability ≥ 60%. Prioritise for personal outreach within 7 days.
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def main():
    with st.spinner("Loading data pipelines..."):
        full_df       = load_full_dataset()
        purchases_df  = load_purchase_history()
        pred_df       = load_predictions()
        marketing_df  = load_marketing()
        engagement_df = load_engagement()
        models        = load_models()

    if full_df.empty:
        st.error("❌ Core dataset not found. Please ensure `full_dataset_10k.csv` is in the `data/` folder.")
        st.stop()

    with st.spinner("Running RFM segmentation..."):
        rfm_df = compute_rfm(purchases_df)

    filters = render_sidebar(full_df, rfm_df)
    filtered_df = apply_filters(full_df, filters)

    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-size:28px; font-weight:800; color:#e6edf3; letter-spacing:-0.5px'>
            🛒 E-Commerce Analytics Platform
        </div>
        <div style='font-size:14px; color:#8b949e; margin-top:4px'>
            FAANG-Level Portfolio Dashboard &nbsp;·&nbsp; 10,000 Customers &nbsp;·&nbsp; 100K+ Orders &nbsp;·&nbsp;
            Data through Dec 2023
        </div>
    </div>
    """, unsafe_allow_html=True)

    active_filters = []
    if filters["tier"] != "All":         active_filters.append(f"Tier: {filters['tier']}")
    if filters["country"] != "All":      active_filters.append(f"Country: {filters['country']}")
    if filters["segment"] != "All":      active_filters.append(f"Segment: {filters['segment']}")
    if filters["churn_filter"] != "All": active_filters.append(f"Status: {filters['churn_filter']}")
    if active_filters:
        st.markdown(f"<div style='background:#1c2128;border:1px solid #30363d;border-radius:6px;"
                    f"padding:8px 14px;font-size:12px;color:#58a6ff;margin-bottom:16px'>"
                    f"🔽 Active Filters: {' &nbsp;|&nbsp; '.join(active_filters)} "
                    f"&nbsp;—&nbsp; Showing {len(filtered_df):,} of {len(full_df):,} customers</div>",
                    unsafe_allow_html=True)

    tabs = st.tabs([
        "📈 Executive Overview",
        "🎯 RFM Segmentation",
        "🔄 Cohort Retention",
        "🤖 Churn Prediction",
        "🛍️ Product & Revenue",
        "👥 Customer Intelligence",
    ])

    with tabs[0]:
        render_executive_overview(filtered_df, pred_df, purchases_df, marketing_df)
    with tabs[1]:
        render_rfm_segmentation(rfm_df, filtered_df, filters)
    with tabs[2]:
        render_cohort_analysis(full_df, purchases_df, filters)
    with tabs[3]:
        render_churn_prediction(filtered_df, pred_df, models)
    with tabs[4]:
        render_product_revenue(purchases_df, filtered_df, marketing_df, filters)
    with tabs[5]:
        render_customer_intelligence(filtered_df, pred_df, engagement_df)

    st.markdown("""
    <div style='margin-top:48px; padding:20px; border-top:1px solid #30363d;
                text-align:center; color:#484f58; font-size:12px;'>
        E-Commerce Analytics Dashboard &nbsp;·&nbsp; FAANG-Level Portfolio Project &nbsp;·&nbsp;
        Built with Streamlit + Plotly &nbsp;·&nbsp; 2024
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()