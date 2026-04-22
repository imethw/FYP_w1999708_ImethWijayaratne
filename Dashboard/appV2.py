import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: #8b949e !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3fb950;
    margin-bottom: 0.3rem;
}
.player-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.player-card:hover { border-color: #3fb950; }
.player-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e6edf3;
}
.player-meta { font-size: 0.82rem; color: #8b949e; margin-top: 0.2rem; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-right: 4px;
}
.badge-pos { background: #1f3a24; color: #3fb950; }
.badge-atk { background: #3a1f1f; color: #f85149; }
.badge-mid { background: #1f2a3a; color: #58a6ff; }
.badge-def { background: #2a2a1f; color: #e3b341; }
.badge-gk  { background: #2a1f3a; color: #bc8cff; }
.badge-gem { background: #1a2a1f; color: #3fb950; }
.insight-box {
    background: #161b22;
    border-left: 3px solid #3fb950;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0 1rem;
    font-size: 0.85rem;
    color: #8b949e;
}
.divider { border: none; border-top: 1px solid #21262d; margin: 1.5rem 0; }
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e6edf3;
    margin-bottom: 0.2rem;
}
.page-sub { font-size: 0.9rem; color: #8b949e; margin-bottom: 1.5rem; }
.main .block-container { background-color: #0d1117; padding-top: 2rem; }
body { background-color: #0d1117; }
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
GROUP_COLORS = {
    "Attacker":   "#f85149",
    "Midfielder": "#58a6ff",
    "Defender":   "#e3b341",
    "Goalkeeper": "#bc8cff",
}
BADGE_CLASS = {
    "Attacker":   "badge-atk",
    "Midfielder": "badge-mid",
    "Defender":   "badge-def",
    "Goalkeeper": "badge-gk",
}
CLUSTER_COLORS = ["#3fb950", "#58a6ff", "#f85149", "#e3b341", "#bc8cff",
                  "#d2a8ff", "#ffa657", "#79c0ff"]

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#8b949e",
    legend=dict(font=dict(color="#8b949e")),
)
AXIS_STYLE = dict(gridcolor="#21262d", color="#8b949e")

# ── Position weights (must match notebook exactly) ─────────────────────────────
POSITION_WEIGHTS = {
    "Attacker":   (0.60, 0.40),
    "Midfielder": (0.65, 0.35),
    "Defender":   (0.70, 0.30),
    "Goalkeeper": (0.75, 0.25),
}

# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """ Loads cleaned_player_data.csv — exported by FYP_NEWCode_V2.ipynb.
    The notebook already computes:
        performance_score, value_score, wage_efficiency, position_group,
        primary_position, value_gbp, wage_gbp, etc.
    We only add display-friendly scaled columns here."""
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "cleaned_player_data.csv"))

    # Convenience display columns
    df["value_gbp_m"] = df["value_gbp"] / 1_000_000
    df["wage_gbp_k"]  = df["wage_gbp"]  / 1_000
    # Round computed scores for display
    df["value_score"]       = df["value_score"].round(2)
    df["performance_score"] = df["performance_score"].round(2)
    if "wage_efficiency" in df.columns:
        df["wage_efficiency"] = df["wage_efficiency"].round(2)

    return df


@st.cache_data
def run_clustering(df_hash, n_clusters):
    """
    K-Means clustering using the same 6 features as the notebook:
        Age, Overall, Potential, wage_gbp, value_gbp, performance_score
    """
    df = load_data()

    # Mirror notebook cluster features
    features = ["Age", "Overall", "Potential", "wage_gbp", "value_gbp", "performance_score"]
    X = df[features].dropna()

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)

    df_c = df.loc[X.index].copy()
    df_c["cluster"] = labels

    centers = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=features
    )

    # Auto-label clusters
    med_vfm = df_c.groupby("cluster")["value_score"].median()
    names   = {}
    for i, row in centers.iterrows():
        cluster_med_vfm = med_vfm.get(i, 0)
        if row["value_gbp"] > 30_000_000 and row["Overall"] > 82:
            names[i] = "Elite"
        elif cluster_med_vfm > med_vfm.median() and row["Overall"] < 80:
            names[i] = "Hidden Gem"
        elif row["Potential"] - row["Overall"] > 2:
            names[i] = "High Potential"
        elif row["Overall"] > 78:
            names[i] = "Established"
        else:
            names[i] = "Squad Depth"

    df_c["cluster_label"] = df_c["cluster"].map(names)
    return df_c, centers, names


df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚽ Football Analytics")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.8rem;margin-bottom:1.5rem'>"
        "FYP · Imeth Wijayaratne · 20220336</div>",
        unsafe_allow_html=True
    )
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 Player Analysis", "💰 Value for Money",
         "🏟️ Team Builder", "🤖 Player Clustering", "🌟 Young Talent Finder"],
        label_visibility="collapsed"
    )
    st.markdown("<hr style='border-color:#21262d'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#8b949e;font-size:0.75rem'>"
        "FIFA 23 Dataset · Kaggle<br>"
        f"{len(df):,} players · 31 attributes</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="page-title">Player Performance Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Team selection & cost optimisation · FIFA 23 dataset</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Players",     f"{len(df):,}")
    c2.metric("Avg Overall",       f"{df['Overall'].mean():.1f}")
    c3.metric("Avg Market Value",  f"£{df['value_gbp_m'].mean():.1f}M")
    c4.metric("Avg Perf Score",    f"{df['performance_score'].mean():.1f}")
    c5.metric("Avg VFM Score",     f"{df['value_score'].mean():.1f}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Position group summary bar
    st.markdown('<div class="section-label">Position group summary</div>', unsafe_allow_html=True)
    summary = df.groupby("position_group").agg(
        Players=("Name", "count"),
        Avg_Overall=("Overall", "mean"),
        Avg_Potential=("Potential", "mean"),
        Avg_Perf=("performance_score", "mean"),
        Avg_Value_M=("value_gbp_m", "mean"),
        Avg_VFM=("value_score", "mean"),
    ).round(2).reset_index()
    summary.columns = ["Group", "Players", "Avg Overall", "Avg Potential",
                        "Avg Perf Score", "Avg Value (£M)", "Avg VFM Score"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="section-label">Players per position group</div>', unsafe_allow_html=True)
        counts = df["position_group"].value_counts().reset_index()
        counts.columns = ["Position Group", "Count"]
        fig = px.pie(counts, names="Position Group", values="Count",
                     color="Position Group", color_discrete_map=GROUP_COLORS, hole=0.55)
        fig.update_layout(**PLOT_LAYOUT, margin=dict(t=10, b=10))
        fig.update_traces(textinfo="percent+label", textfont_color="#e6edf3")
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="section-label">Overall rating distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="Overall", color="position_group",
                           color_discrete_map=GROUP_COLORS, barmode="overlay",
                           nbins=30, opacity=0.75)
        fig.update_layout(**PLOT_LAYOUT, margin=dict(t=10, b=10),
                          xaxis=AXIS_STYLE, yaxis=AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Performance score vs market value — all positions</div>', unsafe_allow_html=True)
    fig = px.scatter(df, x="value_gbp_m", y="performance_score",
                     color="position_group", color_discrete_map=GROUP_COLORS,
                     hover_data=["Name", "Club", "Overall", "Potential", "primary_position"],
                     labels={"value_gbp_m": "Market Value (£M)",
                             "performance_score": "Performance Score",
                             "position_group": "Group"},
                     opacity=0.65)
    fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Average performance score by position group — bar
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Avg performance & VFM score by position group</div>', unsafe_allow_html=True)
    perf_bar = df.groupby("position_group")[["performance_score", "value_score"]].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Avg Perf Score", x=perf_bar["position_group"],
                         y=perf_bar["performance_score"], marker_color="#3fb950"))
    fig.add_trace(go.Bar(name="Avg VFM Score",  x=perf_bar["position_group"],
                         y=perf_bar["value_score"],       marker_color="#58a6ff"))
    fig.update_layout(**PLOT_LAYOUT, barmode="group",
                      xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=340, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PLAYER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Player Analysis":
    st.markdown('<div class="page-title">Player Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Search, filter and compare individual players</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        groups = st.multiselect("Position group", df["position_group"].unique().tolist(),
                                default=df["position_group"].unique().tolist())
    with fc2:
        age_range = st.slider("Age range", int(df["Age"].min()), int(df["Age"].max()), (18, 30))
    with fc3:
        search = st.text_input("Search player name", placeholder="e.g. Salah")

    filtered = df[df["position_group"].isin(groups) & df["Age"].between(*age_range)]
    if search:
        filtered = filtered[filtered["Name"].str.contains(search, case=False, na=False)]

    st.markdown(
        f"<div style='color:#8b949e;font-size:0.82rem;margin:0.5rem 0 1rem'>"
        f"{len(filtered):,} players match your filters</div>",
        unsafe_allow_html=True
    )

    col_t, col_b = st.columns([3, 2])
    with col_t:
        st.markdown('<div class="section-label">Performance vs market value</div>', unsafe_allow_html=True)
        fig = px.scatter(filtered, x="value_gbp_m", y="performance_score",
                         color="position_group", color_discrete_map=GROUP_COLORS,
                         size="Overall", size_max=18,
                         hover_data=["Name", "Club", "Age", "Overall", "Potential",
                                     "primary_position", "wage_efficiency"] if "wage_efficiency" in filtered.columns
                                    else ["Name", "Club", "Age", "Overall", "Potential", "primary_position"],
                         labels={"value_gbp_m": "Market Value (£M)",
                                 "performance_score": "Performance Score",
                                 "position_group": "Group"},
                         opacity=0.75)
        fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-label">Top 10 by overall rating</div>', unsafe_allow_html=True)
        for _, row in filtered.nlargest(10, "Overall").iterrows():
            bc = BADGE_CLASS.get(row["position_group"], "badge-pos")
            st.markdown(f"""
            <div class="player-card">
                <div class="player-name">{row['Name']}</div>
                <div class="player-meta">{row['Club']} · Age {row['Age']}</div>
                <div style="margin-top:0.5rem">
                    <span class="badge {bc}">{row['primary_position']}</span>
                    <span class="badge badge-pos">OVR {row['Overall']}</span>
                    <span class="badge badge-pos">POT {row['Potential']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Compare two players</div>', unsafe_allow_html=True)
    player_names = sorted(filtered["Name"].unique().tolist())
    cp1, cp2 = st.columns(2)
    p1 = cp1.selectbox("Player A", player_names, index=0)
    p2 = cp2.selectbox("Player B", player_names, index=min(1, len(player_names) - 1))

    if p1 and p2 and p1 != p2:
        d1 = filtered[filtered["Name"] == p1].iloc[0]
        d2 = filtered[filtered["Name"] == p2].iloc[0]
        metrics = ["Overall", "Potential", "performance_score", "value_gbp_m", "value_score", "Age"]
        labels  = ["Overall", "Potential", "Perf Score", "Value (£M)", "VFM Score", "Age"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name=p1, x=labels, y=[d1[m] for m in metrics], marker_color="#3fb950"))
        fig.add_trace(go.Bar(name=p2, x=labels, y=[d2[m] for m in metrics], marker_color="#58a6ff"))
        fig.update_layout(**PLOT_LAYOUT, barmode="group",
                          xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=350, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Full player table</div>', unsafe_allow_html=True)
    display_cols = ["Name", "Age", "Nationality", "Club", "Position", "primary_position",
                    "position_group", "Overall", "Potential", "performance_score",
                    "value_gbp_m", "value_score"]
    if "wage_efficiency" in filtered.columns:
        display_cols.append("wage_efficiency")
    st.dataframe(
        filtered[display_cols].sort_values("Overall", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VALUE FOR MONEY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Value for Money":
    st.markdown('<div class="page-title">Value for Money</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Identify cost-efficient players relative to their performance</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>VFM Score</strong> = (Performance Score ÷ Market Value) × 1,000,000<br>
        <strong>Performance Score</strong> is a position-weighted blend:<br>
        Attacker 60% Overall + 40% Potential · Midfielder 65/35 · Defender 70/30 · Goalkeeper 75/25
    </div>""", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        vfm_group = st.multiselect("Position group", df["position_group"].unique().tolist(),
                                   default=df["position_group"].unique().tolist())
    with fc2:
        val_range = st.slider("Max market value (£M)", 0.0, float(df["value_gbp_m"].max()),
                              float(df["value_gbp_m"].max()), step=0.5)
    with fc3:
        top_n = st.slider("Show top N players", 5, 50, 20)

    vfm_df  = df[df["position_group"].isin(vfm_group) & (df["value_gbp_m"] <= val_range)]
    top_vfm = vfm_df.nlargest(top_n, "value_score")

    cl, cr = st.columns([3, 2])
    with cl:
        st.markdown('<div class="section-label">Top players by value-for-money score</div>', unsafe_allow_html=True)
        fig = px.bar(top_vfm.sort_values("value_score"), x="value_score", y="Name",
                     color="position_group", color_discrete_map=GROUP_COLORS, orientation="h",
                     hover_data=["Club", "Overall", "Potential", "performance_score", "value_gbp_m"],
                     labels={"value_score": "VFM Score", "position_group": "Group"})
        fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE,
                          yaxis=dict(**AXIS_STYLE, tickfont=dict(size=11)),
                          height=max(380, top_n * 22), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="section-label">VFM score by position group</div>', unsafe_allow_html=True)
        fig = px.box(vfm_df, x="position_group", y="value_score", color="position_group",
                     color_discrete_map=GROUP_COLORS, points="outliers",
                     labels={"position_group": "Group", "value_score": "VFM Score"})
        fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                          xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=300)
        st.plotly_chart(fig, use_container_width=True)

        avg_vfm = vfm_df.groupby("position_group")["value_score"].mean().reset_index()
        avg_vfm.columns = ["Group", "Avg VFM Score"]
        avg_vfm["Avg VFM Score"] = avg_vfm["Avg VFM Score"].round(2)
        st.dataframe(avg_vfm, use_container_width=True, hide_index=True)

    # Wage efficiency section (new in V2)
    if "wage_efficiency" in df.columns:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Wage efficiency — performance per £1k wage</div>', unsafe_allow_html=True)
        st.caption("Wage Efficiency = (Performance Score ÷ Weekly Wage) × 1,000")
        top_wage = vfm_df.nlargest(top_n, "wage_efficiency")
        fig = px.bar(top_wage.sort_values("wage_efficiency"), x="wage_efficiency", y="Name",
                     color="position_group", color_discrete_map=GROUP_COLORS, orientation="h",
                     hover_data=["Club", "Overall", "wage_gbp_k", "performance_score"],
                     labels={"wage_efficiency": "Wage Efficiency", "position_group": "Group"})
        fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE,
                          yaxis=dict(**AXIS_STYLE, tickfont=dict(size=11)),
                          height=max(380, top_n * 22), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Performance vs VFM score — bubble chart</div>', unsafe_allow_html=True)
    st.caption("Bubble size = market value. Top-right = high performance AND high VFM.")
    fig = px.scatter(vfm_df, x="performance_score", y="value_score",
                     color="position_group", color_discrete_map=GROUP_COLORS,
                     size="value_gbp_m", size_max=30,
                     hover_data=["Name", "Club", "Overall", "value_gbp_m", "primary_position"],
                     labels={"performance_score": "Performance Score",
                             "value_score": "VFM Score", "position_group": "Group"},
                     opacity=0.7)
    fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Top VFM player shortlist</div>', unsafe_allow_html=True)
    shortlist_cols = ["Name", "Age", "Nationality", "Club", "Position", "primary_position",
                      "position_group", "Overall", "Potential", "performance_score",
                      "value_gbp_m", "value_score"]
    if "wage_efficiency" in top_vfm.columns:
        shortlist_cols.append("wage_efficiency")
    st.dataframe(top_vfm[shortlist_cols].reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TEAM BUILDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏟️ Team Builder":
    st.markdown('<div class="page-title">Team Builder</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Build an optimal squad within a fixed transfer budget</div>', unsafe_allow_html=True)

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        budget = st.number_input("Total budget (£M)", min_value=1.0, max_value=2000.0,
                                  value=200.0, step=10.0)
    with bc2:
        formation_choice = st.selectbox("Formation", ["4-3-3", "4-4-2", "3-5-2", "4-2-3-1", "5-3-2"])
    with bc3:
        optimise_by = st.selectbox("Optimise by", ["Best value-for-money", "Highest overall rating",
                                                    "Best wage efficiency"])
    with bc4:
        max_age = st.slider("Max player age", 18, 45, 35)

    FORMATIONS = {
        "4-3-3":   {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Attacker": 3},
        "4-4-2":   {"Goalkeeper": 1, "Defender": 4, "Midfielder": 4, "Attacker": 2},
        "3-5-2":   {"Goalkeeper": 1, "Defender": 3, "Midfielder": 5, "Attacker": 2},
        "4-2-3-1": {"Goalkeeper": 1, "Defender": 4, "Midfielder": 5, "Attacker": 1},
        "5-3-2":   {"Goalkeeper": 1, "Defender": 5, "Midfielder": 3, "Attacker": 2},
    }
    requirements  = FORMATIONS[formation_choice]
    total_players = sum(requirements.values())

    sort_col_map = {
        "Best value-for-money":   "value_score",
        "Highest overall rating": "Overall",
        "Best wage efficiency":   "wage_efficiency" if "wage_efficiency" in df.columns else "value_score",
    }
    sort_col      = sort_col_map[optimise_by]
    budget_pounds = budget * 1_000_000

    selected_players = []
    remaining_budget = budget_pounds
    used_names       = set()

    for group, n_needed in requirements.items():
        pool = df[
            (df["position_group"] == group) &
            (~df["Name"].isin(used_names)) &
            (df["value_gbp"] <= remaining_budget) &
            (df["Age"] <= max_age)
        ].sort_values(sort_col, ascending=False)

        picks = []
        for _, row in pool.iterrows():
            if len(picks) >= n_needed:
                break
            if row["value_gbp"] <= remaining_budget:
                picks.append(row)
                remaining_budget -= row["value_gbp"]
                used_names.add(row["Name"])
        selected_players.extend(picks)

    squad_df = pd.DataFrame(selected_players)

    if squad_df.empty:
        st.warning("No players found within this budget. Try increasing it.")
    else:
        total_cost  = squad_df["value_gbp"].sum() / 1_000_000
        budget_left = budget - total_cost

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Players selected", f"{len(squad_df)} / {total_players}")
        k2.metric("Total cost",       f"£{total_cost:.1f}M")
        k3.metric("Budget used",      f"{(total_cost/budget)*100:.1f}%")
        k4.metric("Budget remaining", f"£{budget_left:.1f}M")
        k5.metric("Avg overall",      f"{squad_df['Overall'].mean():.1f}")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        for group in ["Goalkeeper", "Defender", "Midfielder", "Attacker"]:
            gp = squad_df[squad_df["position_group"] == group]
            if gp.empty:
                continue
            bc = BADGE_CLASS.get(group, "badge-pos")
            st.markdown(f'<div class="section-label">{group}s</div>', unsafe_allow_html=True)
            cols = st.columns(min(len(gp), 4))
            for i, (_, row) in enumerate(gp.iterrows()):
                with cols[i % 4]:
                    wage_eff_str = (f"Wage Eff {row['wage_efficiency']:.1f}"
                                    if "wage_efficiency" in row and pd.notna(row["wage_efficiency"])
                                    else "")
                    st.markdown(f"""
                    <div class="player-card">
                        <div class="player-name">{row['Name']}</div>
                        <div class="player-meta">{row['Club']}</div>
                        <div class="player-meta">{row['Nationality']} · Age {row['Age']}</div>
                        <div style="margin-top:0.6rem">
                            <span class="badge {bc}">{row['primary_position']}</span>
                            <span class="badge badge-pos">OVR {row['Overall']}</span>
                        </div>
                        <div style="margin-top:0.5rem;font-size:0.8rem;color:#8b949e">
                            £{row['value_gbp']/1_000_000:.1f}M · VFM {row['value_score']:.1f}
                            {"· " + wage_eff_str if wage_eff_str else ""}
                        </div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="section-label">Budget allocation by group</div>', unsafe_allow_html=True)
            cost_by_group = squad_df.groupby("position_group")["value_gbp"].sum().reset_index()
            cost_by_group["Total Cost (£M)"] = cost_by_group["value_gbp"] / 1_000_000
            fig = px.pie(cost_by_group, names="position_group", values="Total Cost (£M)",
                         color="position_group", color_discrete_map=GROUP_COLORS, hole=0.5)
            fig.update_layout(**PLOT_LAYOUT, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            st.markdown('<div class="section-label">Overall rating per player</div>', unsafe_allow_html=True)
            fig = px.bar(squad_df.sort_values("Overall", ascending=False),
                         x="Name", y="Overall", color="position_group",
                         color_discrete_map=GROUP_COLORS,
                         hover_data=["Club", "primary_position", "value_gbp_m"])
            fig.update_layout(**PLOT_LAYOUT, xaxis=dict(**AXIS_STYLE, tickangle=-40),
                              yaxis=AXIS_STYLE, height=360, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Full squad table</div>', unsafe_allow_html=True)
        sd_cols = ["Name", "Age", "Nationality", "Club", "Position", "primary_position",
                   "position_group", "Overall", "Potential", "performance_score",
                   "value_gbp_m", "value_score"]
        if "wage_efficiency" in squad_df.columns:
            sd_cols.append("wage_efficiency")
        sd = squad_df[sd_cols].copy()
        st.dataframe(sd.reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PLAYER CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Player Clustering":
    st.markdown('<div class="page-title">Player Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">K-Means unsupervised learning — grouping players by performance and value profile</div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        n_clusters = st.slider("Number of clusters (K)", 3, 8, 4)   # default 4 mirrors notebook
    with cc2:
        cl_group = st.multiselect("Filter by position group",
                                  df["position_group"].unique().tolist(),
                                  default=df["position_group"].unique().tolist())

    with st.spinner("Running K-Means..."):
        df_cl, centers, cnames = run_clustering(len(df), n_clusters)

    df_cl = df_cl[df_cl["position_group"].isin(cl_group)]
    unique_labels = sorted(df_cl["cluster_label"].unique().tolist())
    c_color_map   = {lbl: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                     for i, lbl in enumerate(unique_labels)}

    st.markdown("""
    <div class="insight-box">
        K-Means groups players into K clusters based on <strong>Age, Overall, Potential,
        Weekly Wage, Market Value, and Performance Score</strong> — with no prior labels.
        Players in the same cluster share a similar overall profile.
        (Default K=4 matches the notebook analysis.)
    </div>""", unsafe_allow_html=True)

    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="section-label">Performance score vs VFM score by cluster</div>', unsafe_allow_html=True)
        fig = px.scatter(df_cl, x="performance_score", y="value_score",
                         color="cluster_label", color_discrete_map=c_color_map,
                         hover_data=["Name", "Club", "position_group", "Overall", "primary_position"],
                         labels={"performance_score": "Performance Score",
                                 "value_score": "VFM Score",
                                 "cluster_label": "Cluster"},
                         opacity=0.7)
        fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="section-label">Overall rating vs market value by cluster</div>', unsafe_allow_html=True)
        fig = px.scatter(df_cl, x="value_gbp_m", y="Overall",
                         color="cluster_label", color_discrete_map=c_color_map,
                         hover_data=["Name", "Club", "position_group", "primary_position"],
                         labels={"value_gbp_m": "Market Value (£M)",
                                 "cluster_label": "Cluster"},
                         opacity=0.7)
        fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Cluster profile summary</div>', unsafe_allow_html=True)
    agg_dict = dict(
        Players=("Name", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Overall=("Overall", "mean"),
        Avg_Potential=("Potential", "mean"),
        Avg_Perf=("performance_score", "mean"),
        Avg_Value_M=("value_gbp_m", "mean"),
        Avg_VFM=("value_score", "mean"),
    )
    if "wage_efficiency" in df_cl.columns:
        agg_dict["Avg_WageEff"] = ("wage_efficiency", "mean")

    cs = df_cl.groupby("cluster_label").agg(**agg_dict).round(2).reset_index()
    st.dataframe(cs, use_container_width=True, hide_index=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Cluster composition by position group</div>', unsafe_allow_html=True)
    fig = px.histogram(df_cl, x="cluster_label", color="position_group",
                       color_discrete_map=GROUP_COLORS, barmode="stack",
                       labels={"cluster_label": "Cluster", "position_group": "Group"})
    fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=340)
    st.plotly_chart(fig, use_container_width=True)

    # Age vs Overall violin per cluster
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Overall rating distribution by cluster</div>', unsafe_allow_html=True)
    fig = px.violin(df_cl, x="cluster_label", y="Overall",
                    color="cluster_label", color_discrete_map=c_color_map,
                    box=True, points=False,
                    labels={"cluster_label": "Cluster"})
    fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                      xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    selected_cluster = st.selectbox("Explore players in a cluster", unique_labels)
    cp = df_cl[df_cl["cluster_label"] == selected_cluster]
    st.markdown(
        f"<div style='color:#8b949e;font-size:0.82rem;margin-bottom:1rem'>"
        f"{len(cp)} players in this cluster</div>",
        unsafe_allow_html=True
    )
    show_cols = ["Name", "Age", "Nationality", "Club", "Position", "primary_position",
                 "position_group", "Overall", "Potential", "performance_score",
                 "value_gbp_m", "value_score"]
    if "wage_efficiency" in cp.columns:
        show_cols.append("wage_efficiency")
    st.dataframe(cp[show_cols].sort_values("Overall", ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — YOUNG TALENT FINDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌟 Young Talent Finder":
    st.markdown('<div class="page-title">Young Talent Finder</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">High-potential players with room to grow — ideal for long-term recruitment</div>', unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        yt_groups = st.multiselect("Position group", df["position_group"].unique().tolist(),
                                   default=df["position_group"].unique().tolist())
    with fc2:
        max_age_yt = st.slider("Max age", 16, 30, 23)
    with fc3:
        min_pot = st.slider("Min potential rating", 60, 99, 75)
    with fc4:
        max_val_yt = st.slider("Max market value (£M)", 0.0,
                               float(df["value_gbp_m"].max()), 30.0, step=1.0)

    yt_df = df[
        df["position_group"].isin(yt_groups) &
        (df["Age"] <= max_age_yt) &
        (df["Potential"] >= min_pot) &
        (df["value_gbp_m"] <= max_val_yt)
    ].copy()

    yt_df["growth_gap"] = yt_df["Potential"] - yt_df["Overall"]
    vfm_cap             = yt_df["value_score"].quantile(0.95) if len(yt_df) > 0 else 1
    yt_df["talent_score"] = (
        0.5 * yt_df["Potential"] +
        0.3 * yt_df["growth_gap"] +
        0.2 * yt_df["value_score"].clip(upper=vfm_cap)
    ).round(2)

    st.markdown(
        f"<div style='color:#8b949e;font-size:0.82rem;margin:0.5rem 0 1rem'>"
        f"{len(yt_df):,} players match your filters</div>",
        unsafe_allow_html=True
    )

    if yt_df.empty:
        st.warning("No players match these filters. Try relaxing the criteria.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Players found",  f"{len(yt_df):,}")
    k2.metric("Avg age",        f"{yt_df['Age'].mean():.1f}")
    k3.metric("Avg potential",  f"{yt_df['Potential'].mean():.1f}")
    k4.metric("Avg growth gap", f"+{yt_df['growth_gap'].mean():.1f}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="section-label">Age vs potential — sized by growth gap</div>', unsafe_allow_html=True)
        st.caption("Bigger bubble = more room to grow. Top-right = young with high ceiling.")
        fig = px.scatter(yt_df, x="Age", y="Potential",
                         color="position_group", color_discrete_map=GROUP_COLORS,
                         size="growth_gap", size_max=30,
                         hover_data=["Name", "Club", "Overall", "growth_gap",
                                     "value_gbp_m", "primary_position"],
                         labels={"position_group": "Group"},
                         opacity=0.8)
        fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="section-label">Top 20 by talent score</div>', unsafe_allow_html=True)
        fig = px.bar(yt_df.nlargest(20, "talent_score").sort_values("talent_score"),
                     x="talent_score", y="Name",
                     color="position_group", color_discrete_map=GROUP_COLORS, orientation="h",
                     hover_data=["Age", "Overall", "Potential", "growth_gap", "Club"],
                     labels={"talent_score": "Talent Score", "position_group": "Group"})
        fig.update_layout(**PLOT_LAYOUT,
                          xaxis=AXIS_STYLE,
                          yaxis=dict(**AXIS_STYLE, tickfont=dict(size=10)),
                          height=500, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Overall vs potential — gap analysis</div>', unsafe_allow_html=True)
    st.caption("Players furthest above the dashed line have the most room to develop.")
    fig = px.scatter(yt_df, x="Overall", y="Potential",
                     color="position_group", color_discrete_map=GROUP_COLORS,
                     hover_data=["Name", "Club", "Age", "growth_gap", "value_gbp_m", "primary_position"],
                     labels={"position_group": "Group"}, opacity=0.75)
    mn, mx = int(yt_df["Overall"].min()), int(yt_df["Potential"].max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="No growth",
                             line=dict(color="#8b949e", dash="dash", width=1)))
    fig.update_layout(**PLOT_LAYOUT, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Talent gems — high potential, low cost</div>', unsafe_allow_html=True)
    st.caption("High potential, low market value — ideal scouting targets.")
    gems = yt_df[yt_df["value_gbp_m"] <= max_val_yt * 0.5].nlargest(6, "talent_score")
    if not gems.empty:
        gem_cols = st.columns(min(len(gems), 3))
        for i, (_, row) in enumerate(gems.iterrows()):
            bc = BADGE_CLASS.get(row["position_group"], "badge-pos")
            with gem_cols[i % 3]:
                st.markdown(f"""
                <div class="player-card">
                    <div class="player-name">{row['Name']}</div>
                    <div class="player-meta">{row['Club']} · {row['Nationality']}</div>
                    <div style="margin-top:0.5rem">
                        <span class="badge {bc}">{row['primary_position']}</span>
                        <span class="badge badge-gem">Age {row['Age']}</span>
                    </div>
                    <div style="margin-top:0.6rem;font-size:0.82rem;color:#e6edf3">
                        OVR <strong>{row['Overall']}</strong>
                        &nbsp;→&nbsp;
                        POT <strong>{row['Potential']}</strong>
                        &nbsp;(+{row['growth_gap']})
                    </div>
                    <div style="font-size:0.8rem;color:#8b949e;margin-top:0.3rem">
                        £{row['value_gbp_m']:.1f}M · Talent {row['talent_score']:.1f}
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Full young talent table</div>', unsafe_allow_html=True)
    yt_cols = ["Name", "Age", "Nationality", "Club", "Position", "primary_position",
               "position_group", "Overall", "Potential", "growth_gap",
               "value_gbp_m", "talent_score", "value_score"]
    if "wage_efficiency" in yt_df.columns:
        yt_cols.append("wage_efficiency")
    st.dataframe(
        yt_df[yt_cols].sort_values("talent_score", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )
