"""
Tab 2: Data Pipeline & Preparation
Owner: Bikram (Data Engineer)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

# Robust path handling for Streamlit
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(dashboard_dir)
sys.path.append(dashboard_dir)

from utils import (
    section_header, apply_plotly_theme,
    LABELS, LABEL_COLORS,
)

def load_data(filename):
    """Bulletproof data loading for live presentations."""
    paths_to_try = [
        os.path.join(project_root, "data", "processed", filename),
        os.path.join(dashboard_dir, "data", "processed", filename),
        os.path.join("..", "data", "processed", filename)
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

def render():
    section_header(
        "Data Pipeline & Preparation",
        "Bikram Bhattarai - Data Engineer",
    )

    # -- Load datasets dynamically --
    goemotions = load_data("goemotions_5class.csv")
    fpb_train = load_data("fpb_train.csv")
    fpb_val = load_data("fpb_val.csv")
    fpb_test = load_data("fpb_test.csv")

    if goemotions is None or fpb_train is None:
        st.error("🚨 Centralized CSV files not found. Please run 'python src/data_preprocessing.py' before presenting!")
        st.stop()

    # -- Dataset overview metrics --
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GoEmotions", f"{len(goemotions):,}", "General domain")
    c2.metric("FPB Train", f"{len(fpb_train):,}", "Financial domain")
    c3.metric("FPB Validation", f"{len(fpb_val):,}")
    c4.metric("FPB Test", f"{len(fpb_test):,}", "Fixed evaluation target")

    st.divider()

    # -- Interactive dataset explorer --
    st.markdown("### 📊 Dataset Explorer")
    dataset_choice = st.selectbox(
        "Select dataset to explore",
        ["GoEmotions (General)", "FPB Train (Domain)", "FPB Test (Evaluation)"],
    )

    dataset_map = {
        "GoEmotions (General)": goemotions,
        "FPB Train (Domain)": fpb_train,
        "FPB Test (Evaluation)": fpb_test,
    }
    selected_df = dataset_map[dataset_choice]

    # Class filter
    selected_classes = st.multiselect("Filter by sentiment class", LABELS, default=LABELS)
    filtered = selected_df[selected_df["label"].isin(selected_classes)]

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Label distribution bar chart
        counts = filtered["label"].value_counts().reindex(LABELS, fill_value=0)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts.index.tolist(),
                    y=counts.values.tolist(),
                    marker_color=[LABEL_COLORS[l] for l in counts.index],
                    text=[f"{v:,} ({100*v/len(selected_df):.1f}%)" if len(selected_df)>0 else "0" for v in counts.values],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(title=f"Label Distribution - {dataset_choice}", yaxis_title="Count", height=380)
        st.plotly_chart(apply_plotly_theme(fig), width="stretch", use_container_width=True)

    with col2:
        # Word count distribution (Safely handled nulls)
        filtered_wc = filtered.copy()
        filtered_wc["text"] = filtered_wc["text"].fillna("")
        filtered_wc["word_count"] = filtered_wc["text"].str.split().str.len()

        fig = px.box(
            filtered_wc, x="label", y="word_count", color="label",
            color_discrete_map=LABEL_COLORS, title=f"Text Length by Class"
        )
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig), width="stretch", use_container_width=True)

    # Sample texts
    with st.expander("👀 View Random Sample Texts"):
        for lbl in selected_classes:
            subset = filtered[filtered["label"] == lbl]
            if len(subset) > 0:
                sample = subset["text"].sample(1).iloc[0] # Random sample for better live demo
                st.markdown(f"**{lbl}:** {sample}")

    st.divider()

    # -- Cross-domain comparison --
    st.markdown("### 🔄 Cross-Domain Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Side-by-side class distributions
        go_counts = goemotions["label"].value_counts().reindex(LABELS, fill_value=0)
        fpb_counts = fpb_train["label"].value_counts().reindex(LABELS, fill_value=0)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["GoEmotions", "FPB Train"])
        fig.add_trace(go.Bar(x=LABELS, y=(100 * go_counts / len(goemotions)).values, marker_color=[LABEL_COLORS[l] for l in LABELS], showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=LABELS, y=(100 * fpb_counts / len(fpb_train)).values, marker_color=[LABEL_COLORS[l] for l in LABELS], showlegend=False), row=1, col=2)
        fig.update_yaxes(title_text="% of dataset", row=1, col=1)
        fig.update_layout(title="Class Distribution Comparison (%)", height=380)
        st.plotly_chart(apply_plotly_theme(fig), width="stretch", use_container_width=True)

    with col2:
        # Dataset size comparison
        fig = go.Figure()
        conditions = ["Model A\n(General)", "Model B\n(Domain)", "Model C\n(Mixed)"]
        go_sizes = [len(goemotions), 0, len(goemotions)]
        fpb_sizes = [0, len(fpb_train), len(fpb_train)]

        fig.add_trace(go.Bar(name="GoEmotions", x=conditions, y=go_sizes, marker_color="#4C72B0"))
        fig.add_trace(go.Bar(name="FPB Train", x=conditions, y=fpb_sizes, marker_color="#DD8452"))
        fig.update_layout(barmode="stack", title="Training Set Composition per Condition", yaxis_title="Samples", height=380)
        st.plotly_chart(apply_plotly_theme(fig), width="stretch", use_container_width=True)

    st.divider()

    # -- Preprocessing & Verification --
    st.markdown("### ⚙️ Pipeline Summary")
    st.info("**7-step text normalisation:** lowercase → remove URLs → remove @mentions → strip hashtags → remove non-ASCII → replace punctuation → normalise whitespace.")
    
    with st.expander(" Split Verification - Zero Data Leakage"):
        st.markdown("Stratified 70/15/15 split on FPB. Verify class proportions are consistent:")
        split_data = []
        for name, df in [("Train", fpb_train), ("Val", fpb_val), ("Test", fpb_test)]:
            row = {"Split": name, "Total": len(df)}
            for lbl in LABELS:
                row[lbl] = f"{100 * (df['label'] == lbl).mean():.1f}%"
            split_data.append(row)
        st.dataframe(pd.DataFrame(split_data), width="stretch", hide_index=True)