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
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (
    section_header, load_csv, apply_plotly_theme,
    LABELS, LABEL_COLORS,
)


def render():
    section_header(
        "Data Pipeline & Preparation",
        "Bikram Bhattarai — Data Engineer",
    )

    # ── Load datasets ──
    goemotions = load_csv("goemotions_5class.csv")
    fpb_train = load_csv("fpb_train.csv")
    fpb_val = load_csv("fpb_val.csv")
    fpb_test = load_csv("fpb_test.csv")

    if goemotions is None or fpb_train is None:
        st.warning(
            "⚠️ CSV files not found in `dashboard/data/`. "
            "Copy your CSVs (goemotions_5class.csv, fpb_train.csv, fpb_val.csv, fpb_test.csv) "
            "into the `dashboard/data/` folder."
        )
        st.stop()

    # ── Dataset overview metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GoEmotions", f"{len(goemotions):,}", "General domain")
    c2.metric("FPB Train", f"{len(fpb_train):,}", "Financial domain")
    c3.metric("FPB Validation", f"{len(fpb_val):,}")
    c4.metric("FPB Test", f"{len(fpb_test):,}", "Fixed evaluation target")

    st.markdown("")

    # ── Interactive dataset explorer ──
    st.markdown("### Dataset Explorer")
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
    selected_classes = st.multiselect(
        "Filter by sentiment class",
        LABELS,
        default=LABELS,
    )
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
                    text=[f"{v:,} ({100*v/len(selected_df):.1f}%)" for v in counts.values],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title=f"Label Distribution — {dataset_choice}",
            yaxis_title="Count",
            height=380,
        )
        st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    with col2:
        # Word count distribution
        filtered_wc = filtered.copy()
        filtered_wc["word_count"] = filtered_wc["text"].str.split().str.len()

        fig = px.box(
            filtered_wc,
            x="label",
            y="word_count",
            color="label",
            color_discrete_map=LABEL_COLORS,
            title=f"Text Length by Class — {dataset_choice}",
        )
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    # Sample texts
    with st.expander("📝 Sample texts from selected dataset"):
        for lbl in selected_classes:
            subset = filtered[filtered["label"] == lbl]
            if len(subset) > 0:
                sample = subset["text"].iloc[0]
                st.markdown(
                    f"**{lbl}:** {sample[:200]}{'...' if len(sample) > 200 else ''}"
                )

    st.markdown("---")

    # ── Cross-domain comparison ──
    st.markdown("### Cross-Domain Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Side-by-side class distributions
        go_counts = goemotions["label"].value_counts().reindex(LABELS, fill_value=0)
        fpb_counts = fpb_train["label"].value_counts().reindex(LABELS, fill_value=0)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["GoEmotions", "FPB Train"])

        fig.add_trace(
            go.Bar(
                x=LABELS,
                y=(100 * go_counts / len(goemotions)).values,
                marker_color=[LABEL_COLORS[l] for l in LABELS],
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=LABELS,
                y=(100 * fpb_counts / len(fpb_train)).values,
                marker_color=[LABEL_COLORS[l] for l in LABELS],
                showlegend=False,
            ),
            row=1, col=2,
        )
        fig.update_yaxes(title_text="% of dataset", row=1, col=1)
        fig.update_layout(title="Class Distribution Comparison (%)", height=380)
        st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    with col2:
        # Dataset size comparison
        fig = go.Figure()
        conditions = ["Model A\n(General)", "Model B\n(Domain)", "Model C\n(Mixed)"]
        go_sizes = [len(goemotions), 0, len(goemotions)]
        fpb_sizes = [0, len(fpb_train), len(fpb_train)]

        fig.add_trace(go.Bar(
            name="GoEmotions", x=conditions, y=go_sizes,
            marker_color="#4C72B0",
        ))
        fig.add_trace(go.Bar(
            name="FPB Train", x=conditions, y=fpb_sizes,
            marker_color="#DD8452",
        ))
        fig.update_layout(
            barmode="stack",
            title="Training Set Composition per Condition",
            yaxis_title="Samples",
            height=380,
        )
        st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    # ── Key stats table ──
    st.markdown("### Dataset Summary")
    summary_data = []
    for name, df in [
        ("GoEmotions", goemotions),
        ("FPB Train", fpb_train),
        ("FPB Val", fpb_val),
        ("FPB Test", fpb_test),
    ]:
        wc = df["text"].str.split().str.len()
        summary_data.append({
            "Dataset": name,
            "Rows": f"{len(df):,}",
            "Avg Words": f"{wc.mean():.1f}",
            "Median Words": f"{wc.median():.0f}",
            "Labels": df["label"].nunique(),
            "Minority Class": df["label"].value_counts().idxmin(),
            "Minority Count": f"{df['label'].value_counts().min():,}",
        })
    st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)

    # ── Preprocessing pipeline ──
    st.markdown("### Preprocessing Pipeline")
    st.markdown(
        """
    <div class="highlight-box">
        <strong>7-step text normalisation:</strong> lowercase → remove URLs → remove @mentions → 
        strip hashtags → remove non-ASCII → replace punctuation → normalise whitespace.
        <br><br>
        <strong>Note:</strong> Numbers are preserved for BERT (WordPiece tokenization handles them natively).
        TF-IDF baselines replaced digits with NUM — a known limitation documented in the baseline notebook.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── FPB split verification ──
    with st.expander("🔍 Split Verification — No Data Leakage"):
        st.markdown("Stratified 70/15/15 split on FPB. Verify class proportions are consistent:")
        split_data = []
        for name, df in [("Train", fpb_train), ("Val", fpb_val), ("Test", fpb_test)]:
            row = {"Split": name, "Total": len(df)}
            for lbl in LABELS:
                row[lbl] = f"{100 * (df['label'] == lbl).mean():.1f}%"
            split_data.append(row)
        st.dataframe(pd.DataFrame(split_data), width="stretch", hide_index=True)
