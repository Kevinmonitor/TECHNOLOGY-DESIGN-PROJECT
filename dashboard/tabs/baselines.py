"""
Tab 3: Baseline Models (TF-IDF)
Owner: Hanok (ML Engineer) / Bikram (Data Insights)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_dir = os.path.dirname(current_dir)
sys.path.append(dashboard_dir)

from utils import (
    section_header, load_csv, apply_plotly_theme,
    LABELS, LABEL_COLORS, MODEL_COLORS,
)

def render():
    section_header(
        "Baseline Models - TF-IDF + Classical ML",
        "Hanok (Modeling) & Bikram (Data Insights)",
    )

    # ── Load results ──
    results_df = load_csv("task2_baseline_results.csv")

    if results_df is None:
        st.error("🚨 `task2_baseline_results.csv` not found. Please ensure baseline artifacts are generated.")
        st.stop()

    # ── Top-level metrics ──
    st.markdown("### 🏆 Best Performance per Training Condition")
    best_rows = []
    for condition in results_df["Model Condition"].unique():
        subset = results_df[results_df["Model Condition"] == condition]
        best = subset.loc[subset["F1 (macro)"].idxmax()]
        best_rows.append(best)

    cols = st.columns(3)
    condition_short = {"Model A (General Only)": "A", "Model B (Domain Only)": "B", "Model C (Mixed)": "C"}
    for col, row in zip(cols, best_rows):
        short = condition_short.get(row["Model Condition"], "?")
        with col:
            st.metric(
                label=f"Model {short} - {row['Classifier']}",
                value=f"F1: {row['F1 (macro)']:.3f}",
                delta=f"Acc: {row['Accuracy']:.3f}",
            )

    st.divider()

    # ── BIKRAM'S PRESENTATION MOMENT: Vocabulary Overlap ──
    st.markdown("### 🧠 Data Engineer Insight: Why Model A Failed")
    
    col_text, col_chart = st.columns([1.5, 1])
    
    with col_text:
        st.markdown(
            """
            **The Problem:** Model A (trained purely on GoEmotions) performed poorly on the Financial PhraseBank test set, specifically completely failing to detect *Fear* and *Joy*.
            
            **The Data Diagnosis:** Through exploratory data analysis, we performed a vocabulary overlap check. Because Baseline models rely on TF-IDF (exact word matches), a lack of shared vocabulary means the model is mathematically blind to the target domain.
            
            * Only **53.8%** of the words used in the Financial domain exist in the General domain.
            * Crucial financial terms (e.g., 'ebitda', 'amortisation', 'bullish') are treated as unknown tokens.
            * **Conclusion:** This data-engineering insight validates our entire architectural strategy for Domain Adaptation (Model B and C).
            """
        )
    
    with col_chart:
        # Donut chart for Vocab Overlap
        fig_vocab = go.Figure(data=[go.Pie(
            labels=['Shared Vocabulary', 'Missing Financial Terms'],
            values=[53.8, 46.2],
            hole=.6,
            marker_colors=['#4C72B0', '#EAEAEA'], # Blue for shared, light grey for missing
            textinfo='label+percent'
        )])
        fig_vocab.update_layout(
            title_text="Vocabulary Overlap (General vs Financial)",
            showlegend=False,
            margin=dict(t=40, b=0, l=0, r=0),
            height=250
        )
        st.plotly_chart(apply_plotly_theme(fig_vocab), use_container_width=True)

    st.divider()

    # ── Interactive filters & Chart ──
    st.markdown("### 📊 Experiment Explorer")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_conditions = st.multiselect("Training Conditions", results_df["Model Condition"].unique().tolist(), default=results_df["Model Condition"].unique().tolist())
    with filter_col2:
        selected_classifiers = st.multiselect("Classifiers", results_df["Classifier"].unique().tolist(), default=results_df["Classifier"].unique().tolist())

    filtered = results_df[(results_df["Model Condition"].isin(selected_conditions)) & (results_df["Classifier"].isin(selected_classifiers))]

    metric_choice = st.selectbox("Select metric to compare", ["F1 (macro)", "Accuracy"])
    fig = px.bar(
        filtered, x="Model Condition", y=metric_choice, color="Classifier",
        barmode="group", color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"], text=metric_choice
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=400, yaxis_range=[0, 1])
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.divider()

    # ── Per-class F1 heatmap ──
    st.markdown("### 🎯 Per-Class F1 Analysis")
    
    # Hardcoded from notebook output
    perclass_data = {
        "Experiment": [
            "A - LogReg", "A - SVM", "A - NB",
            "B - LogReg", "B - SVM", "B - NB",
            "C - LogReg", "C - SVM", "C - NB",
        ],
        "Fear":     [0.000, 0.040, 0.000, 0.167, 0.250, 0.200, 0.148, 0.133, 0.000],
        "Joy":      [0.000, 0.000, 0.000, 0.286, 0.316, 0.462, 0.269, 0.296, 0.133],
        "Neutral":  [0.688, 0.693, 0.681, 0.816, 0.835, 0.811, 0.817, 0.828, 0.803],
        "Optimism": [0.367, 0.370, 0.389, 0.649, 0.651, 0.631, 0.636, 0.643, 0.620],
        "Sadness":  [0.000, 0.189, 0.000, 0.715, 0.763, 0.600, 0.800, 0.813, 0.640],
    }
    perclass_df = pd.DataFrame(perclass_data).set_index("Experiment")

    fig = go.Figure(data=go.Heatmap(
        z=perclass_df.values, x=perclass_df.columns.tolist(), y=perclass_df.index.tolist(),
        text=np.round(perclass_df.values, 3).astype(str), texttemplate="%{text}",
        colorscale="RdYlGn", zmin=0, zmax=1
    ))
    fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.success("**Baseline benchmark to beat:** Model B + Linear SVM achieves **0.563 macro F1 / 0.762 accuracy**. Advanced LLM/BERT experiments must exceed this to justify their computational cost.")