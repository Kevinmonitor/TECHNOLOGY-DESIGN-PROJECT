"""
Tab 3: Baseline Models (TF-IDF)
Owner: Hanok (ML Engineer)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (
    section_header, load_csv, apply_plotly_theme,
    LABELS, LABEL_COLORS, MODEL_COLORS,
)


def render():
    section_header(
        "Baseline Models - TF-IDF + Classical ML",
        "Hanok - ML Engineer",
    )

    # ── Load results ──
    results_df = load_csv("task2_baseline_results.csv")

    if results_df is None:
        st.warning(
            "⚠️ `task2_baseline_results.csv` not found in `dashboard/data/`. "
            "Copy it from Hanok's baseline notebook output."
        )
        st.stop()

    # ── Top-level metrics: best per condition ──
    st.markdown("### Best Performance per Training Condition")

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

    st.markdown("")

    # ── Interactive filters ──
    st.markdown("### Experiment Explorer")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_conditions = st.multiselect(
            "Training Conditions",
            results_df["Model Condition"].unique().tolist(),
            default=results_df["Model Condition"].unique().tolist(),
        )
    with filter_col2:
        selected_classifiers = st.multiselect(
            "Classifiers",
            results_df["Classifier"].unique().tolist(),
            default=results_df["Classifier"].unique().tolist(),
        )

    filtered = results_df[
        (results_df["Model Condition"].isin(selected_conditions))
        & (results_df["Classifier"].isin(selected_classifiers))
    ]

    # ── Metrics comparison chart ──
    metric_choice = st.selectbox(
        "Select metric to compare",
        ["F1 (macro)", "F1 (weighted)", "Accuracy", "Precision (macro)", "Recall (macro)"],
    )

    fig = px.bar(
        filtered,
        x="Model Condition",
        y=metric_choice,
        color="Classifier",
        barmode="group",
        title=f"{metric_choice} - All Experiments",
        color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
        text=metric_choice,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=450, yaxis_range=[0, 1])
    st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    # ── Full results table ──
    with st.expander("📋 Full Results Table"):
        st.dataframe(
            filtered.style.format({
                "Accuracy": "{:.4f}",
                "Precision (macro)": "{:.4f}",
                "Recall (macro)": "{:.4f}",
                "F1 (macro)": "{:.4f}",
                "F1 (weighted)": "{:.4f}",
            }),
            width="stretch",
            hide_index=True,
        )

    st.markdown("---")

    # ── Per-class F1 heatmap ──
    st.markdown("### Per-Class F1 Analysis")

    # NOTE: This section requires per-class F1 data.
    # If you have it in a separate CSV (e.g., task2_perclass_f1.csv), load it here.
    # Otherwise, you can hardcode the values from Hanok's notebook output.
    # Format: rows = experiment, columns = class F1 scores

    # Hardcoded from notebook output (replace with CSV load if available)
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

    # Interactive heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=perclass_df.values,
            x=perclass_df.columns.tolist(),
            y=perclass_df.index.tolist(),
            text=np.round(perclass_df.values, 3).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            colorbar=dict(title="F1"),
        )
    )
    fig.update_layout(
        title="Per-Class F1 Scores - All 9 Experiments",
        height=400,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    st.markdown(
        """
    <div class="highlight-box">
        <strong>Key observation:</strong> Fear (n=7 test) and Joy (n=9 test) show near-zero F1 
        for Model A across all classifiers. Domain-specific training (Model B) rescues these classes 
        partially, but sample scarcity remains the bottleneck.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Confusion matrices ──
    st.markdown("### Confusion Matrices")
    st.markdown("*Select a condition to view its best classifier's confusion matrix.*")

    # NOTE: To show real confusion matrices, save them as CSVs from Hanok's notebook:
    #   pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv('cm_a.csv')
    # Then load here. For now, this is a placeholder structure.

    cm_condition = st.selectbox(
        "Select condition",
        ["Model A (General Only)", "Model B (Domain Only)", "Model C (Mixed)"],
        key="cm_select",
    )

    st.info(
        "💡 **To enable confusion matrices:** Save confusion matrix arrays from the baseline "
        "notebook as CSVs (e.g., `cm_a_svm.csv`) into `dashboard/data/` and load them here. "
        "Use `pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(path)`."
    )

    st.markdown("---")

    # ── Key findings ──
    st.markdown("### Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">🔍 Domain Gap (Model A → B)</div>
            <ul>
                <li>~30pp macro F1 gain from domain-specific training</li>
                <li>165/727 test samples (23%) rescued by domain data</li>
                <li>TF-IDF features from GoEmotions are nearly useless for financial text</li>
                <li>Only 53.8% vocabulary overlap between domains</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="dashboard-card">
            <div class="card-title">⚠️ Naive Mixing Fails (Model C)</div>
            <ul>
                <li>Model C slightly underperforms Model B at TF-IDF level</li>
                <li>13:1 GoEmotions-to-FPB ratio dilutes financial vocabulary</li>
                <li>Motivates sequential fine-tuning over naive concatenation</li>
                <li>Bag-of-words cannot selectively attend to domain tokens</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <div class="highlight-box">
        <strong>Baseline benchmark to beat:</strong> Model B + Linear SVM achieves 
        <strong>0.563 macro F1 / 0.762 accuracy</strong>. 
        BERT experiments (Tasks 3–4) must exceed this to justify added complexity.
    </div>
    """,
        unsafe_allow_html=True,
    )
