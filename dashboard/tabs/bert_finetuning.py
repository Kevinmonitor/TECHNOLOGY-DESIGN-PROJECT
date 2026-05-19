"""
Tab 4: BERT Fine-tuning (Models A & B)
Owner: Aniketh (NLP Engineer)

STATUS: Results available — update CSV filenames below once Aniketh exports them.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (
    section_header, load_csv, apply_plotly_theme, placeholder_tab,
    LABELS, LABEL_COLORS,
)


def render():
    section_header(
        "BERT Fine-tuning — Models A & B",
        "Aniketh — NLP Engineer",
    )

    # ── Try loading BERT results ──
    # Expected CSV format (one row per model):
    #   model, accuracy, f1_macro, f1_weighted, precision_macro, recall_macro,
    #   f1_fear, f1_joy, f1_neutral, f1_optimism, f1_sadness
    #
    # Aniketh: export your results to these filenames:
    bert_results = load_csv("bert_results.csv")

    if bert_results is None:
        # ── Show placeholder with expected structure ──
        st.info(
            "📂 **Aniketh:** Export your BERT results to `dashboard/data/bert_results.csv` "
            "with columns: model, accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, "
            "f1_fear, f1_joy, f1_neutral, f1_optimism, f1_sadness"
        )

        st.markdown("")

        # Show what we know so far + framework
        st.markdown("### Experiment Design")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            <div class="dashboard-card">
                <div class="card-title">BERT Model A (General Domain)</div>
                <ul>
                    <li><code>bert-base-uncased</code> fine-tuned on GoEmotions</li>
                    <li>43,404 training samples</li>
                    <li>5-class classification head</li>
                    <li>Evaluated on FPB test set (727 samples)</li>
                </ul>
                <p style="color:#00d4ff">Measures: Can general emotion knowledge transfer?</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div class="dashboard-card">
                <div class="card-title">BERT Model B (Domain Specific)</div>
                <ul>
                    <li><code>bert-base-uncased</code> fine-tuned on FPB train</li>
                    <li>3,392 training samples</li>
                    <li>5-class classification head</li>
                    <li>Evaluated on FPB test set (727 samples)</li>
                </ul>
                <p style="color:#00d4ff">Measures: In-domain performance ceiling with limited data</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown("### TF-IDF Baselines to Beat")
        baseline_comparison = pd.DataFrame({
            "Model": ["TF-IDF Model A (SVM)", "TF-IDF Model B (SVM)"],
            "Accuracy": [0.561, 0.762],
            "F1 (Macro)": [0.258, 0.563],
            "F1 (Weighted)": [0.477, 0.754],
        })
        st.dataframe(baseline_comparison, width="stretch", hide_index=True)

        st.markdown(
            """
        <div class="highlight-box">
            <strong>Expected outcome:</strong> BERT should significantly outperform TF-IDF 
            baselines due to contextual embeddings capturing word order, negation, and 
            domain-specific semantics that bag-of-words misses.
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    # ═══════════════════════════════════════════════════════
    # BELOW THIS LINE: renders when bert_results.csv exists
    # ═══════════════════════════════════════════════════════

    # ── Headline metrics ──
    cols = st.columns(len(bert_results))
    for col, (_, row) in zip(cols, bert_results.iterrows()):
        with col:
            st.metric(
                label=f"BERT {row['model']}",
                value=f"F1: {row['f1_macro']:.3f}",
                delta=f"Acc: {row['accuracy']:.3f}",
            )

    st.markdown("")

    # ── BERT vs TF-IDF comparison ──
    st.markdown("### BERT vs TF-IDF Baselines")

    comparison_data = []
    # TF-IDF baselines (hardcoded from Hanok's results)
    comparison_data.append({"Model": "TF-IDF A (SVM)", "Type": "Baseline", "F1 (macro)": 0.258, "Accuracy": 0.561})
    comparison_data.append({"Model": "TF-IDF B (SVM)", "Type": "Baseline", "F1 (macro)": 0.563, "Accuracy": 0.762})

    for _, row in bert_results.iterrows():
        comparison_data.append({
            "Model": f"BERT {row['model']}",
            "Type": "BERT",
            "F1 (macro)": row["f1_macro"],
            "Accuracy": row["accuracy"],
        })

    comp_df = pd.DataFrame(comparison_data)

    fig = px.bar(
        comp_df,
        x="Model",
        y="F1 (macro)",
        color="Type",
        title="F1 (Macro) — BERT vs TF-IDF Baselines",
        color_discrete_map={"Baseline": "#8899aa", "BERT": "#00d4ff"},
        text="F1 (macro)",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=450, yaxis_range=[0, 1])
    st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    # ── Per-class F1 ──
    st.markdown("### Per-Class F1 Breakdown")

    perclass_cols = ["f1_fear", "f1_joy", "f1_neutral", "f1_optimism", "f1_sadness"]
    if all(c in bert_results.columns for c in perclass_cols):
        perclass_data = []
        for _, row in bert_results.iterrows():
            for lbl, col_name in zip(LABELS, perclass_cols):
                perclass_data.append({
                    "Model": f"BERT {row['model']}",
                    "Class": lbl,
                    "F1": row[col_name],
                })

        fig = px.bar(
            pd.DataFrame(perclass_data),
            x="Class",
            y="F1",
            color="Model",
            barmode="group",
            title="Per-Class F1 — BERT Models",
            color_discrete_sequence=["#4C72B0", "#DD8452"],
        )
        fig.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(apply_plotly_theme(fig), width="stretch")

    # ── Training details ──
    with st.expander("🔧 Hyperparameters & Training Details"):
        st.markdown(
            """
        | Parameter | Value |
        |-----------|-------|
        | Base model | `bert-base-uncased` |
        | Max sequence length | 128 |
        | Batch size | 16 |
        | Learning rate | 2e-5 |
        | Epochs | 3–5 (early stopping) |
        | Optimizer | AdamW |
        | Warmup steps | 10% of training |
        
        *Update these values with Aniketh's actual hyperparameters.*
        """
        )
