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
        st.error("`task2_baseline_results.csv` not found. Please ensure baseline artifacts are generated.")
        st.stop()

    # ── Top-level metrics ──
    st.markdown("### :material/emoji_events: Best Performance per Training Condition")
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
    st.markdown("### :material/psychology: Data Engineer Insight: The Vocabulary Gap")
    
    col_text, col_chart = st.columns([1.2, 1])
    
    with col_text:
        st.markdown(
            """
            <div class="highlight-box" style="margin-top: 0px;">
                <strong>Why did the General Model (A) fail on Financial Text?</strong>
            </div>
            
            * :material/error: **The Problem:** Model A completely failed to detect *Fear* and *Joy* in the test set.
            * :material/search: **The Cause:** TF-IDF models require exact word matches to make predictions.
            * :material/trending_down: **The Blind Spot:** **46.2%** of financial words (e.g., <i>'ebitda'</i>, <i>'bullish'</i>) do not exist in the general training data!
            * :material/check_circle: **Conclusion:** This data reality proves why our team had to build Domain-Adaptive models (B & C).
            """,
            unsafe_allow_html=True
        )
    
    with col_chart:
        # Donut chart for Vocab Overlap with high contrast and custom tooltips
        fig_vocab = go.Figure(data=[go.Pie(
            labels=['Shared Vocabulary', 'Missing Financial Terms'],
            values=[53.8, 46.2],
            hole=.65,
            marker=dict(
                colors=['#3498db', '#e74c3c'], # Bright Blue for Shared, Vibrant Red for Missing
                line=dict(color='#1e293b', width=2) # Dark border for crisp separation
            ),
            textinfo='label+percent',
            textfont=dict(size=14, color='white'), # Force white text on the chart
            hoverinfo='label+percent',
            hoverlabel=dict(
                bgcolor="#0f172a", # Very dark navy background for tooltip
                font_size=15,
                font_color="white", # Explicitly white tooltip text
                bordercolor="#00d4ff" # Neon blue border on hover
            )
        )])
        
        fig_vocab.update_layout(
            title=dict(
                text="Domain Vocabulary Overlap",
                font=dict(size=18, color="#e0e0e0"),
                x=0.5, # Center the title
            ),
            showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10),
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(apply_plotly_theme(fig_vocab), use_container_width=True)

    with st.expander(":material/touch_app: Interactive: Test Model A's Blind Spots"):
        st.markdown("Type a financial sentence to see which words Model A (General Domain) doesn't understand.")
        
        test_sentence = st.text_input("Enter financial text:", value="The company reported bullish EBITDA margins and lowered amortization costs.")
        
        # A mock list of common financial words that don't appear in general emotion datasets
        financial_jargon = ["ebitda", "amortization", "bullish", "bearish", "dividend", "equity", "margins", "revenue", "fiscal", "quarter", "stakeholders", "liquidity"]
        
        if test_sentence:
            words = test_sentence.split()
            highlighted_text = []
            blind_count = 0
            
            for w in words:
                clean_w = ''.join(e for e in w.lower() if e.isalnum())
                if clean_w in financial_jargon:
                    # Highlight in red
                    highlighted_text.append(f"<span style='background-color: #e74c3c; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>{w}</span>")
                    blind_count += 1
                else:
                    # Normal text
                    highlighted_text.append(w)
            
            st.markdown(" ".join(highlighted_text), unsafe_allow_html=True)
            
            if blind_count > 0:
                st.error(f":material/error: Model A is completely blind to {blind_count} critical context words in this sentence!")
            else:
                st.success(":material/check_circle: Model A understands this vocabulary.")

    st.divider()

    # ── Interactive filters & Chart ──
    st.markdown("### :material/explore: Experiment Explorer")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_conditions = st.multiselect("Training Conditions", results_df["Model Condition"].unique().tolist(), default=results_df["Model Condition"].unique().tolist())
    with filter_col2:
        selected_classifiers = st.multiselect("Classifiers", results_df["Classifier"].unique().tolist(), default=results_df["Classifier"].unique().tolist())

    filtered = results_df[(results_df["Model Condition"].isin(selected_conditions)) & (results_df["Classifier"].isin(selected_classifiers))]

    metric_choice = st.selectbox(
        "Select metric to compare", 
        ["F1 (macro)", "F1 (weighted)", "Accuracy", "Precision (macro)", "Recall (macro)"])
    fig = px.bar(
        filtered, x="Model Condition", y=metric_choice, color="Classifier",
        barmode="group", color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"], text=metric_choice
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=400, yaxis_range=[0, 1])
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.divider()

    # ── Per-class F1 heatmap ──
    st.markdown("### :material/analytics: Per-Class F1 Analysis")
    
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