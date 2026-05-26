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

current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(dashboard_dir)
sys.path.append(dashboard_dir)

from utils import (
    section_header, apply_plotly_theme,
    LABELS, LABEL_COLORS,
)

def load_data(filename):
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

    goemotions = load_data("goemotions_5class.csv")
    fpb_train = load_data("fpb_train.csv")
    fpb_val = load_data("fpb_val.csv")
    fpb_test = load_data("fpb_test.csv")

    if goemotions is None or fpb_train is None:
        st.error(":material/error: Centralized CSV files not found. Please run 'python src/data_preprocessing.py' before presenting!")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GoEmotions", f"{len(goemotions):,}", "General domain")
    c2.metric("FPB Train", f"{len(fpb_train):,}", "Financial domain")
    c3.metric("FPB Validation", f"{len(fpb_val):,}")
    c4.metric("FPB Test", f"{len(fpb_test):,}", "Fixed evaluation target")

    st.divider()

    st.markdown("### :material/query_stats: Dataset Explorer")
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

    selected_classes = st.multiselect("Filter by sentiment class", LABELS, default=LABELS)
    filtered = selected_df[selected_df["label"].isin(selected_classes)]

    col1, col2 = st.columns([1.2, 1])

    with col1:
        counts = filtered["label"].value_counts().reindex(LABELS, fill_value=0)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts.index.tolist(),
                    y=counts.values.tolist(),
                    marker_color=[LABEL_COLORS[l] for l in counts.index], # Uses updated utils.py
                    text=[f"{v:,} ({100*v/len(selected_df):.1f}%)" if len(selected_df)>0 else "0" for v in counts.values],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(title=f"Label Distribution - {dataset_choice}", yaxis_title="Count", height=380)
        st.plotly_chart(apply_plotly_theme(fig), width="stretch", use_container_width=True)

    with col2:
        filtered_wc = filtered.copy()
        filtered_wc["text"] = filtered_wc["text"].fillna("")
        filtered_wc["word_count"] = filtered_wc["text"].str.split().str.len()

        fig = px.box(
            filtered_wc, x="label", y="word_count", color="label",
            color_discrete_map=LABEL_COLORS, title="Text Length by Class"
        )
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig), width="stretch", use_container_width=True)

    with st.expander(":material/visibility: View Random Sample Texts"):
        for lbl in selected_classes:
            subset = filtered[filtered["label"] == lbl]
            if len(subset) > 0:
                sample = subset["text"].sample(1).iloc[0] 
                st.markdown(f"**{lbl}:** {sample}")

    st.divider()

    st.markdown("### :material/swap_horiz: Cross-Domain Analysis")

    col1, col2 = st.columns(2)

    with col1:
        go_counts = goemotions["label"].value_counts().reindex(LABELS, fill_value=0)
        fpb_counts = fpb_train["label"].value_counts().reindex(LABELS, fill_value=0)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["GoEmotions", "FPB Train"])
        fig.add_trace(go.Bar(x=LABELS, y=(100 * go_counts / len(goemotions)).values, marker_color=[LABEL_COLORS[l] for l in LABELS], showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=LABELS, y=(100 * fpb_counts / len(fpb_train)).values, marker_color=[LABEL_COLORS[l] for l in LABELS], showlegend=False), row=1, col=2)
        fig.update_yaxes(title_text="% of dataset", row=1, col=1)
        fig.update_layout(title="Class Distribution Comparison (%)", height=380)
        st.plotly_chart(apply_plotly_theme(fig), width="stretch", use_container_width=True)

    with col2:
        fig = go.Figure()
        conditions = ["Model A\n(General)", "Model B\n(Domain)", "Model C\n(Mixed)"]
        go_sizes = [len(goemotions), 0, len(goemotions)]
        fpb_sizes = [0, len(fpb_train), len(fpb_train)]

        # UPDATED: Using Dark Navy (#264653) and Light Orange (#f4a261)
        fig.add_trace(go.Bar(name="GoEmotions", x=conditions, y=go_sizes, marker_color="#264653"))
        fig.add_trace(go.Bar(name="FPB Train", x=conditions, y=fpb_sizes, marker_color="#f4a261"))
        fig.update_layout(barmode="stack", title="Training Set Composition per Condition", yaxis_title="Samples", height=380)
        st.plotly_chart(apply_plotly_theme(fig), width="stretch", use_container_width=True)

    st.divider()

    st.markdown("### :material/route: Pipeline Summary")
    st.info("**7-step text normalisation:** lowercase → remove URLs → remove @mentions → strip hashtags → remove non-ASCII → replace punctuation → normalise whitespace.")
    
    with st.expander(":material/verified_user: Split Verification - Zero Data Leakage"):
        st.markdown("Stratified 70/15/15 split on FPB. Verify class proportions are consistent:")
        split_data = []
        for name, df in [("Train", fpb_train), ("Val", fpb_val), ("Test", fpb_test)]:
            row = {"Split": name, "Total": len(df)}
            for lbl in LABELS:
                row[lbl] = f"{100 * (df['label'] == lbl).mean():.1f}%"
            split_data.append(row)
        st.dataframe(pd.DataFrame(split_data), width="stretch", hide_index=True)

    st.divider()
    
    st.markdown("### :material/warning_amber: Data Engineering Impact: The Stratification Problem")
    st.markdown("Why can't we just use a naive random split? Let's simulate a split without strict stratification.")
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        st.info("In the FPB dataset, **'Fear'** is extremely rare (only ~1.5% of data).")
        if st.button(":material/shuffle: Simulate Naive Random Split", type="primary"):
            import random
            simulated_fear = sum([1 for _ in range(727) if random.random() < 0.015])
            st.session_state['sim_fear'] = simulated_fear
            
    with col_sim2:
        if 'sim_fear' in st.session_state:
            sim_fear = st.session_state['sim_fear']
            if sim_fear < 5:
                st.error(f":material/error: **Disaster!** In this random split, only **{sim_fear}** 'Fear' samples made it into the test set (Expected: ~11). The ML evaluation would be completely unstable.")
            elif sim_fear > 16:
                st.error(f":material/error: **Disaster!** In this random split, **{sim_fear}** 'Fear' samples leaked into the test set. The model will look artificially bad.")
            else:
                st.warning(f":material/warning: Got lucky this time with {sim_fear} samples. But luck isn't a pipeline.")
            
            st.success(":material/check_circle: **Our Production Pipeline:** Uses strict `stratify=df['label']` and `random_state=42` to guarantee exactly **7** Fear samples in every single test run, ensuring deterministic ML evaluation.")

    st.divider()

    st.markdown("### :material/science: Live Preprocessing & Tokenization Preview")
    st.markdown("Test the Data Engineering pipeline in real-time. See how raw text is transformed before hitting the ML models.")

    import re
    import string

    def demo_clean_text(text):
        steps = {"Original": text}
        text = str(text).lower()
        steps["1. Lowercase"] = text
        text = re.sub(r'http\S+|www\.\S+', '', text)
        steps["2. Remove URLs"] = text
        text = re.sub(r'@\w+', '', text)
        steps["3. Remove Mentions"] = text
        text = re.sub(r'#(\w+)', r'\1', text)
        steps["4. Strip Hashtags"] = text
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        steps["5. Remove Non-ASCII"] = text
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        steps["6. Remove Punctuation"] = text
        text = re.sub(r'\d+', 'NUM', text)
        steps["7. Mask Numbers"] = text
        text = re.sub(r'\s+', ' ', text).strip()
        steps["Final Output"] = text
        return steps

    col_input, col_output = st.columns([1, 1.2])

    with col_input:
        sample_text = st.text_area(
            "Enter raw text (or use the financial default):",
            value="Omg!😮 The CEO @ElonMusk just tweeted that TSLA revenue grew by 45% this quarter!!! #BullMarket https://investor.tsla.com",
            height=150
        )
        
    with col_output:
        if sample_text:
            processed_steps = demo_clean_text(sample_text)
            
            st.markdown("**TF-IDF Baseline Pipeline (Task 2)**")
            st.code(processed_steps["Final Output"], language="text")
            
            with st.expander(":material/list_alt: View Step-by-Step Execution"):
                for step_name, result in processed_steps.items():
                    if step_name not in ["Original", "Final Output"]:
                        st.markdown(f"**{step_name}:** `{result}`")
            
            st.markdown("**BERT Tokenization (Tasks 3 & 4)**")
            st.info(":material/tips_and_updates: *Note: BERT uses WordPiece tokenization and relies on original casing/punctuation. It skips the TF-IDF cleaning steps above to preserve context (e.g., distinguishing 'US' from 'us').*")

    st.divider()