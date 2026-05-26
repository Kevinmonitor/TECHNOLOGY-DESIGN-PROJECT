"""
Shared utilities, constants, and helper functions for all dashboard pages.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# ── Project constants ──
LABELS = ["Fear", "Joy", "Neutral", "Optimism", "Sadness"]

# Add this to utils.py
LABEL_COLORS = {
    "Fear": "#e76f51",      # Burnt Orange
    "Sadness": "#f4a261",   # Light Orange
    "Neutral": "#e9c46a",   # Mustard Yellow
    "Optimism": "#2a9d8f",  # Vibrant Teal
    "Joy": "#264653"        # Dark Navy/Teal
}

MODEL_COLORS = {
    "Model A": "#264653", # Dark Navy: The baseline/foundational model (General text)
    "Model B": "#e76f51", # Burnt Orange: High contrast, representing the shift to the Financial domain
    "Model C": "#2a9d8f", # Vibrant Teal: The "synthesis" or "success" color representing your sequential transfer
}

# Plotly theme that respects Streamlit's dark mode
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0e0", family="Segoe UI, sans-serif"),
    xaxis=dict(gridcolor="rgba(100,100,100,0.2)", zerolinecolor="rgba(100,100,100,0.2)"),
    yaxis=dict(gridcolor="rgba(100,100,100,0.2)", zerolinecolor="rgba(100,100,100,0.2)"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def apply_plotly_theme(fig):
    """Apply the dashboard theme to any plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def data_path(filename):
    """Return the full path to a file in the data/ directory."""
    return os.path.join(os.path.dirname(__file__), "data", filename)


def load_csv(filename):
    """Load a CSV from the data/ directory. Returns None if file doesn't exist."""
    path = data_path(filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def section_header(title, subtitle=None):
    """Render a styled section header."""
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")


def placeholder_tab(task_name, owner, description):
    """Render a placeholder for tabs where results aren't ready yet."""
    st.markdown(f"### {task_name}")
    st.markdown(f"**Owner:** {owner}")
    st.markdown("---")
    st.info(f"⏳ {description}. Drop results CSV into `dashboard/data/` to activate this tab.")


def confusion_matrix_heatmap(cm, labels, title="Confusion Matrix"):
    """Create a plotly confusion matrix heatmap."""
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    text = []
    for i in range(len(labels)):
        row = []
        for j in range(len(labels)):
            row.append(f"{cm[i][j]}<br>({cm_norm[i][j]:.0%})")
        text.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm, x=labels, y=labels,
            text=text, texttemplate="%{text}", textfont=dict(size=11),
            colorscale="Blues", showscale=True,
            colorbar=dict(title="Recall"),
        )
    )
    fig.update_layout(
        title=title, xaxis_title="Predicted", yaxis_title="True",
        yaxis=dict(autorange="reversed"), width=450, height=400,
    )
    return apply_plotly_theme(fig)
