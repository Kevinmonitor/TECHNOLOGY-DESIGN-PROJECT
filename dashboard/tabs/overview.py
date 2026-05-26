"""
Tab 1: Project Overview
Owner: Fin (Project Lead)
"""

import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import section_header, LABELS, LABEL_COLORS, apply_plotly_theme
import plotly.graph_objects as go


def render():
    st.markdown("""
    <style>
    /* Clean card style */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1e3a5f;  /* deep navy */
        margin: 0;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
    }

    /* Highlight box for research question */
    .highlight-box {
        background: #f0f7ff;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
        color: #1e293b;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Professional table */
    .pro-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        font-size: 0.9rem;
        background: white;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .pro-table thead th {
        background: #f8fafc;
        color: #1e3a5f;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e2e8f0;
        text-align: left;
    }
    .pro-table tbody td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #f1f5f9;
        color: #334155;
    }
    .pro-table tbody tr:last-child td {
        border-bottom: none;
    }
    .pro-table tbody tr:hover td {
        background: #f8fafc;
    }

    /* Milestone timeline cards */
    .milestone-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 0.5rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        transition: all 0.2s;
        height: 100%;
    }
    .milestone-card.completed {
        border-left: 3px solid #10b981;  /* green */
    }
    .milestone-card.in-progress {
        border-left: 3px solid #2563eb;  /* blue */
        background: #f8faff;
    }
    .milestone-card.upcoming {
        border-left: 3px solid #cbd5e1;
        background: #f9fafb;
        color: #64748b;
    }
    .milestone-week {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.3px;
    }
    .milestone-task {
        font-weight: 600;
        margin: 0.3rem 0;
        color: #0f172a;
    }
    .milestone-status {
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Team cards */
    .team-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
        height: 100%;
    }
    .team-card:hover {
        box-shadow: 0 12px 25px -8px rgba(0,0,0,0.1);
        transform: translateY(-3px);
        border-color: #2563eb;
    }
    .team-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.2rem;
    }
    .team-role {
        font-weight: 600;
        color: #2563eb;
        margin-bottom: 0.5rem;
    }
    .team-desc {
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.4;
    }

    /* Section headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

    section_header(
        "Domain-Adaptive Sentiment Analysis",
        "Comparing Training Strategies for Financial Sentiment Classification",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">:material/memory: Models Compared</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">2</div>
            <div class="metric-label">:material/dataset: Datasets</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">727</div>
            <div class="metric-label">:material/science: Test Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5</div>
            <div class="metric-label">:material/label: Label Classes</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div class="highlight-box">
        <strong>:material/search: Research Question:</strong> How does training data composition and transfer learning 
        strategy affect sentiment classification performance when moving from general-domain to 
        financial-domain text?
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-title">:material/science: Experimental Framework</div>', unsafe_allow_html=True)
        st.markdown("""
        <table class="pro-table">
            <thead>
                <tr><th>Model</th><th>Training Strategy</th><th>Purpose</th></tr>
            </thead>
            <tbody>
                <tr><td><b>Model A</b></td><td>BERT on GoEmotions only</td><td>General-domain baseline</td></tr>
                <tr><td><b>Model B</b></td><td>BERT on FPB only</td><td>Domain-specific baseline</td></tr>
                <tr><td><b>Model C</b></td><td>GoEmotions → FPB sequential</td><td>Transfer learning (core experiment)</td></tr>
                <tr><td><b>LLM</b></td><td>Zero/few-shot prompting</td><td>No fine-tuning reference</td></tr>
            </tbody>
        </table>
        <p style="font-size:0.85rem; color:#475569; margin-top:0.7rem;">
        :material/push_pin: All models evaluated on the <b>same FPB test set</b> (727 samples).<br>
        :material/push_pin: TF-IDF baselines established first to isolate architecture vs data effects.
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">:material/pie_chart: 5‑Class Sentiment Scheme</div>', unsafe_allow_html=True)
        test_dist = {"Fear": 7, "Joy": 9, "Neutral": 422, "Optimism": 191, "Sadness": 98}
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(test_dist.keys()),
                    values=list(test_dist.values()),
                    hole=0.5,
                    marker=dict(
                        colors=[LABEL_COLORS[l] for l in test_dist.keys()],
                        line=dict(color='white', width=2)
                    ),
                    textinfo="label+percent",
                    textfont=dict(size=12, color='#1e293b'),
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
                )
            ]
        )
        fig.update_layout(
            title=dict(text="FPB Test Set Distribution", font=dict(size=16, color='#1e3a5f')),
            height=340,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=10, l=10, r=10)
        )
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.markdown("")

    st.markdown('<div class="section-title">:material/calendar_month: Project Milestones</div>', unsafe_allow_html=True)
    milestones = {
        "Weeks 1-3": ("Individual Research Reports", "Complete"),
        "Week 4": ("Team Design Brief", "Complete"),
        "Weeks 5-7": ("Data Pipeline & Preparation", "Complete"),
        "Week 8": ("Baseline Modelling (TF-IDF)", "Complete"),
        "Weeks 9-10": ("BERT Fine-tuning & LLM Experiments", "Complete"),
        "Week 11": ("Evaluation & Dashboard", "Complete"),
        "Week 12": ("Demo & Final Report", "Upcoming"),
    }

    cols = st.columns(len(milestones))
    for col, (week, (task, status)) in zip(cols, milestones.items()):
        
        if 'Complete' in status:
            status_class = "completed"
            status_icon = ":material/check_circle: Complete"
        elif 'Progress' in status:
            status_class = "in-progress"
            status_icon = ":material/sync: In Progress"
        else:
            status_class = "upcoming"
            status_icon = ":material/event_upcoming: Upcoming"

        with col:
            st.markdown(f"""
            <div class="milestone-card {status_class}">
                <div class="milestone-week">{week}</div>
                <div class="milestone-task">{task}</div>
                <div class="milestone-status">{status_icon}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    st.markdown('<div class="section-title">:material/groups: Team Roles</div>', unsafe_allow_html=True)
    team = [
        ("Fin", "Project Lead", ":material/explore: Project coordination, Model C oversight, presentation"),
        ("Bikram", "Data Engineer", ":material/settings: Data pipeline, preprocessing, CSV exports,Baseline, dashboard"),
        ("Hanok", "ML Engineer", ":material/trending_up: TF-IDF baselines (Models A/B/C), hyperparameter tuning"),
        ("Aniketh", "NLP Engineer", ":material/memory: BERT fine-tuning (Models A & B)"),
        ("Kevin", "LLM Analyst", ":material/science: Zero-shot & few-shot with Gemma 3 / Llama 3.2"),
        ("Himanshu", "Eval Engineer", ":material/bar_chart: Cross-model evaluation, metrics, final comparison"),
    ]

    cols = st.columns(3)
    for i, (name, role, desc) in enumerate(team):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="team-card">
                <div class="team-name">{name}</div>
                <div class="team-role">{role}</div>
                <div class="team-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)