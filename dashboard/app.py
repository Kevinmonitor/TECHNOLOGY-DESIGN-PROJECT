"""
Domain-Adaptive Sentiment Analysis Dashboard
COS60011 - Technology Design Project | Semester 1, 2026
Swinburne University of Technology

Run: streamlit run app.py
"""

import streamlit as st

# ── Page config ──
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean theme CSS that works with Streamlit 1.57 ──
st.markdown("""
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(100, 100, 100, 0.3);
        border-radius: 8px;
        padding: 16px 20px;
    }

    /* Highlight box */
    .highlight-box {
        background-color: rgba(15, 52, 96, 0.6);
        border-left: 4px solid #00d4ff;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 8px 0;
    }

    /* Dashboard card */
    .dashboard-card {
        background: rgba(22, 33, 62, 0.6);
        border: 1px solid rgba(15, 52, 96, 0.6);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-title {
        color: #00d4ff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Hide default Streamlit footer/header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──
st.sidebar.markdown("## 📊 Sentiment Analysis")
st.sidebar.markdown("**COS60011** - Sem 1, 2026")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Team")
st.sidebar.markdown("""
- **Fin** - Project Lead  
- **Bikram** - Data Engineer  
- **Hanok** - ML Engineer  
- **Aniketh** - NLP Engineer  
- **Kevin** - LLM Analyst  
- **Himanshu** - Evaluation Engineer  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("*Swinburne University of Technology*")


# ── Main content: use tabs for navigation ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Overview",
    "🔧 Data Pipeline",
    "📈 Baselines",
    "🤖 BERT Fine-tuning",
    "🔄 Sequential Transfer",
    "💬 LLM Experiments",
    "📊 Evaluation",
])

with tab1:
    from tabs import overview
    overview.render()

with tab2:
    from tabs import data_pipeline
    data_pipeline.render()

with tab3:
    from tabs import baselines
    baselines.render()

with tab4:
    from tabs import bert_finetuning
    bert_finetuning.render()

with tab5:
    from tabs import sequential_transfer
    sequential_transfer.render()

with tab6:
    from tabs import llm_experiments
    llm_experiments.render()
with tab7:
    from tabs import evaluation
    evaluation.render()
