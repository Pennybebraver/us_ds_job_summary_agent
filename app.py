"""
Streamlit dashboard for the US DS Job Summary Agent.
Launch with: streamlit run app.py
"""

import ast
import glob
import logging
import os

import pandas as pd
import streamlit as st

from src.career.advisor import generate_career_suggestions
from src.dashboard.filters import render_sidebar_filters
from src.dashboard.pages import (
    render_career_advisor,
    render_category_cluster,
    render_data_explorer,
    render_experience_analysis,
    render_overview,
    render_regional_analysis,
    render_salary_analysis,
    render_skills,
)
from src.nlp.clustering import get_cluster_summary
from src.nlp.summarizer import get_skill_frequency

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DS/ML Job Market Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load and parse processed job data from CSV."""
    df = pd.read_csv(path)

    # Parse key_skills column from string to dict
    if "key_skills" in df.columns:
        df["key_skills"] = df["key_skills"].apply(_safe_parse_dict)

    # Ensure numeric columns are properly typed
    numeric_cols = [
        "salary_min", "salary_max",
        "salary_min_annual", "salary_max_annual",
        "yoe_min", "yoe_max",
        "latitude", "longitude",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _safe_parse_dict(val):
    """Safely parse a string representation of a dict."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return {}
    return {}


def main():
    """Main Streamlit app entry point."""
    st.title("📊 Data Science & ML Job Market Dashboard")

    # --- Sidebar: Data Source ---
    with st.sidebar:
        st.header("Data Source")
        data_files = sorted(
            glob.glob("data/processed_jobs_*.csv"),
            reverse=True,
        )

        if not data_files:
            st.error("No processed data found. Run the pipeline first:")
            st.code("python -m src.main --full-run", language="bash")
            st.stop()

        selected_file = st.selectbox(
            "Select dataset",
            data_files,
            format_func=lambda x: os.path.basename(x),
        )

        df = load_data(selected_file)
        st.caption(f"{len(df):,} jobs loaded")

        st.divider()

        # --- Sidebar: Filters ---
        filtered_df = render_sidebar_filters(df)
        st.caption(f"{len(filtered_df):,} jobs after filters")

        st.divider()

        # --- Sidebar: PDF Download ---
        st.header("Export")
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                try:
                    from src.reports.pdf_generator import generate_report
                    from src.visualization.charts import create_all_charts

                    sf = get_skill_frequency(filtered_df)
                    cs = get_cluster_summary(filtered_df)
                    career = generate_career_suggestions(filtered_df, sf)
                    charts = create_all_charts(filtered_df, sf)
                    report_path = generate_report(
                        df=filtered_df,
                        cluster_summary=cs,
                        charts=charts,
                        career_suggestions=career,
                        skill_freq=sf,
                    )
                    with open(report_path, "rb") as f:
                        st.download_button(
                            "📥 Download PDF",
                            data=f.read(),
                            file_name=os.path.basename(report_path),
                            mime="application/pdf",
                        )
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")

    # --- Compute derived data ---
    skill_freq = get_skill_frequency(filtered_df)
    cluster_summary = get_cluster_summary(filtered_df)
    career_suggestions = generate_career_suggestions(
        filtered_df, skill_freq
    )

    # --- Tab navigation ---
    tabs = st.tabs([
        "📋 Overview",
        "🗺️ Regional",
        "📊 Categories",
        "💰 Salary",
        "📈 Experience",
        "🛠️ Skills",
        "🎯 Career Advisor",
        "🔍 Data Explorer",
    ])

    with tabs[0]:
        render_overview(filtered_df, skill_freq, cluster_summary)
    with tabs[1]:
        render_regional_analysis(filtered_df)
    with tabs[2]:
        render_category_cluster(filtered_df, cluster_summary)
    with tabs[3]:
        render_salary_analysis(filtered_df)
    with tabs[4]:
        render_experience_analysis(filtered_df)
    with tabs[5]:
        render_skills(filtered_df, skill_freq)
    with tabs[6]:
        render_career_advisor(career_suggestions)
    with tabs[7]:
        render_data_explorer(filtered_df)


if __name__ == "__main__":
    main()
