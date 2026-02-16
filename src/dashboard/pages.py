"""
Tab rendering functions for the Streamlit dashboard.
Each function renders one tab of the dashboard.
"""

import logging
from collections import Counter
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.dashboard.plotly_charts import (
    plotly_cluster_distribution,
    plotly_jobs_by_category,
    plotly_regional_breakdown,
    plotly_salary_by_cluster,
    plotly_salary_by_region,
    plotly_salary_histogram,
    plotly_skills_demand,
    plotly_top_companies,
    plotly_us_choropleth,
    plotly_yoe_by_category,
    plotly_yoe_distribution,
)

logger = logging.getLogger(__name__)


def render_overview(
    df: pd.DataFrame,
    skill_freq: Dict[str, Counter],
    cluster_summary: pd.DataFrame,
) -> None:
    """Render the Overview tab with executive summary metrics."""
    st.header("Executive Summary")

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Jobs", f"{len(df):,}")

    with col2:
        n_companies = (
            df["company"].nunique() if "company" in df.columns else 0
        )
        st.metric("Companies", f"{n_companies:,}")

    with col3:
        n_regions = (
            df["region"].nunique() if "region" in df.columns else 0
        )
        st.metric("Regions", n_regions)

    with col4:
        median_sal = None
        if "salary_max_annual" in df.columns:
            median_sal = df["salary_max_annual"].dropna().median()
        if median_sal and not pd.isna(median_sal):
            st.metric("Median Salary", f"${median_sal:,.0f}")
        else:
            st.metric("Median Salary", "N/A")

    with col5:
        median_yoe = None
        if "yoe_min" in df.columns:
            median_yoe = df["yoe_min"].dropna().median()
        if median_yoe and not pd.isna(median_yoe):
            st.metric("Median YOE", f"{median_yoe:.1f} yrs")
        else:
            st.metric("Median YOE", "N/A")

    st.divider()

    # Two-column chart layout
    col_left, col_right = st.columns(2)

    with col_left:
        fig = plotly_jobs_by_category(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available.")

    with col_right:
        fig = plotly_cluster_distribution(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cluster distribution data available.")

    # Cluster summary table
    if not cluster_summary.empty:
        st.subheader("Category Summary")
        st.dataframe(
            cluster_summary,
            use_container_width=True,
            hide_index=True,
        )


def render_regional_analysis(df: pd.DataFrame) -> None:
    """Render the Regional Analysis tab."""
    st.header("Regional Analysis")

    # US Choropleth
    fig = plotly_us_choropleth(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No state-level data available for choropleth.")

    # City-level heatmap via Folium
    st.subheader("City-Level Job Density")
    geo_df = pd.DataFrame()
    if "latitude" in df.columns and "longitude" in df.columns:
        geo_df = df.dropna(subset=["latitude", "longitude"])

    if not geo_df.empty:
        try:
            import folium
            import folium.plugins as plugins
            from streamlit_folium import st_folium

            center_lat = geo_df["latitude"].mean()
            center_lon = geo_df["longitude"].mean()
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=4,
                tiles="cartodbpositron",
            )
            heat_data = geo_df[["latitude", "longitude"]].values.tolist()
            plugins.HeatMap(
                heat_data,
                min_opacity=0.3,
                radius=15,
                blur=10,
            ).add_to(m)
            st_folium(m, width=900, height=500)
        except ImportError:
            st.warning(
                "Install `streamlit-folium` for interactive heatmaps."
            )
    else:
        st.info("No geocoded data available for city heatmap.")

    # Regional breakdown chart
    fig = plotly_regional_breakdown(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Region stats table
    if "region" in df.columns and not df.empty:
        st.subheader("Region Statistics")
        region_stats = _build_region_stats(df)
        if not region_stats.empty:
            st.dataframe(
                region_stats,
                use_container_width=True,
                hide_index=True,
            )


def render_category_cluster(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    """Render the Category/Cluster tab."""
    st.header("Job Categories")

    col_left, col_right = st.columns(2)

    with col_left:
        fig = plotly_cluster_distribution(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cluster data available.")

    with col_right:
        fig = plotly_jobs_by_category(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Cluster summary table
    if not cluster_summary.empty:
        st.subheader("Category Details")
        st.dataframe(
            cluster_summary,
            use_container_width=True,
            hide_index=True,
        )

    # Salary by cluster
    fig = plotly_salary_by_cluster(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def render_salary_analysis(df: pd.DataFrame) -> None:
    """Render the Salary Analysis tab."""
    st.header("Salary Analysis")

    # Check if any salary data exists
    has_salary = (
        "salary_max_annual" in df.columns
        and df["salary_max_annual"].notna().any()
    )

    if not has_salary:
        st.warning(
            "No salary data available. Salary information may not be "
            "provided by all job boards."
        )
        return

    # Overall salary stats
    sal_data = df["salary_max_annual"].dropna()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Median", f"${sal_data.median():,.0f}")
    with col2:
        st.metric("Mean", f"${sal_data.mean():,.0f}")
    with col3:
        st.metric("25th Pctl", f"${sal_data.quantile(0.25):,.0f}")
    with col4:
        st.metric("75th Pctl", f"${sal_data.quantile(0.75):,.0f}")

    st.divider()

    # Salary histogram
    fig = plotly_salary_histogram(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Salary by region
    fig = plotly_salary_by_region(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Salary by cluster
    fig = plotly_salary_by_cluster(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def render_experience_analysis(df: pd.DataFrame) -> None:
    """Render the Experience Analysis tab."""
    st.header("Experience Requirements")

    has_yoe = (
        "yoe_min" in df.columns
        and df["yoe_min"].notna().any()
    )

    if not has_yoe:
        st.warning(
            "No experience data available. YOE is extracted from "
            "job descriptions — ensure descriptions are being fetched."
        )
        return

    # YOE stats
    yoe_data = df["yoe_min"].dropna()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Median YOE", f"{yoe_data.median():.1f} years")
    with col2:
        st.metric("Most Common", f"{yoe_data.mode().iloc[0]:.0f} years")
    with col3:
        pct_entry = (yoe_data <= 2).mean() * 100
        st.metric("Entry Level (0-2 yrs)", f"{pct_entry:.0f}%")

    st.divider()

    # YOE distribution
    fig = plotly_yoe_distribution(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # YOE by category
    fig = plotly_yoe_by_category(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def render_skills(
    df: pd.DataFrame,
    skill_freq: Dict[str, Counter],
) -> None:
    """Render the Skills tab."""
    st.header("Skills & Technology Demand")

    # Top skills chart
    fig = plotly_skills_demand(skill_freq)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "No skill data available. Skills are extracted from "
            "job descriptions — ensure descriptions are being fetched."
        )
        return

    # Per-category skill breakdown
    st.subheader("Skills by Category")
    for category, counter in skill_freq.items():
        if counter:
            category_name = category.replace("_", " ").title()
            with st.expander(f"{category_name}", expanded=False):
                if isinstance(counter, Counter):
                    items = counter.most_common(10)
                elif isinstance(counter, dict):
                    items = sorted(
                        counter.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                else:
                    continue

                skill_df = pd.DataFrame(
                    items, columns=["Skill", "Count"]
                )
                st.dataframe(
                    skill_df,
                    use_container_width=True,
                    hide_index=True,
                )

    # Top companies
    st.divider()
    fig = plotly_top_companies(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


def render_career_advisor(career_suggestions: List[str]) -> None:
    """Render the Career Advisor tab."""
    st.header("Career Enhancement Suggestions")

    if not career_suggestions:
        st.info(
            "No career suggestions available. Run the full pipeline "
            "to generate insights."
        )
        return

    for suggestion in career_suggestions:
        if ":" in suggestion:
            label, body = suggestion.split(":", 1)
            with st.expander(label.strip(), expanded=True):
                st.write(body.strip())
        else:
            st.write(suggestion)


def render_data_explorer(df: pd.DataFrame) -> None:
    """Render the Data Explorer tab with raw data table."""
    st.header("Data Explorer")
    st.write(f"**{len(df):,}** job listings")

    # Select columns to display
    display_cols = [
        c for c in [
            "title", "company", "location", "region",
            "cluster_label", "salary_min_annual",
            "salary_max_annual", "yoe_min", "yoe_max",
            "date_posted", "url",
        ]
        if c in df.columns
    ]

    if not display_cols:
        st.warning("No displayable columns found.")
        return

    # Search filter
    search_term = st.text_input(
        "Search jobs (title, company, location):",
        placeholder="e.g., Machine Learning, Google, New York",
    )

    display_df = df[display_cols].copy()
    if search_term:
        search_lower = search_term.lower()
        mask = pd.Series(False, index=display_df.index)
        for col in ["title", "company", "location"]:
            if col in display_df.columns:
                mask = mask | (
                    display_df[col]
                    .fillna("")
                    .str.lower()
                    .str.contains(search_lower, na=False)
                )
        display_df = display_df[mask]
        st.write(f"**{len(display_df):,}** matching results")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    # CSV download
    csv = display_df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="filtered_jobs.csv",
        mime="text/csv",
    )


def _build_region_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary stats table by region."""
    if "region" not in df.columns or df.empty:
        return pd.DataFrame()

    groups = df.groupby("region")
    stats = []

    for region, group in groups:
        row = {"Region": region, "Jobs": len(group)}

        if "company" in group.columns:
            row["Companies"] = group["company"].nunique()

        if "salary_max_annual" in group.columns:
            sal = group["salary_max_annual"].dropna()
            if not sal.empty:
                row["Median Salary"] = f"${sal.median():,.0f}"
            else:
                row["Median Salary"] = "N/A"
        else:
            row["Median Salary"] = "N/A"

        if "yoe_min" in group.columns:
            yoe = group["yoe_min"].dropna()
            if not yoe.empty:
                row["Median YOE"] = f"{yoe.median():.1f}"
            else:
                row["Median YOE"] = "N/A"
        else:
            row["Median YOE"] = "N/A"

        stats.append(row)

    result = pd.DataFrame(stats)
    if not result.empty:
        result = result.sort_values("Jobs", ascending=False)
    return result
