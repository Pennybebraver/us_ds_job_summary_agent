"""
Interactive Plotly chart functions for the Streamlit dashboard.
Each function returns a plotly.graph_objects.Figure or None if data is insufficient.
"""

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Consistent color palette
COLORS = px.colors.qualitative.Set2
TEMPLATE = "plotly_white"


def plotly_jobs_by_category(df: pd.DataFrame) -> Optional[go.Figure]:
    """Horizontal bar chart of job counts by cluster/category."""
    if "cluster_label" not in df.columns or df.empty:
        return None

    counts = df["cluster_label"].value_counts().head(12)
    if counts.empty:
        return None

    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        color=counts.index,
        color_discrete_sequence=COLORS,
        labels={"x": "Number of Jobs", "y": "Category"},
        title="Job Openings by Category",
        template=TEMPLATE,
    )
    fig.update_layout(
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        height=450,
    )
    return fig


def plotly_salary_by_region(df: pd.DataFrame) -> Optional[go.Figure]:
    """Box plot of salary distribution by region."""
    if "salary_max_annual" not in df.columns or "region" not in df.columns:
        return None

    sal_df = df.dropna(subset=["salary_max_annual", "region"])
    if len(sal_df) < 5:
        return None

    # Only include regions with enough data
    region_counts = sal_df["region"].value_counts()
    valid_regions = region_counts[region_counts >= 3].index.tolist()
    sal_df = sal_df[sal_df["region"].isin(valid_regions)]

    if sal_df.empty:
        return None

    fig = px.box(
        sal_df,
        x="region",
        y="salary_max_annual",
        color="region",
        color_discrete_sequence=COLORS,
        labels={
            "salary_max_annual": "Annual Salary (USD)",
            "region": "Region",
        },
        title="Salary Distribution by Region",
        template=TEMPLATE,
    )
    fig.update_layout(
        showlegend=False,
        yaxis_tickformat="$,.0f",
        height=500,
    )
    return fig


def plotly_yoe_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """Histogram of years of experience requirements."""
    if "yoe_min" not in df.columns:
        return None

    yoe_df = df["yoe_min"].dropna()
    if yoe_df.empty:
        return None

    median_yoe = yoe_df.median()

    fig = px.histogram(
        yoe_df,
        nbins=20,
        labels={"value": "Years of Experience", "count": "Number of Jobs"},
        title="Years of Experience Requirements",
        template=TEMPLATE,
        color_discrete_sequence=[COLORS[0]],
    )
    fig.add_vline(
        x=median_yoe,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {median_yoe:.1f} yrs",
        annotation_position="top right",
    )
    fig.update_layout(
        showlegend=False,
        height=400,
    )
    return fig


def plotly_top_companies(
    df: pd.DataFrame, top_n: int = 15
) -> Optional[go.Figure]:
    """Horizontal bar chart of top hiring companies."""
    if "company" not in df.columns or df.empty:
        return None

    companies = df["company"].dropna()
    if companies.empty:
        return None

    counts = companies.value_counts().head(top_n)

    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        color_discrete_sequence=[COLORS[1]],
        labels={"x": "Number of Openings", "y": "Company"},
        title=f"Top {top_n} Hiring Companies",
        template=TEMPLATE,
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(400, top_n * 30),
    )
    return fig


def plotly_skills_demand(
    skill_freq: Dict[str, Counter],
    top_n: int = 20,
) -> Optional[go.Figure]:
    """Horizontal bar chart of most in-demand skills."""
    if not skill_freq:
        return None

    all_skills: Counter = Counter()
    for category_counter in skill_freq.values():
        if isinstance(category_counter, Counter):
            all_skills.update(category_counter)
        elif isinstance(category_counter, dict):
            all_skills.update(category_counter)

    if not all_skills:
        return None

    top_skills: List[Tuple[str, int]] = all_skills.most_common(top_n)
    skills, counts = zip(*top_skills)

    fig = px.bar(
        x=list(counts),
        y=list(skills),
        orientation="h",
        color_discrete_sequence=[COLORS[2]],
        labels={"x": "Frequency", "y": "Skill"},
        title=f"Top {top_n} Most In-Demand Skills",
        template=TEMPLATE,
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(400, top_n * 25),
    )
    return fig


def plotly_cluster_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """Donut chart showing job distribution by cluster."""
    if "cluster_label" not in df.columns or df.empty:
        return None

    counts = df["cluster_label"].value_counts()
    if counts.empty:
        return None

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        color_discrete_sequence=COLORS,
        title="Job Distribution by Category",
        template=TEMPLATE,
        hole=0.3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=450)
    return fig


def plotly_salary_by_cluster(df: pd.DataFrame) -> Optional[go.Figure]:
    """Box plot of salary distribution by job category."""
    if (
        "salary_max_annual" not in df.columns
        or "cluster_label" not in df.columns
    ):
        return None

    sal_df = df.dropna(subset=["salary_max_annual", "cluster_label"])
    if len(sal_df) < 5:
        return None

    # Only show clusters with enough salary data
    cluster_counts = sal_df["cluster_label"].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 3].index.tolist()
    sal_df = sal_df[sal_df["cluster_label"].isin(valid_clusters)]

    if sal_df.empty:
        return None

    fig = px.box(
        sal_df,
        x="cluster_label",
        y="salary_max_annual",
        color="cluster_label",
        color_discrete_sequence=COLORS,
        labels={
            "salary_max_annual": "Annual Salary (USD)",
            "cluster_label": "Category",
        },
        title="Salary Distribution by Job Category",
        template=TEMPLATE,
    )
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        yaxis_tickformat="$,.0f",
        height=500,
    )
    return fig


def plotly_regional_breakdown(df: pd.DataFrame) -> Optional[go.Figure]:
    """Stacked bar chart of categories by region."""
    if (
        "region" not in df.columns
        or "cluster_label" not in df.columns
        or df.empty
    ):
        return None

    cross_tab = pd.crosstab(df["region"], df["cluster_label"])
    if cross_tab.empty:
        return None

    melted = cross_tab.reset_index().melt(
        id_vars="region",
        var_name="Category",
        value_name="Count",
    )

    fig = px.bar(
        melted,
        x="region",
        y="Count",
        color="Category",
        barmode="stack",
        color_discrete_sequence=COLORS,
        labels={"region": "Region", "Count": "Number of Jobs"},
        title="Job Categories by Region",
        template=TEMPLATE,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
    )
    return fig


def plotly_us_choropleth(df: pd.DataFrame) -> Optional[go.Figure]:
    """US state-level choropleth map of job density."""
    if "state" not in df.columns or df.empty:
        return None

    state_counts = df["state"].dropna().value_counts().reset_index()
    state_counts.columns = ["state", "count"]

    if state_counts.empty:
        return None

    fig = px.choropleth(
        state_counts,
        locations="state",
        locationmode="USA-states",
        color="count",
        color_continuous_scale="Blues",
        scope="usa",
        labels={"count": "Job Count", "state": "State"},
        title="DS/ML Job Openings by State",
        template=TEMPLATE,
    )
    fig.update_layout(
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        height=500,
    )
    return fig


def plotly_salary_histogram(df: pd.DataFrame) -> Optional[go.Figure]:
    """Histogram of overall salary distribution."""
    if "salary_max_annual" not in df.columns:
        return None

    sal_data = df["salary_max_annual"].dropna()
    if sal_data.empty:
        return None

    median_sal = sal_data.median()

    fig = px.histogram(
        sal_data,
        nbins=30,
        labels={"value": "Annual Salary (USD)", "count": "Number of Jobs"},
        title="Overall Salary Distribution",
        template=TEMPLATE,
        color_discrete_sequence=[COLORS[3]],
    )
    fig.add_vline(
        x=median_sal,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: ${median_sal:,.0f}",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_tickformat="$,.0f",
        showlegend=False,
        height=400,
    )
    return fig


def plotly_yoe_by_category(df: pd.DataFrame) -> Optional[go.Figure]:
    """Box plot of years of experience by job category."""
    if (
        "yoe_min" not in df.columns
        or "cluster_label" not in df.columns
    ):
        return None

    yoe_df = df.dropna(subset=["yoe_min", "cluster_label"])
    if len(yoe_df) < 5:
        return None

    # Only show clusters with enough data
    cluster_counts = yoe_df["cluster_label"].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 3].index.tolist()
    yoe_df = yoe_df[yoe_df["cluster_label"].isin(valid_clusters)]

    if yoe_df.empty:
        return None

    fig = px.box(
        yoe_df,
        x="cluster_label",
        y="yoe_min",
        color="cluster_label",
        color_discrete_sequence=COLORS,
        labels={
            "yoe_min": "Years of Experience",
            "cluster_label": "Category",
        },
        title="Experience Requirements by Category",
        template=TEMPLATE,
    )
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=500,
    )
    return fig
