"""
Sidebar filter logic for the Streamlit dashboard.
Renders filter controls and applies them to the DataFrame.
"""

import pandas as pd
import streamlit as st


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filter controls and return filtered DataFrame."""
    st.header("Filters")

    selected_regions = None
    selected_categories = None
    salary_range = None
    yoe_range = None

    # Region filter
    if "region" in df.columns:
        regions = sorted(df["region"].dropna().unique().tolist())
        if regions:
            selected_regions = st.multiselect(
                "Regions",
                regions,
                default=regions,
            )

    # Category filter
    if "cluster_label" in df.columns:
        categories = sorted(
            df["cluster_label"].dropna().unique().tolist()
        )
        if categories:
            selected_categories = st.multiselect(
                "Categories",
                categories,
                default=categories,
            )

    # Salary range filter
    if "salary_max_annual" in df.columns:
        sal_data = df["salary_max_annual"].dropna()
        if not sal_data.empty and len(sal_data) > 0:
            sal_min = float(sal_data.min())
            sal_max = float(sal_data.max())
            if sal_min < sal_max:
                salary_range = st.slider(
                    "Salary Range (Annual USD)",
                    min_value=sal_min,
                    max_value=sal_max,
                    value=(sal_min, sal_max),
                    step=10000.0,
                    format="$%,.0f",
                )

    # YOE range filter
    if "yoe_min" in df.columns:
        yoe_data = df["yoe_min"].dropna()
        if not yoe_data.empty and len(yoe_data) > 0:
            yoe_range = st.slider(
                "Years of Experience",
                min_value=0,
                max_value=20,
                value=(0, 20),
            )

    return apply_filters(
        df, selected_regions, selected_categories,
        salary_range, yoe_range,
    )


def apply_filters(
    df: pd.DataFrame,
    regions=None,
    categories=None,
    salary_range=None,
    yoe_range=None,
) -> pd.DataFrame:
    """Apply all selected filters to the DataFrame."""
    filtered = df.copy()

    if regions is not None and "region" in filtered.columns:
        # Keep rows that match selected regions OR have no region
        mask = filtered["region"].isin(regions) | filtered["region"].isna()
        filtered = filtered[mask]

    if categories is not None and "cluster_label" in filtered.columns:
        mask = (
            filtered["cluster_label"].isin(categories)
            | filtered["cluster_label"].isna()
        )
        filtered = filtered[mask]

    if salary_range is not None and "salary_max_annual" in filtered.columns:
        mask = (
            filtered["salary_max_annual"].isna()
            | filtered["salary_max_annual"].between(
                salary_range[0], salary_range[1]
            )
        )
        filtered = filtered[mask]

    if yoe_range is not None and "yoe_min" in filtered.columns:
        mask = (
            filtered["yoe_min"].isna()
            | filtered["yoe_min"].between(yoe_range[0], yoe_range[1])
        )
        filtered = filtered[mask]

    return filtered
