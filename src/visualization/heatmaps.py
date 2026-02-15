"""
Geographic visualization module for creating heatmaps of job distribution.
Supports US choropleth, city-level heatmaps, and international maps.
"""

import logging
import os
from typing import Optional

import folium
import folium.plugins as plugins
import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)


def create_us_choropleth(
    df: pd.DataFrame,
    output_path: str = "outputs/us_choropleth.png",
    title: str = "DS/ML Job Openings by State",
) -> Optional[str]:
    """
    Create a US choropleth map showing job count by state.

    Args:
        df: DataFrame with 'state' column.
        output_path: Path to save the image.
        title: Chart title.

    Returns:
        Path to saved image, or None if failed.
    """
    if df.empty or "state" not in df.columns:
        logger.warning("No state data available for choropleth")
        return None

    state_counts = (
        df[df["state"].notna()]
        .groupby("state")
        .size()
        .reset_index(name="count")
    )

    if state_counts.empty:
        logger.warning("No state-level data available")
        return None

    fig = px.choropleth(
        state_counts,
        locations="state",
        locationmode="USA-states",
        color="count",
        color_continuous_scale="YlOrRd",
        scope="usa",
        title=title,
        labels={"count": "Job Count"},
    )

    fig.update_layout(
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="white",
        font=dict(size=14),
        title_font_size=18,
        width=1000,
        height=600,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_image(output_path, engine="kaleido")
    logger.info(f"US choropleth saved to {output_path}")
    return output_path


def create_city_heatmap(
    df: pd.DataFrame,
    output_path: str = "outputs/city_heatmap.html",
    title: str = "Job Density by City",
) -> Optional[str]:
    """
    Create an interactive city-level heatmap using Folium.

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns.
        output_path: Path to save HTML file.
        title: Map title.

    Returns:
        Path to saved HTML file, or None if failed.
    """
    geo_df = df.dropna(subset=["latitude", "longitude"])

    if geo_df.empty:
        logger.warning("No geocoded data available for heatmap")
        return None

    # Center map on US
    center_lat = geo_df["latitude"].mean()
    center_lon = geo_df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles="cartodbpositron",
    )

    # Add heatmap layer
    heat_data = geo_df[["latitude", "longitude"]].values.tolist()
    plugins.HeatMap(
        heat_data,
        min_opacity=0.3,
        radius=15,
        blur=10,
        max_zoom=13,
    ).add_to(m)

    # Add title
    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
    z-index:9999;background:white;padding:10px 20px;border-radius:5px;
    box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:16px;font-weight:bold;">
    {title}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    logger.info(f"City heatmap saved to {output_path}")
    return output_path


def create_cluster_heatmap(
    df: pd.DataFrame,
    output_dir: str = "outputs",
) -> dict:
    """
    Create separate heatmaps for each job cluster.

    Args:
        df: DataFrame with cluster_label, latitude, longitude.
        output_dir: Directory to save maps.

    Returns:
        Dict mapping cluster name to output file path.
    """
    if "cluster_label" not in df.columns:
        logger.warning("No cluster data available for heatmaps")
        return {}

    output_paths = {}
    os.makedirs(output_dir, exist_ok=True)

    for label in df["cluster_label"].unique():
        cluster_df = df[df["cluster_label"] == label]
        geo_df = cluster_df.dropna(subset=["latitude", "longitude"])

        if len(geo_df) < 2:
            continue

        safe_name = label.replace("/", "_").replace(" ", "_").lower()
        output_path = os.path.join(output_dir, f"heatmap_{safe_name}.html")

        create_city_heatmap(
            geo_df,
            output_path=output_path,
            title=f"Job Distribution: {label}",
        )
        output_paths[label] = output_path

    logger.info(f"Created {len(output_paths)} cluster heatmaps")
    return output_paths


def create_international_map(
    df: pd.DataFrame,
    output_path: str = "outputs/international_heatmap.html",
) -> Optional[str]:
    """
    Create a map showing job distribution in Singapore and Hong Kong.

    Args:
        df: DataFrame with latitude, longitude, region.
        output_path: Path to save HTML file.

    Returns:
        Path to saved HTML file, or None if failed.
    """
    intl_regions = ["Singapore", "Hong Kong"]
    intl_df = df[df["region"].isin(intl_regions)]
    geo_df = intl_df.dropna(subset=["latitude", "longitude"])

    if geo_df.empty:
        logger.warning("No international job data available")
        return None

    # Center on Southeast Asia
    m = folium.Map(
        location=[15.0, 110.0],
        zoom_start=4,
        tiles="cartodbpositron",
    )

    # Add markers with cluster info
    for _, row in geo_df.iterrows():
        popup_text = (
            f"<b>{row.get('title', 'N/A')}</b><br>"
            f"Company: {row.get('company', 'N/A')}<br>"
            f"Location: {row.get('location', 'N/A')}<br>"
            f"Category: {row.get('cluster_label', 'N/A')}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            popup=popup_text,
            color="red",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    # Add heatmap layer
    heat_data = geo_df[["latitude", "longitude"]].values.tolist()
    if heat_data:
        plugins.HeatMap(
            heat_data,
            min_opacity=0.3,
            radius=20,
            blur=15,
        ).add_to(m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    logger.info(f"International map saved to {output_path}")
    return output_path


def create_region_comparison_chart(
    df: pd.DataFrame,
    output_path: str = "outputs/region_comparison.png",
) -> Optional[str]:
    """
    Create a bar chart comparing job counts across regions.

    Returns:
        Path to saved image.
    """
    if "region" not in df.columns:
        return None

    region_counts = (
        df["region"]
        .value_counts()
        .reset_index()
    )
    region_counts.columns = ["Region", "Count"]
    region_counts = region_counts[region_counts["Region"] != "Other"]

    if region_counts.empty:
        return None

    fig = px.bar(
        region_counts,
        x="Region",
        y="Count",
        title="Job Openings by Region",
        color="Count",
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        width=900,
        height=500,
        font=dict(size=12),
        title_font_size=16,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_image(output_path, engine="kaleido")
    logger.info(f"Region comparison chart saved to {output_path}")
    return output_path
