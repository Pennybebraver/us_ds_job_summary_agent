"""
Analytics charts module for generating job market insights visualizations.
"""

import logging
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)

# Style configuration
sns.set_theme(style="whitegrid")
COLORS = sns.color_palette("husl", 12)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_SQUARE = (8, 8)
DPI = 150


def create_all_charts(
    df: pd.DataFrame,
    skill_freq: Dict,
    output_dir: str = "outputs",
) -> Dict[str, str]:
    """
    Generate all analytics charts.

    Args:
        df: Processed job DataFrame.
        skill_freq: Skill frequency data from summarizer.
        output_dir: Directory to save chart images.

    Returns:
        Dict mapping chart name to file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    charts = {}

    chart_functions = [
        ("jobs_by_category", lambda: plot_jobs_by_category(df, output_dir)),
        ("salary_by_region", lambda: plot_salary_by_region(df, output_dir)),
        ("yoe_distribution", lambda: plot_yoe_distribution(df, output_dir)),
        ("top_companies", lambda: plot_top_companies(df, output_dir)),
        ("skills_demand", lambda: plot_skills_demand(skill_freq, output_dir)),
        ("cluster_distribution", lambda: plot_cluster_distribution(df, output_dir)),
        ("salary_by_cluster", lambda: plot_salary_by_cluster(df, output_dir)),
        ("regional_breakdown", lambda: plot_regional_breakdown(df, output_dir)),
    ]

    for name, func in chart_functions:
        try:
            path = func()
            if path:
                charts[name] = path
                logger.info(f"Created chart: {name}")
        except Exception as e:
            logger.warning(f"Failed to create chart '{name}': {e}")

    plt.close("all")
    return charts


def plot_jobs_by_category(
    df: pd.DataFrame,
    output_dir: str = "outputs",
) -> Optional[str]:
    """Bar chart of job count by cluster category."""
    if "cluster_label" not in df.columns:
        return None

    counts = df["cluster_label"].value_counts().head(12)
    if counts.empty:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    counts.plot(kind="barh", ax=ax, color=COLORS[:len(counts)], edgecolor="white")
    ax.set_xlabel("Number of Job Openings")
    ax.set_ylabel("")
    ax.set_title("Job Openings by Category", fontsize=16, fontweight="bold")
    ax.invert_yaxis()

    for i, v in enumerate(counts.values):
        ax.text(v + 0.5, i, str(v), va="center", fontweight="bold")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "jobs_by_category.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_salary_by_region(
    df: pd.DataFrame,
    output_dir: str = "outputs",
) -> Optional[str]:
    """Box plot of salary distribution by region."""
    sal_df = df.dropna(subset=["salary_max_annual"])
    sal_df = sal_df[sal_df["region"] != "Other"]

    if sal_df.empty or len(sal_df) < 5:
        return None

    # Filter regions with enough data
    region_counts = sal_df["region"].value_counts()
    valid_regions = region_counts[region_counts >= 3].index.tolist()
    sal_df = sal_df[sal_df["region"].isin(valid_regions)]

    if sal_df.empty:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.boxplot(
        data=sal_df,
        x="region",
        y="salary_max_annual",
        palette="Set2",
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Annual Salary (USD)")
    ax.set_title("Salary Distribution by Region", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "salary_by_region.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_yoe_distribution(
    df: pd.DataFrame,
    output_dir: str = "outputs",
) -> Optional[str]:
    """Histogram of years of experience requirements."""
    yoe_df = df.dropna(subset=["yoe_min"])
    if yoe_df.empty:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    yoe_values = yoe_df["yoe_min"].clip(0, 20)

    ax.hist(
        yoe_values,
        bins=range(0, 22),
        color=COLORS[2],
        edgecolor="white",
        alpha=0.8,
    )
    ax.set_xlabel("Minimum Years of Experience Required")
    ax.set_ylabel("Number of Job Openings")
    ax.set_title(
        "Years of Experience Requirements", fontsize=16, fontweight="bold"
    )

    # Add median line
    median_yoe = yoe_values.median()
    ax.axvline(
        median_yoe, color="red", linestyle="--", linewidth=2,
        label=f"Median: {median_yoe:.0f} years"
    )
    ax.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "yoe_distribution.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_top_companies(
    df: pd.DataFrame,
    output_dir: str = "outputs",
    top_n: int = 15,
) -> Optional[str]:
    """Horizontal bar chart of top hiring companies."""
    if "company" not in df.columns:
        return None

    company_counts = (
        df["company"]
        .value_counts()
        .head(top_n)
    )

    if company_counts.empty:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    company_counts.plot(
        kind="barh",
        ax=ax,
        color=COLORS[4],
        edgecolor="white",
    )
    ax.set_xlabel("Number of Openings")
    ax.set_ylabel("")
    ax.set_title(
        f"Top {top_n} Hiring Companies", fontsize=16, fontweight="bold"
    )
    ax.invert_yaxis()

    for i, v in enumerate(company_counts.values):
        ax.text(v + 0.2, i, str(v), va="center", fontweight="bold")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "top_companies.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_skills_demand(
    skill_freq: Dict,
    output_dir: str = "outputs",
) -> Optional[str]:
    """Bar chart of most in-demand skills."""
    all_skills = {}
    for category, counter in skill_freq.items():
        for skill, count in counter.items():
            all_skills[skill] = all_skills.get(skill, 0) + count

    if not all_skills:
        return None

    # Sort and take top 20
    sorted_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)
    top_skills = sorted_skills[:20]

    if not top_skills:
        return None

    skills, counts = zip(*top_skills)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    y_pos = range(len(skills))
    ax.barh(y_pos, counts, color=COLORS[6], edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(skills)
    ax.set_xlabel("Frequency in Job Descriptions")
    ax.set_title(
        "Top 20 Most In-Demand Skills", fontsize=16, fontweight="bold"
    )
    ax.invert_yaxis()

    for i, v in enumerate(counts):
        ax.text(v + 0.5, i, str(v), va="center", fontweight="bold")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "skills_demand.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cluster_distribution(
    df: pd.DataFrame,
    output_dir: str = "outputs",
) -> Optional[str]:
    """Pie chart of job distribution by cluster."""
    if "cluster_label" not in df.columns:
        return None

    counts = df["cluster_label"].value_counts()
    if counts.empty:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=COLORS[:len(counts)],
        startangle=90,
        pctdistance=0.85,
    )

    # Style
    for text in autotexts:
        text.set_fontsize(9)
        text.set_fontweight("bold")

    ax.set_title(
        "Job Distribution by Category", fontsize=16, fontweight="bold"
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cluster_distribution.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_salary_by_cluster(
    df: pd.DataFrame,
    output_dir: str = "outputs",
) -> Optional[str]:
    """Box plot of salary by job category cluster."""
    if "cluster_label" not in df.columns:
        return None

    sal_df = df.dropna(subset=["salary_max_annual"])
    if sal_df.empty or len(sal_df) < 5:
        return None

    # Filter clusters with enough salary data
    cluster_counts = sal_df["cluster_label"].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 3].index.tolist()
    sal_df = sal_df[sal_df["cluster_label"].isin(valid_clusters)]

    if sal_df.empty:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.boxplot(
        data=sal_df,
        x="cluster_label",
        y="salary_max_annual",
        palette="Set3",
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Annual Salary (USD)")
    ax.set_title(
        "Salary Distribution by Job Category", fontsize=16, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "salary_by_cluster.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_regional_breakdown(
    df: pd.DataFrame,
    output_dir: str = "outputs",
) -> Optional[str]:
    """Stacked bar chart showing category breakdown by region."""
    if "cluster_label" not in df.columns or "region" not in df.columns:
        return None

    df_filtered = df[df["region"] != "Other"]
    if df_filtered.empty:
        return None

    cross_tab = pd.crosstab(df_filtered["region"], df_filtered["cluster_label"])
    if cross_tab.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    cross_tab.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        colormap="tab20",
        edgecolor="white",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Number of Openings")
    ax.set_title(
        "Job Categories by Region", fontsize=16, fontweight="bold"
    )
    ax.legend(
        title="Category",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "regional_breakdown.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
