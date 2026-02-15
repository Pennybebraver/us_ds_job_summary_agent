"""
Career advisor module that analyzes job market trends and provides
actionable career enhancement suggestions for DS/ML professionals.
"""

import logging
from collections import Counter
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Skill evolution tracking - emerging vs established
EMERGING_SKILLS = {
    "llm", "large language models", "generative ai", "rag",
    "retrieval augmented generation", "fine-tuning", "langchain",
    "prompt engineering", "mlops", "feature store",
    "model monitoring", "vertex ai", "sagemaker",
    "dbt", "databricks", "ray", "jax",
    "rust", "go",
}

ESTABLISHED_SKILLS = {
    "python", "sql", "r", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "spark", "hadoop",
    "aws", "gcp", "azure", "docker", "kubernetes",
    "deep learning", "machine learning", "statistics",
}


def generate_career_suggestions(
    df: pd.DataFrame,
    skill_freq: Dict[str, Counter],
) -> List[str]:
    """
    Generate career enhancement suggestions based on job market analysis.

    Args:
        df: Processed and clustered job DataFrame.
        skill_freq: Skill frequency data from summarizer.

    Returns:
        List of career suggestion strings.
    """
    suggestions = []

    # 1. Top skills to learn
    skill_suggestions = _analyze_skill_gaps(skill_freq)
    suggestions.extend(skill_suggestions)

    # 2. Emerging category insights
    category_suggestions = _analyze_category_trends(df)
    suggestions.extend(category_suggestions)

    # 3. Salary optimization
    salary_suggestions = _analyze_salary_trends(df)
    suggestions.extend(salary_suggestions)

    # 4. Geographic opportunities
    geo_suggestions = _analyze_geographic_trends(df)
    suggestions.extend(geo_suggestions)

    # 5. Experience level insights
    yoe_suggestions = _analyze_yoe_trends(df)
    suggestions.extend(yoe_suggestions)

    # 6. General career advice
    suggestions.extend(_general_advice(df, skill_freq))

    return suggestions


def _analyze_skill_gaps(skill_freq: Dict[str, Counter]) -> List[str]:
    """Identify trending skills professionals should learn."""
    suggestions = []

    # Flatten all skills
    all_skills = Counter()
    for counter in skill_freq.values():
        all_skills.update(counter)

    if not all_skills:
        return suggestions

    # Top emerging skills
    emerging_counts = {
        skill: count for skill, count in all_skills.items()
        if skill in EMERGING_SKILLS
    }

    if emerging_counts:
        top_emerging = sorted(
            emerging_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        skills_str = ", ".join(s[0] for s in top_emerging)
        suggestions.append(
            f"TRENDING SKILLS: The most in-demand emerging skills are: "
            f"{skills_str}. Prioritize learning these to stay competitive."
        )

    # Most requested technical skills
    top_5 = all_skills.most_common(5)
    if top_5:
        skills_str = ", ".join(s[0] for s in top_5)
        suggestions.append(
            f"MUST-HAVE SKILLS: The top 5 most requested skills are: "
            f"{skills_str}. Ensure proficiency in these fundamentals."
        )

    # Cloud platform trends
    cloud_skills = skill_freq.get("cloud", Counter())
    if cloud_skills:
        top_cloud = cloud_skills.most_common(1)
        if top_cloud:
            suggestions.append(
                f"CLOUD PLATFORM: {top_cloud[0][0].upper()} is the most "
                f"requested cloud platform. Consider getting certified."
            )

    return suggestions


def _analyze_category_trends(df: pd.DataFrame) -> List[str]:
    """Analyze job category trends."""
    suggestions = []

    if "cluster_label" not in df.columns:
        return suggestions

    category_counts = df["cluster_label"].value_counts()

    if category_counts.empty:
        return suggestions

    # Fastest growing categories (by count)
    top_category = category_counts.index[0]
    top_count = category_counts.iloc[0]
    pct = top_count / len(df) * 100

    suggestions.append(
        f"HOT CATEGORY: '{top_category}' represents {pct:.1f}% of all "
        f"openings ({top_count} jobs). This is the highest-demand area."
    )

    # Niche opportunities (smaller but potentially less competitive)
    if len(category_counts) > 3:
        niche_categories = category_counts.tail(3)
        niche_str = ", ".join(niche_categories.index.tolist())
        suggestions.append(
            f"NICHE OPPORTUNITIES: Consider specializing in less crowded "
            f"areas like {niche_str} for potentially less competition "
            f"and unique career positioning."
        )

    return suggestions


def _analyze_salary_trends(df: pd.DataFrame) -> List[str]:
    """Analyze salary trends for career optimization."""
    suggestions = []

    if "salary_max_annual" not in df.columns:
        return suggestions

    sal_df = df.dropna(subset=["salary_max_annual"])
    if sal_df.empty:
        return suggestions

    # Overall salary stats
    median_sal = sal_df["salary_max_annual"].median()
    p75_sal = sal_df["salary_max_annual"].quantile(0.75)
    p90_sal = sal_df["salary_max_annual"].quantile(0.90)

    suggestions.append(
        f"SALARY BENCHMARKS: Median max salary is ${median_sal:,.0f}. "
        f"Top 25% earn above ${p75_sal:,.0f}, and top 10% earn above "
        f"${p90_sal:,.0f}. Use these as negotiation benchmarks."
    )

    # Highest paying categories
    if "cluster_label" in sal_df.columns:
        cat_salary = (
            sal_df.groupby("cluster_label")["salary_max_annual"]
            .median()
            .sort_values(ascending=False)
        )

        if not cat_salary.empty:
            top_paying = cat_salary.index[0]
            top_sal = cat_salary.iloc[0]
            suggestions.append(
                f"HIGHEST PAYING: '{top_paying}' roles have the highest "
                f"median salary at ${top_sal:,.0f}. Transitioning to this "
                f"area could boost earning potential."
            )

    # Highest paying regions
    if "region" in sal_df.columns:
        region_salary = (
            sal_df[sal_df["region"] != "Other"]
            .groupby("region")["salary_max_annual"]
            .median()
            .sort_values(ascending=False)
        )

        if not region_salary.empty:
            top_region = region_salary.index[0]
            top_region_sal = region_salary.iloc[0]
            suggestions.append(
                f"TOP PAYING REGION: '{top_region}' offers the highest "
                f"median salary at ${top_region_sal:,.0f}. Consider "
                f"opportunities in this area."
            )

    return suggestions


def _analyze_geographic_trends(df: pd.DataFrame) -> List[str]:
    """Analyze geographic job distribution."""
    suggestions = []

    if "region" not in df.columns:
        return suggestions

    region_counts = df[df["region"] != "Other"]["region"].value_counts()

    if region_counts.empty:
        return suggestions

    # Top hiring region
    top_region = region_counts.index[0]
    top_count = region_counts.iloc[0]

    suggestions.append(
        f"TOP HUB: '{top_region}' leads with {top_count} openings. "
        f"Major tech hubs continue to dominate DS/ML hiring."
    )

    # International opportunities
    intl_regions = ["Singapore", "Hong Kong"]
    intl_df = df[df["region"].isin(intl_regions)]
    if not intl_df.empty:
        suggestions.append(
            f"INTERNATIONAL: {len(intl_df)} openings found in "
            f"Singapore/Hong Kong. International experience can "
            f"differentiate your profile and expand opportunities."
        )

    return suggestions


def _analyze_yoe_trends(df: pd.DataFrame) -> List[str]:
    """Analyze experience requirements."""
    suggestions = []

    if "yoe_min" not in df.columns:
        return suggestions

    yoe_df = df.dropna(subset=["yoe_min"])
    if yoe_df.empty:
        return suggestions

    median_yoe = yoe_df["yoe_min"].median()
    entry_level = len(yoe_df[yoe_df["yoe_min"] <= 2])
    mid_level = len(yoe_df[(yoe_df["yoe_min"] > 2) & (yoe_df["yoe_min"] <= 5)])
    senior_level = len(yoe_df[yoe_df["yoe_min"] > 5])

    suggestions.append(
        f"EXPERIENCE: Median YOE requirement is {median_yoe:.0f} years. "
        f"Distribution: Entry-level (0-2yr): {entry_level}, "
        f"Mid-level (3-5yr): {mid_level}, Senior (5+yr): {senior_level}."
    )

    if entry_level > 0:
        entry_pct = entry_level / len(yoe_df) * 100
        suggestions.append(
            f"ENTRY-LEVEL: {entry_pct:.0f}% of roles require 0-2 years. "
            f"New graduates should focus on building strong portfolios "
            f"and contributing to open-source projects."
        )

    return suggestions


def _general_advice(
    df: pd.DataFrame,
    skill_freq: Dict[str, Counter],
) -> List[str]:
    """General career enhancement advice."""
    suggestions = [
        "PORTFOLIO: Build end-to-end ML projects that demonstrate "
        "business impact, not just model accuracy. Include deployment.",

        "NETWORKING: Engage with the DS/ML community through conferences "
        "(NeurIPS, ICML, KDD), local meetups, and online communities.",

        "CONTINUOUS LEARNING: The field evolves rapidly. Dedicate at least "
        "5 hours per week to learning new tools, reading papers, and "
        "taking courses on platforms like Coursera or fast.ai.",

        "COMMUNICATION: Strong communication skills differentiate senior "
        "from junior roles. Practice presenting technical concepts to "
        "non-technical stakeholders.",
    ]

    return suggestions
