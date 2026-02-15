"""
Job description summarizer using extractive summarization and skill extraction.
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Common DS/ML skills to extract
SKILL_KEYWORDS = {
    "languages": [
        "python", "r", "sql", "java", "scala", "c++", "julia",
        "javascript", "typescript", "go", "rust", "matlab",
    ],
    "ml_frameworks": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "catboost", "hugging face", "huggingface",
        "transformers", "jax", "mlflow", "wandb",
    ],
    "data_tools": [
        "spark", "hadoop", "hive", "kafka", "airflow", "dbt",
        "snowflake", "redshift", "bigquery", "databricks", "dask",
        "ray", "presto", "athena",
    ],
    "cloud": [
        "aws", "gcp", "azure", "sagemaker", "vertex ai",
        "ec2", "s3", "lambda", "emr",
    ],
    "techniques": [
        "deep learning", "machine learning", "nlp",
        "natural language processing", "computer vision",
        "reinforcement learning", "recommendation systems",
        "time series", "forecasting", "a/b testing",
        "causal inference", "bayesian", "generative ai",
        "large language models", "llm", "rag",
        "retrieval augmented generation", "fine-tuning",
        "neural networks", "gradient boosting",
        "random forest", "regression", "classification",
        "clustering", "dimensionality reduction",
        "feature engineering", "mlops", "model deployment",
    ],
}


def summarize_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary and extracted skills to job DataFrame.

    Args:
        df: DataFrame with 'description' column.

    Returns:
        DataFrame with added columns: summary, key_skills, responsibilities
    """
    if df.empty or "description" not in df.columns:
        return df

    logger.info(f"Summarizing {len(df)} job descriptions...")

    summaries = []
    skills_list = []
    responsibilities_list = []

    for idx, row in df.iterrows():
        desc = str(row.get("description", ""))

        summary = _extractive_summary(desc, num_sentences=3)
        skills = extract_skills(desc)
        responsibilities = _extract_responsibilities(desc)

        summaries.append(summary)
        skills_list.append(skills)
        responsibilities_list.append(responsibilities)

    df["summary"] = summaries
    df["key_skills"] = skills_list
    df["responsibilities"] = responsibilities_list

    logger.info("Summarization complete")
    return df


def _extractive_summary(text: str, num_sentences: int = 3) -> str:
    """
    Create extractive summary by selecting most informative sentences.
    Uses a simple scoring heuristic based on keyword density.
    """
    if not text or len(text) < 50:
        return text

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Score sentences by keyword relevance
    all_keywords = []
    for category_keywords in SKILL_KEYWORDS.values():
        all_keywords.extend(category_keywords)

    important_words = set(all_keywords + [
        "responsible", "develop", "build", "lead", "design",
        "implement", "analyze", "collaborate", "research",
        "experience", "required", "preferred", "team",
    ])

    scored = []
    for i, sentence in enumerate(sentences):
        words = sentence.lower().split()
        score = sum(1 for w in words if w in important_words)
        # Boost earlier sentences (usually more important)
        position_boost = max(0, 1.0 - (i / len(sentences)) * 0.5)
        score *= (1 + position_boost)
        scored.append((score, i, sentence))

    # Select top sentences, maintain original order
    scored.sort(reverse=True)
    top_indices = sorted([s[1] for s in scored[:num_sentences]])
    selected = [sentences[i] for i in top_indices]

    return " ".join(selected)


def extract_skills(text: str) -> Dict[str, List[str]]:
    """
    Extract skills from job description text.

    Returns:
        Dict mapping skill category to list of found skills.
    """
    text_lower = text.lower()
    found_skills = {}

    for category, keywords in SKILL_KEYWORDS.items():
        matched = []
        for keyword in keywords:
            # Use word boundary matching for short keywords
            if len(keyword) <= 3:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    matched.append(keyword)
            else:
                if keyword in text_lower:
                    matched.append(keyword)

        if matched:
            found_skills[category] = matched

    return found_skills


def _extract_responsibilities(text: str) -> List[str]:
    """Extract key responsibilities from job description."""
    responsibilities = []

    # Look for bullet-pointed or listed responsibilities
    patterns = [
        r'[-•]\s*(.{20,150}?)(?=[-•\n]|$)',
        r'\d+\.\s*(.{20,150}?)(?=\d+\.|$)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        responsibilities.extend(matches)

    # Deduplicate and limit
    seen = set()
    unique = []
    for r in responsibilities:
        r_clean = r.strip()
        if r_clean not in seen and len(r_clean) > 20:
            seen.add(r_clean)
            unique.append(r_clean)

    return unique[:10]  # Top 10 responsibilities


def get_skill_frequency(df: pd.DataFrame) -> Dict[str, Counter]:
    """
    Compute skill frequency across all jobs.

    Returns:
        Dict mapping skill category to Counter of skill frequencies.
    """
    category_counters = {cat: Counter() for cat in SKILL_KEYWORDS}

    for _, row in df.iterrows():
        skills = row.get("key_skills", {})
        if isinstance(skills, dict):
            for category, skill_list in skills.items():
                if category in category_counters:
                    category_counters[category].update(skill_list)

    return category_counters


def get_top_skills(df: pd.DataFrame, top_n: int = 20) -> List[Tuple[str, int]]:
    """Get the top N most mentioned skills across all jobs."""
    all_skills = Counter()

    for _, row in df.iterrows():
        skills = row.get("key_skills", {})
        if isinstance(skills, dict):
            for skill_list in skills.values():
                all_skills.update(skill_list)

    return all_skills.most_common(top_n)
