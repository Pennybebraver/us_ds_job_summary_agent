"""
Job description clustering using sentence-transformers embeddings and KMeans.
Clusters jobs into categories like: fraud detection, analytics, CV, LLM, etc.
"""

import logging
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# Pre-defined category keywords for cluster labeling
CATEGORY_KEYWORDS = {
    "LLM/GenAI": [
        "llm", "large language model", "generative ai", "gpt",
        "chatbot", "rag", "retrieval augmented", "fine-tuning",
        "prompt engineering", "langchain", "transformer",
    ],
    "NLP": [
        "nlp", "natural language", "text mining", "text classification",
        "sentiment analysis", "named entity", "information extraction",
        "speech recognition", "conversational ai",
    ],
    "Computer Vision": [
        "computer vision", "image recognition", "object detection",
        "image classification", "segmentation", "opencv", "cnn",
        "convolutional", "visual", "video analysis",
    ],
    "Analytics/A-B Testing": [
        "a/b test", "ab test", "experimentation", "causal inference",
        "statistical analysis", "product analytics", "business intelligence",
        "dashboarding", "metrics", "kpi",
    ],
    "Fraud Detection/Risk": [
        "fraud", "risk", "anomaly detection", "anti-money laundering",
        "compliance", "credit risk", "financial crime",
    ],
    "Recommendation Systems": [
        "recommendation", "recommender", "personalization",
        "collaborative filtering", "content-based filtering",
        "ranking", "search relevance",
    ],
    "MLOps/ML Engineering": [
        "mlops", "model deployment", "ml infrastructure",
        "ml pipeline", "feature store", "model monitoring",
        "ci/cd", "kubernetes", "docker", "microservice",
    ],
    "Robotics/Autonomous": [
        "robotics", "autonomous", "self-driving", "perception",
        "planning", "control systems", "sensor fusion", "lidar",
    ],
    "Time Series/Forecasting": [
        "time series", "forecasting", "demand prediction",
        "supply chain", "inventory", "pricing",
    ],
    "Healthcare/Bio": [
        "healthcare", "biotech", "clinical", "drug discovery",
        "genomics", "medical imaging", "health",
    ],
}


def cluster_jobs(
    df: pd.DataFrame,
    min_clusters: int = 5,
    max_clusters: int = 15,
    random_state: int = 42,
    use_transformers: bool = True,
) -> pd.DataFrame:
    """
    Cluster job descriptions into categories.

    Args:
        df: DataFrame with 'description' column.
        min_clusters: Minimum number of clusters to try.
        max_clusters: Maximum number of clusters to try.
        random_state: Random seed for reproducibility.
        use_transformers: Whether to use sentence-transformers (slower but better).

    Returns:
        DataFrame with added columns: cluster_id, cluster_label, embedding
    """
    if df.empty or "description" not in df.columns:
        return df

    descriptions = df["description"].fillna("").tolist()
    valid_mask = [len(d.strip()) > 50 for d in descriptions]

    if sum(valid_mask) < min_clusters:
        logger.warning(
            f"Only {sum(valid_mask)} valid descriptions, "
            f"need at least {min_clusters}. Skipping clustering."
        )
        df["cluster_id"] = -1
        df["cluster_label"] = "Uncategorized"
        return df

    logger.info(f"Clustering {sum(valid_mask)} job descriptions...")

    # Generate embeddings
    if use_transformers:
        embeddings = _get_transformer_embeddings(descriptions)
    else:
        embeddings = _get_tfidf_embeddings(descriptions)

    # Find optimal number of clusters
    n_samples = len(embeddings)
    actual_max = min(max_clusters, n_samples - 1)
    actual_min = min(min_clusters, actual_max)

    optimal_k = _find_optimal_clusters(
        embeddings, actual_min, actual_max, random_state
    )
    logger.info(f"Optimal clusters: {optimal_k}")

    # Perform clustering
    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=random_state,
        n_init=10,
    )
    cluster_ids = kmeans.fit_predict(embeddings)

    df["cluster_id"] = cluster_ids

    # Label clusters
    cluster_labels = _label_clusters(df, optimal_k)
    df["cluster_label"] = df["cluster_id"].map(cluster_labels)

    # Store embeddings for visualization
    df["embedding"] = list(embeddings)

    # Log cluster distribution
    distribution = df["cluster_label"].value_counts()
    logger.info(f"Cluster distribution:\n{distribution}")

    return df


def _get_transformer_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformers model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Truncate long texts
        truncated = [t[:2000] if len(t) > 2000 else t for t in texts]

        logger.info("Generating embeddings...")
        embeddings = model.encode(
            truncated,
            show_progress_bar=True,
            batch_size=32,
        )
        return np.array(embeddings)

    except ImportError:
        logger.warning(
            "sentence-transformers not available, falling back to TF-IDF"
        )
        return _get_tfidf_embeddings(texts)


def _get_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """Generate TF-IDF embeddings as fallback."""
    logger.info("Using TF-IDF embeddings...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix.toarray()


def _find_optimal_clusters(
    embeddings: np.ndarray,
    min_k: int,
    max_k: int,
    random_state: int,
) -> int:
    """Find optimal number of clusters using silhouette score."""
    if min_k >= max_k:
        return min_k

    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        try:
            kmeans = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
            )
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels, sample_size=min(1000, len(embeddings)))

            if score > best_score:
                best_score = score
                best_k = k

            logger.debug(f"k={k}, silhouette={score:.4f}")

        except Exception as e:
            logger.debug(f"Error with k={k}: {e}")

    return best_k


def _label_clusters(df: pd.DataFrame, n_clusters: int) -> Dict[int, str]:
    """
    Label each cluster based on keyword matching in descriptions.
    """
    labels = {}

    for cluster_id in range(n_clusters):
        cluster_descs = df[df["cluster_id"] == cluster_id]["description"].tolist()

        if not cluster_descs:
            labels[cluster_id] = f"Cluster {cluster_id}"
            continue

        # Combine all descriptions in this cluster
        combined = " ".join(cluster_descs).lower()

        # Score each category
        category_scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(combined.count(kw) for kw in keywords)
            if score > 0:
                category_scores[category] = score

        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            labels[cluster_id] = best_category
        else:
            # Fallback: use most common title words
            titles = df[df["cluster_id"] == cluster_id]["title"].tolist()
            common_words = _get_common_title_words(titles)
            labels[cluster_id] = common_words or f"Cluster {cluster_id}"

    # Handle duplicate labels by appending numbers
    seen = Counter()
    for k, v in labels.items():
        seen[v] += 1
        if seen[v] > 1:
            labels[k] = f"{v} ({seen[v]})"

    return labels


def _get_common_title_words(titles: List[str]) -> str:
    """Get the most common meaningful words from job titles."""
    stop_words = {
        "the", "a", "an", "and", "or", "in", "at", "of", "for",
        "to", "with", "is", "-", "&", "/", "senior", "junior",
        "lead", "staff", "principal", "manager", "director",
        "i", "ii", "iii", "iv", "v",
    }

    words = Counter()
    for title in titles:
        for word in title.lower().split():
            if word not in stop_words and len(word) > 2:
                words[word] += 1

    if words:
        top_words = [w for w, _ in words.most_common(3)]
        return " ".join(top_words).title()

    return ""


def get_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for each cluster.

    Returns:
        DataFrame with cluster stats: count, avg_salary, top_skills, etc.
    """
    if "cluster_label" not in df.columns:
        return pd.DataFrame()

    summaries = []

    for label in df["cluster_label"].unique():
        cluster_df = df[df["cluster_label"] == label]

        summary = {
            "cluster": label,
            "count": len(cluster_df),
            "pct": f"{len(cluster_df) / len(df) * 100:.1f}%",
        }

        # Average salary
        if "salary_min_annual" in cluster_df.columns:
            avg_sal = cluster_df["salary_min_annual"].dropna().mean()
            if not pd.isna(avg_sal):
                summary["avg_salary_min"] = f"${avg_sal:,.0f}"

        if "salary_max_annual" in cluster_df.columns:
            avg_sal = cluster_df["salary_max_annual"].dropna().mean()
            if not pd.isna(avg_sal):
                summary["avg_salary_max"] = f"${avg_sal:,.0f}"

        # Average YOE
        if "yoe_min" in cluster_df.columns:
            avg_yoe = cluster_df["yoe_min"].dropna().mean()
            if not pd.isna(avg_yoe):
                summary["avg_yoe"] = f"{avg_yoe:.1f}"

        # Top locations
        if "region" in cluster_df.columns:
            top_regions = cluster_df["region"].value_counts().head(3)
            summary["top_regions"] = ", ".join(
                f"{r} ({c})" for r, c in top_regions.items()
            )

        summaries.append(summary)

    return pd.DataFrame(summaries).sort_values("count", ascending=False)
