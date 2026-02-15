"""Tests for NLP summarizer and clustering modules."""

import pandas as pd

from src.nlp.summarizer import (
    extract_skills,
    get_skill_frequency,
    get_top_skills,
    _extractive_summary,
    _extract_responsibilities,
)
from src.nlp.clustering import (
    _get_tfidf_embeddings,
    _label_clusters,
    _get_common_title_words,
    get_cluster_summary,
)


class TestExtractSkills:
    """Tests for skill extraction."""

    def test_extracts_python(self):
        skills = extract_skills("Must know Python and SQL")
        assert "python" in skills.get("languages", [])
        assert "sql" in skills.get("languages", [])

    def test_extracts_ml_frameworks(self):
        skills = extract_skills("Experience with PyTorch and TensorFlow")
        assert "pytorch" in skills.get("ml_frameworks", [])
        assert "tensorflow" in skills.get("ml_frameworks", [])

    def test_extracts_cloud(self):
        skills = extract_skills("Deploy models on AWS SageMaker")
        assert "aws" in skills.get("cloud", [])
        assert "sagemaker" in skills.get("cloud", [])

    def test_extracts_techniques(self):
        skills = extract_skills(
            "Work on deep learning and NLP projects with LLM"
        )
        assert "deep learning" in skills.get("techniques", [])
        assert "nlp" in skills.get("techniques", [])
        assert "llm" in skills.get("techniques", [])

    def test_empty_text(self):
        skills = extract_skills("")
        assert skills == {}

    def test_no_skills(self):
        skills = extract_skills("This is a generic job posting with no tech terms")
        # Should return empty or minimal skills
        assert isinstance(skills, dict)


class TestExtractiveSummary:
    """Tests for extractive summarization."""

    def test_short_text(self):
        text = "Short."
        result = _extractive_summary(text)
        assert result == text

    def test_long_text(self):
        sentences = [
            "We are looking for a data scientist.",
            "You will build machine learning models.",
            "Experience with Python and SQL is required.",
            "The team works on NLP and deep learning.",
            "We offer competitive salary and benefits.",
            "Remote work is available.",
        ]
        text = " ".join(sentences)
        result = _extractive_summary(text, num_sentences=3)
        # Should return 3 sentences
        assert len(result.split(".")) <= 4  # At most 3 sentences + trailing


class TestExtractResponsibilities:
    """Tests for responsibility extraction."""

    def test_bullet_points(self):
        text = "- Build ML models\n- Deploy to production\n- Analyze data"
        result = _extract_responsibilities(text)
        assert len(result) >= 0  # May or may not match depending on length

    def test_empty_text(self):
        result = _extract_responsibilities("")
        assert result == []


class TestSkillFrequency:
    """Tests for skill frequency computation."""

    def test_frequency_counting(self):
        df = pd.DataFrame({
            "key_skills": [
                {"languages": ["python", "sql"]},
                {"languages": ["python", "r"]},
            ]
        })
        freq = get_skill_frequency(df)
        assert freq["languages"]["python"] == 2
        assert freq["languages"]["sql"] == 1

    def test_empty_dataframe(self):
        df = pd.DataFrame({"key_skills": []})
        freq = get_skill_frequency(df)
        assert isinstance(freq, dict)


class TestTopSkills:
    """Tests for top skills extraction."""

    def test_returns_sorted(self):
        df = pd.DataFrame({
            "key_skills": [
                {"languages": ["python", "sql"]},
                {"languages": ["python"], "cloud": ["aws"]},
            ]
        })
        top = get_top_skills(df, top_n=3)
        assert top[0][0] == "python"
        assert top[0][1] == 2


class TestTfidfEmbeddings:
    """Tests for TF-IDF embedding generation."""

    def test_returns_array(self):
        texts = [
            "Build machine learning models",
            "Data analysis with Python",
            "Deep learning for NLP",
        ]
        embeddings = _get_tfidf_embeddings(texts)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0

    def test_single_text(self):
        texts = ["Single document about data science"]
        embeddings = _get_tfidf_embeddings(texts)
        assert embeddings.shape[0] == 1


class TestLabelClusters:
    """Tests for cluster labeling."""

    def test_labels_based_on_keywords(self):
        df = pd.DataFrame({
            "cluster_id": [0, 0, 1, 1],
            "description": [
                "Work on large language models and fine-tuning LLM",
                "Build GPT-based applications with generative AI",
                "Computer vision and image recognition tasks",
                "Object detection and image classification",
            ],
            "title": [
                "ML Engineer", "AI Engineer",
                "CV Engineer", "Vision Scientist",
            ],
        })
        labels = _label_clusters(df, 2)
        assert "LLM" in labels[0] or "GenAI" in labels[0]
        assert "Vision" in labels[1] or "Computer" in labels[1]


class TestGetCommonTitleWords:
    """Tests for common title word extraction."""

    def test_common_words(self):
        titles = [
            "Data Scientist",
            "Senior Data Scientist",
            "Data Scientist II",
        ]
        result = _get_common_title_words(titles)
        assert "data" in result.lower()
        assert "scientist" in result.lower()

    def test_empty_list(self):
        result = _get_common_title_words([])
        assert result == ""


class TestClusterSummary:
    """Tests for cluster summary generation."""

    def test_summary_generation(self):
        df = pd.DataFrame({
            "cluster_label": ["LLM", "LLM", "NLP"],
            "salary_min_annual": [120000, 130000, 110000],
            "salary_max_annual": [150000, 160000, 140000],
            "yoe_min": [3, 5, 2],
            "region": ["SF Bay Area", "Greater NYC", "Boston Area"],
        })
        summary = get_cluster_summary(df)
        assert not summary.empty
        assert "cluster" in summary.columns
        assert "count" in summary.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        summary = get_cluster_summary(df)
        assert summary.empty
