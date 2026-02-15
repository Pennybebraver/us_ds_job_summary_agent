"""
Integration tests for the job summary agent pipeline.
Tests the full flow with mock data (no actual web scraping).
"""

import os
import pandas as pd
import pytest

from src.processor.data_processor import process_jobs
from src.nlp.summarizer import summarize_jobs, get_skill_frequency
from src.nlp.clustering import cluster_jobs, get_cluster_summary
from src.visualization.charts import create_all_charts
from src.reports.pdf_generator import generate_report
from src.career.advisor import generate_career_suggestions


@pytest.fixture
def mock_scraped_data():
    """Create realistic mock scraped data."""
    return pd.DataFrame({
        "title": [
            "Senior Data Scientist - NLP",
            "Machine Learning Engineer",
            "Applied Scientist - Computer Vision",
            "Data Scientist - Fraud Detection",
            "ML Engineer - LLM/GenAI",
            "Research Scientist - NLP",
            "Data Scientist - A/B Testing",
            "Senior ML Engineer - MLOps",
            "Data Scientist - Recommendation Systems",
            "Applied Scientist - Robotics",
        ],
        "company": [
            "Google", "Meta", "Amazon", "Stripe",
            "OpenAI", "Microsoft", "Netflix", "Uber",
            "Spotify", "Tesla",
        ],
        "location": [
            "New York, NY", "San Francisco, CA", "Seattle, WA",
            "San Francisco, CA", "San Francisco, CA", "Boston, MA",
            "New York, NY", "Seattle, WA", "New York, NY",
            "Palo Alto, CA",
        ],
        "description": [
            "We need a data scientist with 3-5 years of experience in NLP "
            "and natural language processing. Must know Python, PyTorch, "
            "and transformers. Work on text classification and sentiment "
            "analysis. Experience with deep learning required. Salary "
            "range $150,000 - $200,000.",

            "Looking for an ML engineer with 5+ years experience in "
            "machine learning. Proficiency in Python, TensorFlow, and "
            "AWS SageMaker required. Build and deploy ML pipelines. "
            "Knowledge of Kubernetes and Docker preferred.",

            "Applied Scientist role focused on computer vision and "
            "image recognition. 4-7 years experience needed. Work with "
            "CNNs, object detection, and image classification. Python, "
            "PyTorch, OpenCV expertise required.",

            "Data Scientist for fraud detection and risk modeling. "
            "3+ years experience in anomaly detection and statistical "
            "analysis. Python, SQL, scikit-learn. Knowledge of "
            "financial crime and compliance helpful.",

            "ML Engineer to work on large language models and "
            "generative AI. Experience with LLM fine-tuning, RAG, "
            "prompt engineering. Python, PyTorch, LangChain. "
            "2-5 years experience. $180,000 - $250,000.",

            "Research Scientist for NLP team. PhD preferred. "
            "Experience with natural language processing, text mining, "
            "and information extraction. 3+ years. Python, "
            "deep learning, transformers.",

            "Data Scientist for experimentation and A/B testing. "
            "Statistical analysis, causal inference, product analytics. "
            "Python, SQL, R. 2-4 years experience. "
            "$130,000 - $170,000.",

            "Senior ML Engineer for MLOps and ML infrastructure. "
            "Model deployment, CI/CD, feature store, model monitoring. "
            "Python, Kubernetes, Docker, AWS. 5+ years. "
            "$160,000 - $220,000.",

            "Data Scientist working on recommendation systems and "
            "personalization. Collaborative filtering, ranking. "
            "Python, Spark, SQL. 3-5 years experience. "
            "$140,000 - $190,000.",

            "Applied Scientist for autonomous driving and robotics. "
            "Sensor fusion, perception, planning. Python, C++, "
            "ROS. 4+ years experience.",
        ],
        "salary_min": [
            150000, None, None, None, 180000,
            None, 130000, 160000, 140000, None,
        ],
        "salary_max": [
            200000, None, None, None, 250000,
            None, 170000, 220000, 190000, None,
        ],
        "url": [f"http://example.com/job/{i}" for i in range(10)],
        "date_posted": ["2024-01-15"] * 10,
        "site": ["linkedin"] * 5 + ["indeed"] * 5,
        "search_query": [
            "Data Scientist", "ML Engineer", "Applied Scientist",
            "Data Scientist", "ML Engineer", "Research Scientist",
            "Data Scientist", "ML Engineer", "Data Scientist",
            "Applied Scientist",
        ],
        "search_location": [
            "New York, NY", "San Francisco, CA", "Seattle, WA",
            "San Francisco, CA", "San Francisco, CA", "Boston, MA",
            "New York, NY", "Seattle, WA", "New York, NY",
            "San Francisco, CA",
        ],
    })


class TestFullPipeline:
    """Integration test for the complete pipeline."""

    def test_process_then_summarize(self, mock_scraped_data):
        """Test processing followed by summarization."""
        processed = process_jobs(mock_scraped_data)

        assert "latitude" in processed.columns
        assert "region" in processed.columns
        assert "yoe_min" in processed.columns

        summarized = summarize_jobs(processed)
        assert "summary" in summarized.columns
        assert "key_skills" in summarized.columns

    def test_process_cluster_report(self, mock_scraped_data, tmp_path):
        """Test the full pipeline: process → cluster → report."""
        # Process
        processed = process_jobs(mock_scraped_data)
        assert len(processed) == 10

        # Summarize
        processed = summarize_jobs(processed)

        # Cluster (use TF-IDF for speed in tests)
        clustered = cluster_jobs(
            processed,
            min_clusters=3,
            max_clusters=5,
            use_transformers=False,
        )
        assert "cluster_id" in clustered.columns
        assert "cluster_label" in clustered.columns

        # Verify clusters are assigned
        assert clustered["cluster_id"].nunique() >= 2

        # Get summaries
        cluster_summary = get_cluster_summary(clustered)
        assert not cluster_summary.empty

        # Skill frequency
        skill_freq = get_skill_frequency(clustered)
        assert "languages" in skill_freq

        # Charts
        output_dir = str(tmp_path / "outputs")
        charts = create_all_charts(clustered, skill_freq, output_dir)
        assert len(charts) > 0

        # Career suggestions
        career_suggestions = generate_career_suggestions(clustered, skill_freq)
        assert len(career_suggestions) > 0

        # PDF Report
        report_dir = str(tmp_path / "reports")
        report_path = generate_report(
            df=clustered,
            cluster_summary=cluster_summary,
            charts=charts,
            career_suggestions=career_suggestions,
            output_dir=report_dir,
        )
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 1000  # Non-trivial PDF

    def test_career_advisor(self, mock_scraped_data):
        """Test career advisor generates meaningful suggestions."""
        processed = process_jobs(mock_scraped_data)
        processed = summarize_jobs(processed)

        skill_freq = get_skill_frequency(processed)
        suggestions = generate_career_suggestions(processed, skill_freq)

        assert len(suggestions) > 5
        # Should contain skill-related suggestions
        skill_related = [s for s in suggestions if "SKILL" in s.upper()]
        assert len(skill_related) > 0

    def test_empty_data_pipeline(self, tmp_path):
        """Test pipeline handles empty data gracefully."""
        empty_df = pd.DataFrame({
            "title": [], "company": [], "location": [],
            "description": [], "salary_min": [], "salary_max": [],
            "url": [], "date_posted": [], "site": [],
            "search_query": [], "search_location": [],
        })

        processed = process_jobs(empty_df)
        assert len(processed) == 0

        summarized = summarize_jobs(processed)
        assert len(summarized) == 0
