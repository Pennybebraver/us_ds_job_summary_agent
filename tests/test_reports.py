"""Tests for PDF report generator."""

import os
import pandas as pd
import pytest

from src.reports.pdf_generator import JobReportPDF, generate_report


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for report testing."""
    return pd.DataFrame({
        "title": ["Data Scientist", "ML Engineer", "Applied Scientist"],
        "company": ["Google", "Meta", "Amazon"],
        "location": ["NYC", "SF", "Seattle"],
        "region": ["Greater NYC", "SF Bay Area", "Seattle Area"],
        "state": ["NY", "CA", "WA"],
        "cluster_label": ["LLM/GenAI", "NLP", "Computer Vision"],
        "salary_min_annual": [120000, 130000, 140000],
        "salary_max_annual": [150000, 160000, 170000],
        "yoe_min": [3, 5, 4],
        "yoe_max": [5, 8, 7],
        "summary": ["Build LLM apps", "NLP research", "CV models"],
        "key_skills": [
            {"languages": ["python"]},
            {"languages": ["python", "sql"]},
            {"languages": ["python"], "techniques": ["computer vision"]},
        ],
    })


@pytest.fixture
def cluster_summary():
    """Sample cluster summary."""
    return pd.DataFrame({
        "cluster": ["LLM/GenAI", "NLP", "Computer Vision"],
        "count": [1, 1, 1],
        "pct": ["33.3%", "33.3%", "33.3%"],
    })


class TestJobReportPDF:
    """Tests for the PDF class."""

    def test_create_pdf(self):
        pdf = JobReportPDF()
        pdf.add_page()
        pdf.chapter_title("Test Section")
        pdf.body_text("This is a test paragraph.")
        assert pdf.page_no() >= 1

    def test_bullet_point(self):
        pdf = JobReportPDF()
        pdf.add_page()
        pdf.bullet_point("Test bullet")
        assert pdf.page_no() >= 1

    def test_add_table(self):
        pdf = JobReportPDF()
        pdf.add_page()
        headers = ["Name", "Value"]
        data = [["Test", "123"], ["Foo", "456"]]
        pdf.add_table(headers, data)
        assert pdf.page_no() >= 1


class TestGenerateReport:
    """Tests for report generation."""

    def test_generates_pdf(self, sample_df, cluster_summary, tmp_path):
        output_dir = str(tmp_path / "reports")
        report_path = generate_report(
            df=sample_df,
            cluster_summary=cluster_summary,
            charts={},
            career_suggestions=[
                "Learn LLM technologies",
                "Get AWS certified",
            ],
            output_dir=output_dir,
        )

        assert report_path is not None
        assert os.path.exists(report_path)
        assert report_path.endswith(".pdf")

        # Check file size > 0
        assert os.path.getsize(report_path) > 0

    def test_generates_with_empty_data(self, tmp_path):
        output_dir = str(tmp_path / "reports")
        df = pd.DataFrame({
            "title": [],
            "company": [],
            "region": [],
            "cluster_label": [],
        })
        cluster_summary = pd.DataFrame()

        report_path = generate_report(
            df=df,
            cluster_summary=cluster_summary,
            charts={},
            career_suggestions=["General advice"],
            output_dir=output_dir,
        )

        assert report_path is not None
        assert os.path.exists(report_path)

    def test_generates_with_charts(self, sample_df, cluster_summary, tmp_path):
        """Test report with chart images (if they exist)."""
        output_dir = str(tmp_path / "reports")
        charts = {}  # Empty charts dict is fine

        report_path = generate_report(
            df=sample_df,
            cluster_summary=cluster_summary,
            charts=charts,
            career_suggestions=["Test suggestion"],
            output_dir=output_dir,
        )

        assert os.path.exists(report_path)
