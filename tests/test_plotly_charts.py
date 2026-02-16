"""Tests for Plotly chart functions."""

from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.dashboard.plotly_charts import (
    plotly_cluster_distribution,
    plotly_jobs_by_category,
    plotly_regional_breakdown,
    plotly_salary_by_cluster,
    plotly_salary_by_region,
    plotly_salary_histogram,
    plotly_skills_demand,
    plotly_top_companies,
    plotly_us_choropleth,
    plotly_yoe_by_category,
    plotly_yoe_distribution,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with all required columns."""
    return pd.DataFrame({
        "title": ["DS"] * 20 + ["MLE"] * 15 + ["AS"] * 10,
        "company": (
            ["Google"] * 10 + ["Meta"] * 10
            + ["Amazon"] * 10 + ["Apple"] * 5 + ["Netflix"] * 10
        ),
        "region": (
            ["SF Bay Area"] * 15 + ["Greater NYC"] * 10
            + ["Seattle Area"] * 10 + ["Boston Area"] * 10
        ),
        "cluster_label": (
            ["LLM/GenAI"] * 12 + ["NLP"] * 10
            + ["Computer Vision"] * 8 + ["MLOps"] * 8
            + ["Analytics"] * 7
        ),
        "salary_max_annual": (
            [200000] * 10 + [180000] * 10
            + [220000] * 10 + [160000] * 10 + [190000] * 5
        ),
        "yoe_min": (
            [3] * 10 + [5] * 10 + [7] * 10
            + [2] * 10 + [10] * 5
        ),
        "state": (
            ["CA"] * 15 + ["NY"] * 10
            + ["WA"] * 10 + ["MA"] * 10
        ),
    })


@pytest.fixture
def sample_skill_freq():
    """Create a sample skill frequency dict."""
    return {
        "languages": Counter({
            "python": 30, "sql": 25, "r": 10,
        }),
        "ml_frameworks": Counter({
            "tensorflow": 20, "pytorch": 18,
        }),
        "cloud": Counter({
            "aws": 15, "gcp": 12,
        }),
    }


class TestPlolyChartsWithData:
    """Test Plotly chart functions return figures with valid data."""

    def test_jobs_by_category(self, sample_df):
        fig = plotly_jobs_by_category(sample_df)
        assert isinstance(fig, go.Figure)

    def test_salary_by_region(self, sample_df):
        fig = plotly_salary_by_region(sample_df)
        assert isinstance(fig, go.Figure)

    def test_yoe_distribution(self, sample_df):
        fig = plotly_yoe_distribution(sample_df)
        assert isinstance(fig, go.Figure)

    def test_top_companies(self, sample_df):
        fig = plotly_top_companies(sample_df)
        assert isinstance(fig, go.Figure)

    def test_skills_demand(self, sample_skill_freq):
        fig = plotly_skills_demand(sample_skill_freq)
        assert isinstance(fig, go.Figure)

    def test_cluster_distribution(self, sample_df):
        fig = plotly_cluster_distribution(sample_df)
        assert isinstance(fig, go.Figure)

    def test_salary_by_cluster(self, sample_df):
        fig = plotly_salary_by_cluster(sample_df)
        assert isinstance(fig, go.Figure)

    def test_regional_breakdown(self, sample_df):
        fig = plotly_regional_breakdown(sample_df)
        assert isinstance(fig, go.Figure)

    def test_us_choropleth(self, sample_df):
        fig = plotly_us_choropleth(sample_df)
        assert isinstance(fig, go.Figure)

    def test_salary_histogram(self, sample_df):
        fig = plotly_salary_histogram(sample_df)
        assert isinstance(fig, go.Figure)

    def test_yoe_by_category(self, sample_df):
        fig = plotly_yoe_by_category(sample_df)
        assert isinstance(fig, go.Figure)


class TestPlotlyChartsWithEmptyData:
    """Test Plotly chart functions return None with empty data."""

    def test_jobs_by_category_empty(self):
        df = pd.DataFrame()
        assert plotly_jobs_by_category(df) is None

    def test_salary_by_region_empty(self):
        df = pd.DataFrame(columns=["salary_max_annual", "region"])
        assert plotly_salary_by_region(df) is None

    def test_yoe_distribution_empty(self):
        df = pd.DataFrame({"yoe_min": pd.Series(dtype=float)})
        assert plotly_yoe_distribution(df) is None

    def test_top_companies_empty(self):
        df = pd.DataFrame()
        assert plotly_top_companies(df) is None

    def test_skills_demand_empty(self):
        assert plotly_skills_demand({}) is None

    def test_cluster_distribution_empty(self):
        df = pd.DataFrame()
        assert plotly_cluster_distribution(df) is None

    def test_salary_by_cluster_empty(self):
        df = pd.DataFrame(
            columns=["salary_max_annual", "cluster_label"]
        )
        assert plotly_salary_by_cluster(df) is None

    def test_regional_breakdown_empty(self):
        df = pd.DataFrame()
        assert plotly_regional_breakdown(df) is None

    def test_us_choropleth_empty(self):
        df = pd.DataFrame()
        assert plotly_us_choropleth(df) is None

    def test_salary_histogram_empty(self):
        df = pd.DataFrame(
            {"salary_max_annual": pd.Series(dtype=float)}
        )
        assert plotly_salary_histogram(df) is None

    def test_yoe_by_category_empty(self):
        df = pd.DataFrame(
            columns=["yoe_min", "cluster_label"]
        )
        assert plotly_yoe_by_category(df) is None


class TestPlotlyChartsMissingColumns:
    """Test graceful handling of missing columns."""

    def test_salary_by_region_no_salary_col(self):
        df = pd.DataFrame({"region": ["NYC", "SF"]})
        assert plotly_salary_by_region(df) is None

    def test_yoe_distribution_no_yoe_col(self):
        df = pd.DataFrame({"title": ["DS"]})
        assert plotly_yoe_distribution(df) is None

    def test_us_choropleth_no_state_col(self):
        df = pd.DataFrame({"title": ["DS"]})
        assert plotly_us_choropleth(df) is None
