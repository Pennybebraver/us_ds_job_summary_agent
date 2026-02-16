"""Tests for dashboard filter logic."""

import pandas as pd
import pytest

from src.dashboard.filters import apply_filters


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing filters."""
    return pd.DataFrame({
        "title": [
            "Data Scientist", "ML Engineer", "Applied Scientist",
            "Research Scientist", "Data Science Manager",
        ],
        "company": ["Google", "Meta", "Amazon", "Apple", "Netflix"],
        "region": [
            "SF Bay Area", "Greater NYC", "Seattle Area",
            "SF Bay Area", "Greater NYC",
        ],
        "cluster_label": [
            "LLM/GenAI", "MLOps", "Computer Vision",
            "NLP", "Analytics",
        ],
        "salary_max_annual": [200000, 180000, 220000, 190000, 210000],
        "yoe_min": [3, 5, 7, 2, 10],
    })


class TestApplyFilters:
    """Tests for the apply_filters function."""

    def test_no_filters(self, sample_df):
        """All None filters returns full DataFrame."""
        result = apply_filters(sample_df, None, None, None, None)
        assert len(result) == len(sample_df)

    def test_filter_by_region(self, sample_df):
        """Filter by specific regions."""
        result = apply_filters(
            sample_df,
            regions=["SF Bay Area"],
            categories=None,
            salary_range=None,
            yoe_range=None,
        )
        assert len(result) == 2
        assert all(r == "SF Bay Area" for r in result["region"])

    def test_filter_by_category(self, sample_df):
        """Filter by specific categories."""
        result = apply_filters(
            sample_df,
            regions=None,
            categories=["LLM/GenAI", "NLP"],
            salary_range=None,
            yoe_range=None,
        )
        assert len(result) == 2

    def test_filter_by_salary_range(self, sample_df):
        """Filter by salary range."""
        result = apply_filters(
            sample_df,
            regions=None,
            categories=None,
            salary_range=(190000, 220000),
            yoe_range=None,
        )
        assert len(result) == 4  # 200k, 220k, 190k, 210k

    def test_filter_by_yoe_range(self, sample_df):
        """Filter by YOE range."""
        result = apply_filters(
            sample_df,
            regions=None,
            categories=None,
            salary_range=None,
            yoe_range=(0, 5),
        )
        assert len(result) == 3  # yoe 3, 5, 2

    def test_combined_filters(self, sample_df):
        """Multiple filters applied together."""
        result = apply_filters(
            sample_df,
            regions=["SF Bay Area", "Greater NYC"],
            categories=None,
            salary_range=(180000, 210000),
            yoe_range=(0, 5),
        )
        # SF Bay Area + Greater NYC = 4 rows
        # salary 180k-210k = 200k, 180k, 190k, 210k
        # yoe 0-5 = 3, 5, 2
        # Intersection of all three
        assert len(result) >= 1

    def test_filter_preserves_na_region(self):
        """NaN values in region should be kept when filtering."""
        df = pd.DataFrame({
            "region": ["SF Bay Area", None, "Greater NYC"],
            "cluster_label": ["A", "B", "C"],
            "salary_max_annual": [100000, 200000, 150000],
            "yoe_min": [3, 5, 7],
        })
        result = apply_filters(
            df,
            regions=["SF Bay Area"],
            categories=None,
            salary_range=None,
            yoe_range=None,
        )
        # Should include SF Bay Area row + NaN region row
        assert len(result) == 2

    def test_empty_dataframe(self):
        """Apply filters to empty DataFrame."""
        df = pd.DataFrame(columns=[
            "region", "cluster_label",
            "salary_max_annual", "yoe_min",
        ])
        result = apply_filters(df, None, None, None, None)
        assert len(result) == 0

    def test_missing_columns(self):
        """Apply filters when columns are missing."""
        df = pd.DataFrame({
            "title": ["Data Scientist"],
            "company": ["Google"],
        })
        result = apply_filters(
            df,
            regions=["SF Bay Area"],
            categories=["NLP"],
            salary_range=(100000, 200000),
            yoe_range=(0, 5),
        )
        # No matching columns, so filters are no-ops
        assert len(result) == 1
