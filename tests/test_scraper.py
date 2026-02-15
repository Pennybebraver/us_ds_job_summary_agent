"""Tests for the job scraper module."""

import pandas as pd
from unittest.mock import patch

from src.scraper.job_scraper import (
    scrape_jobs,
    _is_us_location,
    _standardize_columns,
    _empty_jobs_dataframe,
    load_config,
)


class TestIsUsLocation:
    """Tests for _is_us_location helper."""

    def test_us_cities(self):
        assert _is_us_location("New York, NY") is True
        assert _is_us_location("San Francisco, CA") is True
        assert _is_us_location("Seattle, WA") is True

    def test_international_cities(self):
        assert _is_us_location("Singapore") is False
        assert _is_us_location("Hong Kong") is False

    def test_case_insensitive(self):
        assert _is_us_location("SINGAPORE") is False
        assert _is_us_location("hong kong") is False


class TestStandardizeColumns:
    """Tests for column standardization."""

    def test_standard_columns(self):
        df = pd.DataFrame({
            "title": ["Data Scientist"],
            "company_name": ["Google"],
            "location": ["NYC"],
            "description": ["Build ML models"],
            "min_amount": [100000],
            "max_amount": [150000],
            "job_url": ["http://example.com"],
            "date_posted": ["2024-01-01"],
            "site": ["linkedin"],
        })
        result = _standardize_columns(df)

        assert "title" in result.columns
        assert "company" in result.columns
        assert "url" in result.columns
        assert result.iloc[0]["company"] == "Google"

    def test_missing_columns(self):
        df = pd.DataFrame({"title": ["Data Scientist"]})
        result = _standardize_columns(df)
        assert "title" in result.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = _standardize_columns(df)
        assert isinstance(result, pd.DataFrame)


class TestEmptyDataframe:
    """Tests for empty DataFrame creation."""

    def test_has_expected_columns(self):
        df = _empty_jobs_dataframe()
        expected = [
            "title", "company", "location", "description",
            "salary_min", "salary_max", "url", "date_posted",
            "site", "search_query", "search_location",
        ]
        for col in expected:
            assert col in df.columns
        assert len(df) == 0


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_config(self):
        config = load_config("config/settings.yaml")
        assert "search" in config
        assert "job_titles" in config["search"]
        assert "locations" in config["search"]


class TestScrapeJobs:
    """Tests for the main scrape_jobs function."""

    @patch("jobspy.scrape_jobs")
    def test_scrape_jobs_with_results(self, mock_scrape):
        """Test scraping returns combined DataFrame."""
        mock_df = pd.DataFrame({
            "title": ["Data Scientist"],
            "company_name": ["TestCo"],
            "location": ["New York, NY"],
            "description": ["Build models"],
            "min_amount": [100000],
            "max_amount": [150000],
            "job_url": ["http://test.com"],
            "date_posted": ["2024-01-01"],
            "site": ["indeed"],
        })
        mock_scrape.return_value = mock_df

        result = scrape_jobs(
            job_titles=["Data Scientist"],
            locations=["New York, NY"],
            sites=["indeed"],
            results_per_query=5,
            request_delay=0,
        )

        assert not result.empty
        assert "title" in result.columns
        mock_scrape.assert_called_once()

    @patch("jobspy.scrape_jobs")
    def test_scrape_jobs_empty_results(self, mock_scrape):
        """Test scraping with no results."""
        mock_scrape.return_value = pd.DataFrame()

        result = scrape_jobs(
            job_titles=["Nonexistent Role"],
            locations=["Nowhere"],
            sites=["indeed"],
            results_per_query=5,
            request_delay=0,
        )

        assert result.empty

    @patch("jobspy.scrape_jobs")
    def test_scrape_jobs_handles_error(self, mock_scrape):
        """Test scraping handles exceptions gracefully."""
        mock_scrape.side_effect = Exception("Rate limited")

        result = scrape_jobs(
            job_titles=["Data Scientist"],
            locations=["New York, NY"],
            sites=["indeed"],
            results_per_query=5,
            request_delay=0,
        )

        assert result.empty
