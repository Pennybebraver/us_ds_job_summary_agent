"""Tests for the data processor module."""

import pandas as pd

from src.processor.data_processor import (
    deduplicate,
    clean_text_fields,
    geocode_locations,
    _parse_yoe,
    _normalize_salary,
    _match_region,
    _match_state,
)


class TestDeduplicate:
    """Tests for job deduplication."""

    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({
            "title": ["DS", "DS", "MLE"],
            "company": ["A", "A", "B"],
            "location": ["NYC", "NYC", "SF"],
            "url": ["u1", "u1", "u2"],
        })
        result = deduplicate(df)
        assert len(result) == 2

    def test_keeps_unique_jobs(self):
        df = pd.DataFrame({
            "title": ["DS", "MLE", "RS"],
            "company": ["A", "B", "C"],
            "location": ["NYC", "SF", "Boston"],
            "url": ["u1", "u2", "u3"],
        })
        result = deduplicate(df)
        assert len(result) == 3

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["title", "company", "location"])
        result = deduplicate(df)
        assert len(result) == 0


class TestCleanTextFields:
    """Tests for text field cleaning."""

    def test_strips_whitespace(self):
        df = pd.DataFrame({"title": ["  Data Scientist  "]})
        result = clean_text_fields(df)
        assert result.iloc[0]["title"] == "Data Scientist"

    def test_removes_html(self):
        df = pd.DataFrame({
            "description": ["<p>Build <b>ML</b> models</p>"]
        })
        result = clean_text_fields(df)
        assert "<" not in result.iloc[0]["description"]

    def test_fills_na_company(self):
        df = pd.DataFrame({"company": [None]})
        result = clean_text_fields(df)
        assert result.iloc[0]["company"] == "Unknown"


class TestGeocodeLocations:
    """Tests for geocoding."""

    def test_known_city(self):
        df = pd.DataFrame({"location": ["New York, NY"]})
        result = geocode_locations(df)
        assert result.iloc[0]["latitude"] is not None
        assert abs(result.iloc[0]["latitude"] - 40.7128) < 1

    def test_singapore(self):
        df = pd.DataFrame({"location": ["Singapore"]})
        result = geocode_locations(df)
        assert result.iloc[0]["latitude"] is not None
        assert abs(result.iloc[0]["latitude"] - 1.35) < 1

    def test_unknown_location(self):
        df = pd.DataFrame({"location": ["ZZZZZ Unknown Place"]})
        result = geocode_locations(df)
        # Should still return a DataFrame (lat/lon may be None)
        assert "latitude" in result.columns


class TestAssignRegions:
    """Tests for region assignment."""

    def test_match_region(self):
        config = {
            "Greater NYC": ["New York", "Manhattan"],
            "SF Bay Area": ["San Francisco", "San Jose"],
        }
        assert _match_region("New York, NY", config) == "Greater NYC"
        assert _match_region("San Francisco, CA", config) == "SF Bay Area"
        assert _match_region("Random Place", config) == "Other"

    def test_match_state(self):
        assert _match_state("New York, NY") == "NY"
        assert _match_state("San Francisco, CA") == "CA"
        assert _match_state("Seattle, WA") == "WA"

    def test_match_state_from_city(self):
        assert _match_state("new york") == "NY"
        assert _match_state("san francisco") == "CA"


class TestExtractYoe:
    """Tests for years of experience extraction."""

    def test_range_format(self):
        assert _parse_yoe("3-5 years of experience") == (3, 5)
        assert _parse_yoe("2 to 4 years experience") == (2, 4)

    def test_plus_format(self):
        assert _parse_yoe("5+ years of experience") == (5, None)

    def test_minimum_format(self):
        assert _parse_yoe("minimum 3 years") == (3, None)
        assert _parse_yoe("at least 2 years") == (2, None)

    def test_no_yoe(self):
        assert _parse_yoe("Great company culture") == (None, None)

    def test_complex_text(self):
        text = "We need someone with 3-5 years of experience in ML"
        assert _parse_yoe(text) == (3, 5)


class TestNormalizeSalaries:
    """Tests for salary normalization."""

    def test_annual_salary(self):
        assert _normalize_salary(150000) == 150000

    def test_hourly_rate(self):
        result = _normalize_salary(75)
        assert result == 75 * 2080  # hourly to annual

    def test_monthly_salary(self):
        result = _normalize_salary(8000)
        assert result == 8000 * 12

    def test_none_value(self):
        assert _normalize_salary(None) is None

    def test_nan_value(self):
        assert _normalize_salary(float("nan")) is None

    def test_string_value(self):
        result = _normalize_salary("$150,000")
        assert result == 150000
