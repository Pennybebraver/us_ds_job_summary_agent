"""Tests for visualization modules."""

import os
import pandas as pd
import pytest
from collections import Counter

from src.visualization.heatmaps import (
    create_us_choropleth,
    create_city_heatmap,
    create_region_comparison_chart,
)
from src.visualization.charts import (
    plot_jobs_by_category,
    plot_salary_by_region,
    plot_yoe_distribution,
    plot_top_companies,
    plot_skills_demand,
    plot_cluster_distribution,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "title": ["DS", "MLE", "DS", "RS", "DS", "MLE"],
        "company": ["A", "B", "C", "A", "B", "C"],
        "location": ["NYC", "SF", "Seattle", "NYC", "SF", "Boston"],
        "state": ["NY", "CA", "WA", "NY", "CA", "MA"],
        "region": [
            "Greater NYC", "SF Bay Area", "Seattle Area",
            "Greater NYC", "SF Bay Area", "Boston Area",
        ],
        "cluster_label": [
            "LLM/GenAI", "NLP", "Analytics/A-B Testing",
            "LLM/GenAI", "Computer Vision", "MLOps/ML Engineering",
        ],
        "salary_min_annual": [120000, 130000, 110000, 125000, 135000, 115000],
        "salary_max_annual": [150000, 160000, 140000, 155000, 170000, 145000],
        "yoe_min": [3, 5, 2, 4, 6, 3],
        "latitude": [40.71, 37.77, 47.61, 40.71, 37.77, 42.36],
        "longitude": [-74.01, -122.42, -122.33, -74.01, -122.42, -71.06],
    })


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory."""
    return str(tmp_path / "outputs")


class TestCharts:
    """Tests for analytics charts."""

    def test_plot_jobs_by_category(self, sample_df, output_dir):
        path = plot_jobs_by_category(sample_df, output_dir)
        assert path is not None
        assert os.path.exists(path)

    def test_plot_salary_by_region(self, sample_df, output_dir):
        path = plot_salary_by_region(sample_df, output_dir)
        # May return None if not enough data per region
        if path is not None:
            assert os.path.exists(path)

    def test_plot_yoe_distribution(self, sample_df, output_dir):
        path = plot_yoe_distribution(sample_df, output_dir)
        assert path is not None
        assert os.path.exists(path)

    def test_plot_top_companies(self, sample_df, output_dir):
        path = plot_top_companies(sample_df, output_dir)
        assert path is not None
        assert os.path.exists(path)

    def test_plot_skills_demand(self, output_dir):
        skill_freq = {
            "languages": Counter({"python": 10, "sql": 8, "r": 3}),
            "ml_frameworks": Counter({"pytorch": 5, "tensorflow": 4}),
        }
        path = plot_skills_demand(skill_freq, output_dir)
        assert path is not None
        assert os.path.exists(path)

    def test_plot_cluster_distribution(self, sample_df, output_dir):
        path = plot_cluster_distribution(sample_df, output_dir)
        assert path is not None
        assert os.path.exists(path)

    def test_empty_dataframe(self, output_dir):
        df = pd.DataFrame()
        assert plot_jobs_by_category(df, output_dir) is None

    def test_missing_columns(self, output_dir):
        df = pd.DataFrame({"title": ["DS"]})
        assert plot_jobs_by_category(df, output_dir) is None


class TestHeatmaps:
    """Tests for geographic heatmaps."""

    def test_city_heatmap(self, sample_df, output_dir):
        output_path = os.path.join(output_dir, "test_heatmap.html")
        path = create_city_heatmap(sample_df, output_path=output_path)
        assert path is not None
        assert os.path.exists(path)
        assert path.endswith(".html")

    def test_city_heatmap_no_geo_data(self, output_dir):
        df = pd.DataFrame({
            "latitude": [None],
            "longitude": [None],
        })
        output_path = os.path.join(output_dir, "empty_heatmap.html")
        result = create_city_heatmap(df, output_path=output_path)
        assert result is None

    def test_us_choropleth(self, sample_df, output_dir):
        output_path = os.path.join(output_dir, "test_choropleth.png")
        path = create_us_choropleth(sample_df, output_path=output_path)
        if path is not None:  # Depends on kaleido being available
            assert os.path.exists(path)

    def test_region_comparison(self, sample_df, output_dir):
        output_path = os.path.join(output_dir, "test_region.png")
        path = create_region_comparison_chart(
            sample_df, output_path=output_path
        )
        if path is not None:
            assert os.path.exists(path)
