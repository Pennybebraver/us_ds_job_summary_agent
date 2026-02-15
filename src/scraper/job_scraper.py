"""
Job scraper module using python-jobspy to search across multiple job boards.
Searches for DS/MLE/Applied Scientist/Research Scientist positions.
"""

import logging
import time
from typing import List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def scrape_jobs(
    job_titles: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    sites: Optional[List[str]] = None,
    results_per_query: int = 50,
    request_delay: int = 3,
    config_path: str = "config/settings.yaml",
) -> pd.DataFrame:
    """
    Scrape jobs from multiple job boards using python-jobspy.

    Args:
        job_titles: List of job titles to search for.
        locations: List of locations to search in.
        sites: List of job board sites to search.
        results_per_query: Maximum results per query.
        request_delay: Delay between requests in seconds.
        config_path: Path to configuration file.

    Returns:
        DataFrame with columns: title, company, location, description,
        salary_min, salary_max, url, date_posted, site, search_query
    """
    from jobspy import scrape_jobs as jobspy_scrape

    config = load_config(config_path)
    search_config = config.get("search", {})

    if job_titles is None:
        job_titles = search_config.get("job_titles", ["Data Scientist"])
    if locations is None:
        us_locations = search_config.get("locations", {}).get("us", [])
        intl_locations = search_config.get("locations", {}).get("international", [])
        locations = us_locations + intl_locations
    if sites is None:
        sites = search_config.get("sites", ["indeed"])

    all_jobs = []
    total_queries = len(job_titles) * len(locations)
    query_count = 0

    for title in job_titles:
        for location in locations:
            query_count += 1
            logger.info(
                f"Scraping [{query_count}/{total_queries}]: "
                f"'{title}' in '{location}'"
            )

            try:
                is_us = _is_us_location(location)
                scrape_kwargs = {
                    "site_name": sites,
                    "search_term": title,
                    "location": location,
                    "results_wanted": results_per_query,
                    "hours_old": 168,  # Last 7 days
                }
                if is_us:
                    scrape_kwargs["country_indeed"] = "USA"
                else:
                    # Map international locations to country codes
                    country = _get_country_code(location)
                    if country:
                        scrape_kwargs["country_indeed"] = country

                jobs = jobspy_scrape(**scrape_kwargs)

                if jobs is not None and not jobs.empty:
                    jobs["search_query"] = title
                    jobs["search_location"] = location
                    all_jobs.append(jobs)
                    logger.info(f"  Found {len(jobs)} jobs")
                else:
                    logger.info("  No jobs found")

            except Exception as e:
                logger.warning(
                    f"  Error scraping '{title}' in '{location}': {e}"
                )

            # Rate limiting
            if query_count < total_queries:
                time.sleep(request_delay)

    if not all_jobs:
        logger.warning("No jobs found across all queries")
        return _empty_jobs_dataframe()

    combined = pd.concat(all_jobs, ignore_index=True)
    combined = _standardize_columns(combined)
    logger.info(f"Total jobs scraped: {len(combined)}")

    return combined


def _is_us_location(location: str) -> bool:
    """Check if a location is in the US."""
    international = ["singapore", "hong kong", "london", "tokyo"]
    return not any(intl in location.lower() for intl in international)


def _get_country_code(location: str) -> Optional[str]:
    """Map international location to a country string for jobspy."""
    location_lower = location.lower()
    country_map = {
        "singapore": "Singapore",
        "hong kong": "Hong Kong",
        "london": "UK",
        "tokyo": "Japan",
    }
    for key, value in country_map.items():
        if key in location_lower:
            return value
    return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame columns to expected format."""
    column_mapping = {
        "title": "title",
        "company_name": "company",
        "company": "company",
        "location": "location",
        "description": "description",
        "min_amount": "salary_min",
        "max_amount": "salary_max",
        "job_url": "url",
        "date_posted": "date_posted",
        "site": "site",
        "search_query": "search_query",
        "search_location": "search_location",
    }

    standardized = pd.DataFrame()
    for source_col, target_col in column_mapping.items():
        if source_col in df.columns:
            standardized[target_col] = df[source_col]
        elif target_col not in standardized.columns:
            standardized[target_col] = None

    return standardized


def _empty_jobs_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with expected columns."""
    columns = [
        "title", "company", "location", "description",
        "salary_min", "salary_max", "url", "date_posted",
        "site", "search_query", "search_location",
    ]
    return pd.DataFrame(columns=columns)
