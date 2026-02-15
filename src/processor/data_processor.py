"""
Data processor module for cleaning, normalizing, geocoding, and enriching job data.
"""

import logging
import re
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Pre-defined geocoding cache to avoid excessive API calls
CITY_COORDINATES = {
    "new york": (40.7128, -74.0060),
    "manhattan": (40.7831, -73.9712),
    "brooklyn": (40.6782, -73.9442),
    "san francisco": (37.7749, -122.4194),
    "san jose": (37.3382, -121.8863),
    "palo alto": (37.4419, -122.1430),
    "mountain view": (37.3861, -122.0839),
    "sunnyvale": (37.3688, -122.0363),
    "menlo park": (37.4530, -122.1817),
    "seattle": (47.6062, -122.3321),
    "bellevue": (47.6101, -122.2015),
    "redmond": (47.6740, -122.1215),
    "boston": (42.3601, -71.0589),
    "cambridge": (42.3736, -71.1097),
    "los angeles": (34.0522, -118.2437),
    "santa monica": (34.0195, -118.4912),
    "chicago": (41.8781, -87.6298),
    "austin": (30.2672, -97.7431),
    "washington": (38.9072, -77.0369),
    "denver": (39.7392, -104.9903),
    "singapore": (1.3521, 103.8198),
    "hong kong": (22.3193, 114.1694),
    "jersey city": (40.7178, -74.0431),
    "newark": (40.7357, -74.1724),
    "hoboken": (40.7440, -74.0324),
    "oakland": (37.8044, -122.2712),
    "redwood city": (37.4852, -122.2364),
    "kirkland": (47.6769, -122.2060),
    "somerville": (42.3876, -71.0995),
    "waltham": (42.3765, -71.2356),
    "pasadena": (34.1478, -118.1445),
    "burbank": (34.1808, -118.3090),
}

# US state mapping for choropleth
CITY_TO_STATE = {
    "new york": "NY", "manhattan": "NY", "brooklyn": "NY",
    "jersey city": "NJ", "newark": "NJ", "hoboken": "NJ",
    "san francisco": "CA", "san jose": "CA", "palo alto": "CA",
    "mountain view": "CA", "sunnyvale": "CA", "menlo park": "CA",
    "oakland": "CA", "redwood city": "CA", "los angeles": "CA",
    "santa monica": "CA", "pasadena": "CA", "burbank": "CA",
    "seattle": "WA", "bellevue": "WA", "redmond": "WA",
    "kirkland": "WA",
    "boston": "MA", "cambridge": "MA", "somerville": "MA",
    "waltham": "MA",
    "chicago": "IL",
    "austin": "TX",
    "washington": "DC",
    "denver": "CO",
}


def process_jobs(
    df: pd.DataFrame,
    config_path: str = "config/settings.yaml"
) -> pd.DataFrame:
    """
    Clean, normalize, and enrich job data.

    Args:
        df: Raw job DataFrame from scraper.
        config_path: Path to configuration file.

    Returns:
        Processed DataFrame with additional columns:
        latitude, longitude, state, region, yoe_min, yoe_max,
        salary_min_annual, salary_max_annual
    """
    if df.empty:
        logger.warning("Empty DataFrame received, returning as-is")
        return df

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Processing {len(df)} jobs...")

    # Step 1: Remove duplicates
    df = deduplicate(df)
    logger.info(f"After dedup: {len(df)} jobs")

    # Step 2: Clean text fields
    df = clean_text_fields(df)

    # Step 3: Geocode locations
    df = geocode_locations(df)

    # Step 4: Assign states and regions
    df = assign_regions(df, config.get("regions", {}))

    # Step 5: Extract years of experience
    df = extract_yoe(df)

    # Step 6: Normalize salaries
    df = normalize_salaries(df)

    logger.info(f"Processing complete: {len(df)} jobs")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate job listings based on title, company, and location."""
    before = len(df)

    # Drop exact duplicates on key fields
    subset_cols = ["title", "company", "location"]
    existing = [c for c in subset_cols if c in df.columns]
    if existing:
        df = df.drop_duplicates(subset=existing, keep="first")

    # Also drop by URL if available
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")

    after = len(df)
    if before != after:
        logger.info(f"Removed {before - after} duplicate jobs")

    return df.reset_index(drop=True)


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize text fields."""
    if "title" in df.columns:
        df["title"] = df["title"].fillna("").str.strip()

    if "company" in df.columns:
        df["company"] = df["company"].fillna("Unknown").str.strip()

    if "location" in df.columns:
        df["location"] = df["location"].fillna("").str.strip()

    if "description" in df.columns:
        df["description"] = (
            df["description"]
            .fillna("")
            .str.replace(r"<[^>]+>", " ", regex=True)  # Remove HTML
            .str.replace(r"\s+", " ", regex=True)  # Collapse whitespace
            .str.strip()
        )

    return df


def geocode_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add latitude/longitude to jobs using cached coordinates.
    Falls back to geopy for unknown locations.
    """
    latitudes = []
    longitudes = []

    for _, row in df.iterrows():
        location = str(row.get("location", "")).lower()
        lat, lon = _lookup_coordinates(location)
        latitudes.append(lat)
        longitudes.append(lon)

    df["latitude"] = latitudes
    df["longitude"] = longitudes

    geocoded = df["latitude"].notna().sum()
    logger.info(f"Geocoded {geocoded}/{len(df)} locations")

    return df


def _lookup_coordinates(location: str) -> Tuple[Optional[float], Optional[float]]:
    """Look up coordinates from cache, fallback to geopy."""
    location_lower = location.lower()

    # Try cache first
    for city, coords in CITY_COORDINATES.items():
        if city in location_lower:
            return coords

    # Fallback to geopy
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="ds_job_agent")
        geo_result = geolocator.geocode(location, timeout=5)
        if geo_result:
            return (geo_result.latitude, geo_result.longitude)
    except Exception as e:
        logger.debug(f"Geocoding failed for '{location}': {e}")

    return (None, None)


def assign_regions(df: pd.DataFrame, region_config: Dict) -> pd.DataFrame:
    """Assign region labels and US states to jobs."""
    regions = []
    states = []

    for _, row in df.iterrows():
        location = str(row.get("location", "")).lower()
        region = _match_region(location, region_config)
        state = _match_state(location)
        regions.append(region)
        states.append(state)

    df["region"] = regions
    df["state"] = states

    return df


def _match_region(location: str, region_config: Dict) -> str:
    """Match a location string to a region."""
    location_lower = location.lower()
    for region_name, cities in region_config.items():
        for city in cities:
            if city.lower() in location_lower:
                return region_name
    return "Other"


def _match_state(location: str) -> Optional[str]:
    """Match a location string to a US state code."""
    location_lower = location.lower()

    # Try direct city match
    for city, state in CITY_TO_STATE.items():
        if city in location_lower:
            return state

    # Try state abbreviation in location (e.g., "Austin, TX")
    state_match = re.search(r",\s*([A-Z]{2})\b", location)
    if state_match:
        return state_match.group(1)

    return None


def extract_yoe(df: pd.DataFrame) -> pd.DataFrame:
    """Extract years of experience requirements from job descriptions."""
    yoe_min_list = []
    yoe_max_list = []

    for _, row in df.iterrows():
        desc = str(row.get("description", ""))
        yoe_min, yoe_max = _parse_yoe(desc)
        yoe_min_list.append(yoe_min)
        yoe_max_list.append(yoe_max)

    df["yoe_min"] = yoe_min_list
    df["yoe_max"] = yoe_max_list

    return df


def _parse_yoe(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse years of experience from text."""
    patterns = [
        # "3-5 years of experience"
        r"(\d+)\s*[-–to]+\s*(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience|exp)",
        # "3+ years of experience"
        r"(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience|exp)",
        # "minimum 3 years"
        r"(?:minimum|min|at\s+least)\s+(\d+)\s*years?",
        # "3 years experience"
        r"(\d+)\s*years?\s*(?:of\s+)?(?:experience|exp|relevant)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                return int(groups[0]), int(groups[1])
            elif len(groups) == 1:
                yoe = int(groups[0])
                return yoe, None

    return (None, None)


def normalize_salaries(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize salary fields to annual USD amounts."""
    df["salary_min_annual"] = df.get("salary_min", pd.Series(dtype=float)).apply(
        _normalize_salary
    )
    df["salary_max_annual"] = df.get("salary_max", pd.Series(dtype=float)).apply(
        _normalize_salary
    )
    return df


def _normalize_salary(value) -> Optional[float]:
    """Normalize a salary value to annual amount."""
    if pd.isna(value) or value is None:
        return None

    try:
        amount = float(value)
    except (ValueError, TypeError):
        # Try to extract number from string
        text = str(value).replace(",", "").replace("$", "")
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            amount = float(match.group(1))
        else:
            return None

    # Heuristic: if < 1000, likely hourly rate
    if amount < 1000:
        return amount * 2080  # 40 hrs/week * 52 weeks
    # If < 10000, likely monthly
    elif amount < 10000:
        return amount * 12
    # Otherwise assume annual
    return amount
