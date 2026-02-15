"""
US DS Job Summary Agent — Main entry point and orchestration.

Usage:
    python -m src.main --full-run          # Run complete pipeline
    python -m src.main --scrape-only       # Only scrape jobs
    python -m src.main --report-only       # Generate report from existing data
    python -m src.main --schedule          # Run on weekly schedule
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import schedule
import time as time_module
import yaml

from src.scraper.job_scraper import scrape_jobs
from src.processor.data_processor import process_jobs
from src.nlp.summarizer import summarize_jobs, get_skill_frequency
from src.nlp.clustering import cluster_jobs, get_cluster_summary
from src.visualization.heatmaps import (
    create_us_choropleth,
    create_city_heatmap,
    create_cluster_heatmap,
    create_international_map,
    create_region_comparison_chart,
)
from src.visualization.charts import create_all_charts
from src.reports.pdf_generator import generate_report
from src.career.advisor import generate_career_suggestions

# Setup logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(config_path: str = "config/settings.yaml"):
    """Configure logging based on settings."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        log_config = config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "agent.log")
    except Exception:
        level = logging.INFO
        log_file = "agent.log"

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def full_run(config_path: str = "config/settings.yaml") -> str:
    """
    Execute the complete pipeline:
    Scrape → Process → Summarize → Cluster → Visualize → Report

    Returns:
        Path to the generated PDF report.
    """
    logger = logging.getLogger(__name__)
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Starting full pipeline run")
    logger.info("=" * 60)

    # Step 1: Scrape jobs
    logger.info("Step 1/7: Scraping jobs...")
    raw_df = scrape_jobs(config_path=config_path)
    logger.info(f"Scraped {len(raw_df)} jobs")

    if raw_df.empty:
        logger.error("No jobs found. Aborting pipeline.")
        return ""

    # Save raw data
    os.makedirs("data", exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    raw_path = f"data/raw_jobs_{date_str}.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info(f"Raw data saved to {raw_path}")

    # Step 2: Process jobs
    logger.info("Step 2/7: Processing jobs...")
    processed_df = process_jobs(raw_df, config_path=config_path)
    logger.info(f"Processed {len(processed_df)} jobs")

    # Step 3: Summarize jobs
    logger.info("Step 3/7: Summarizing job descriptions...")
    processed_df = summarize_jobs(processed_df)

    # Step 4: Cluster jobs
    logger.info("Step 4/7: Clustering job descriptions...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    nlp_config = config.get("nlp", {})

    processed_df = cluster_jobs(
        processed_df,
        min_clusters=nlp_config.get("min_clusters", 5),
        max_clusters=nlp_config.get("max_clusters", 15),
        random_state=nlp_config.get("random_state", 42),
    )

    # Save processed data
    save_cols = [c for c in processed_df.columns if c != "embedding"]
    processed_path = f"data/processed_jobs_{date_str}.csv"
    processed_df[save_cols].to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")

    # Step 5: Generate visualizations
    logger.info("Step 5/7: Generating visualizations...")
    os.makedirs("outputs", exist_ok=True)

    charts = {}

    # Heatmaps
    choropleth_path = create_us_choropleth(processed_df)
    if choropleth_path:
        charts["us_choropleth"] = choropleth_path

    city_heatmap_path = create_city_heatmap(processed_df)
    if city_heatmap_path:
        charts["city_heatmap"] = city_heatmap_path

    intl_map_path = create_international_map(processed_df)
    if intl_map_path:
        charts["international_map"] = intl_map_path

    region_chart_path = create_region_comparison_chart(processed_df)
    if region_chart_path:
        charts["region_comparison"] = region_chart_path

    create_cluster_heatmap(processed_df)

    # Analytics charts
    skill_freq = get_skill_frequency(processed_df)
    analytics_charts = create_all_charts(processed_df, skill_freq)
    charts.update(analytics_charts)

    logger.info(f"Generated {len(charts)} visualizations")

    # Step 6: Generate career suggestions
    logger.info("Step 6/7: Generating career suggestions...")
    career_suggestions = generate_career_suggestions(processed_df, skill_freq)

    # Step 7: Generate PDF report
    logger.info("Step 7/7: Generating PDF report...")
    cluster_summary = get_cluster_summary(processed_df)

    report_config = config.get("report", {})
    output_dir = report_config.get("output_dir", "weekly_reports")

    report_path = generate_report(
        df=processed_df,
        cluster_summary=cluster_summary,
        charts=charts,
        career_suggestions=career_suggestions,
        output_dir=output_dir,
        skill_freq=skill_freq,
    )

    # Summary
    elapsed = datetime.now() - start_time
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed}")
    logger.info(f"Report saved to: {report_path}")
    logger.info(f"Total jobs analyzed: {len(processed_df)}")
    logger.info(f"Categories identified: {processed_df.get('cluster_label', pd.Series()).nunique()}")
    logger.info(f"Charts generated: {len(charts)}")
    logger.info("=" * 60)

    # Print run frequency recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDED RUN FREQUENCY: Weekly (every Sunday at 10 PM)")
    print("Estimated time per run: 10-15 minutes")
    print(f"Actual time this run: {elapsed}")
    print("Cron expression: 0 22 * * 0")
    print("To schedule: python -m src.main --schedule")
    print("=" * 60)

    return report_path


def scrape_only(config_path: str = "config/settings.yaml"):
    """Only run the scraping step and save raw data."""
    logger = logging.getLogger(__name__)
    logger.info("Running scrape-only mode...")

    raw_df = scrape_jobs(config_path=config_path)

    os.makedirs("data", exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = f"data/raw_jobs_{date_str}.csv"
    raw_df.to_csv(path, index=False)

    logger.info(f"Scraped {len(raw_df)} jobs, saved to {path}")
    return path


def report_only(
    data_path: str = None,
    config_path: str = "config/settings.yaml",
):
    """Generate report from existing processed data."""
    logger = logging.getLogger(__name__)
    logger.info("Running report-only mode...")

    if data_path is None:
        # Find most recent processed data
        import glob
        files = sorted(glob.glob("data/processed_jobs_*.csv"), reverse=True)
        if not files:
            logger.error("No processed data found. Run --full-run first.")
            return
        data_path = files[0]

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Regenerate visualizations and report
    os.makedirs("outputs", exist_ok=True)
    skill_freq = get_skill_frequency(df)
    charts = create_all_charts(df, skill_freq)

    choropleth_path = create_us_choropleth(df)
    if choropleth_path:
        charts["us_choropleth"] = choropleth_path

    region_chart_path = create_region_comparison_chart(df)
    if region_chart_path:
        charts["region_comparison"] = region_chart_path

    career_suggestions = generate_career_suggestions(df, skill_freq)
    cluster_summary = get_cluster_summary(df)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    output_dir = config.get("report", {}).get("output_dir", "weekly_reports")

    report_path = generate_report(
        df=df,
        cluster_summary=cluster_summary,
        charts=charts,
        career_suggestions=career_suggestions,
        output_dir=output_dir,
        skill_freq=skill_freq,
    )

    logger.info(f"Report saved to: {report_path}")
    return report_path


def run_scheduled(config_path: str = "config/settings.yaml"):
    """Run the pipeline on a weekly schedule."""
    logger = logging.getLogger(__name__)
    logger.info("Starting scheduled mode - runs every Sunday at 10 PM")

    schedule.every().sunday.at("22:00").do(full_run, config_path=config_path)

    # Also run immediately on start
    logger.info("Running initial pipeline...")
    full_run(config_path=config_path)

    logger.info("Scheduler active. Waiting for next run...")
    while True:
        schedule.run_pending()
        time_module.sleep(60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="US DS Job Summary Agent - Analyze data science job market",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --full-run              # Complete pipeline
  python -m src.main --scrape-only           # Only scrape jobs
  python -m src.main --report-only           # Report from existing data
  python -m src.main --report-only --data data/processed_jobs_2024-01-01.csv
  python -m src.main --schedule              # Weekly schedule

Recommended frequency: Weekly (every Sunday at 10 PM)
Estimated time per run: 10-15 minutes
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--full-run",
        action="store_true",
        help="Run the complete pipeline: scrape, process, analyze, report",
    )
    mode.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only scrape jobs and save raw data",
    )
    mode.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing processed data",
    )
    mode.add_argument(
        "--schedule",
        action="store_true",
        help="Run on a weekly schedule (Sunday 10 PM)",
    )

    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to configuration file (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to existing data file (for --report-only mode)",
    )

    args = parser.parse_args()
    setup_logging(args.config)

    if args.full_run:
        full_run(config_path=args.config)
    elif args.scrape_only:
        scrape_only(config_path=args.config)
    elif args.report_only:
        report_only(data_path=args.data, config_path=args.config)
    elif args.schedule:
        run_scheduled(config_path=args.config)


if __name__ == "__main__":
    main()
