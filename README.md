# US Data Science Job Summary Agent

An automated agent that scrapes, analyzes, and reports on Data Science, Machine Learning Engineer, Applied Scientist, and Research Scientist job openings across the US, Singapore, and Hong Kong.

## Features

- **Multi-board job scraping**: LinkedIn, Indeed, Glassdoor, ZipRecruiter via `python-jobspy`
- **NLP-powered clustering**: Automatically categorizes jobs (LLM/GenAI, Analytics, Computer Vision, Fraud Detection, etc.) using sentence-transformers
- **Geographic heatmaps**: US choropleth by state, city-level density maps, international maps
- **Salary & YOE analysis**: Regional salary distributions, experience requirement breakdowns
- **Professional PDF reports**: Weekly reports with charts, tables, and career suggestions
- **Career advisor**: Trending skills, salary benchmarks, and actionable career recommendations

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Pennybebraver/us_ds_job_summary_agent.git
cd us_ds_job_summary_agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m src.main --full-run
```

## Usage

```bash
# Complete pipeline: scrape → process → analyze → report
python -m src.main --full-run

# Only scrape jobs (saves to data/)
python -m src.main --scrape-only

# Generate report from existing data
python -m src.main --report-only
python -m src.main --report-only --data data/processed_jobs_2024-01-01.csv

# Run on weekly schedule (every Sunday at 10 PM)
python -m src.main --schedule

# Custom config
python -m src.main --full-run --config config/custom_settings.yaml
```

## Project Structure

```
us_ds_job_summary_agent/
├── config/settings.yaml          # Search parameters, locations, NLP settings
├── src/
│   ├── scraper/job_scraper.py    # Multi-board job scraping
│   ├── processor/data_processor.py # Data cleaning, geocoding, normalization
│   ├── nlp/
│   │   ├── summarizer.py         # Job description summarization & skill extraction
│   │   └── clustering.py         # Embedding-based job clustering
│   ├── visualization/
│   │   ├── heatmaps.py           # Geographic heatmaps (Plotly + Folium)
│   │   └── charts.py             # Analytics charts (Matplotlib + Seaborn)
│   ├── reports/pdf_generator.py  # PDF report generation (FPDF2)
│   ├── career/advisor.py         # Career trend analysis & suggestions
│   └── main.py                   # CLI entry point & pipeline orchestration
├── tests/                        # Unit and integration tests
├── data/                         # Raw and processed job data (CSV)
├── outputs/                      # Generated charts and heatmaps
├── weekly_reports/               # Generated PDF reports
└── requirements.txt              # Python dependencies
```

## Configuration

Edit `config/settings.yaml` to customize:
- **Job titles**: Which roles to search for
- **Locations**: US cities and international locations
- **Job boards**: Which sites to scrape
- **NLP settings**: Clustering model and parameters
- **Report settings**: Output directory and schedule

## Run Frequency

**Recommended: Weekly (every Sunday at 10 PM)**

- Job boards refresh listings every few days
- Weekly cadence captures new postings without hitting rate limits
- Estimated time per run: 10-15 minutes
- Cron expression: `0 22 * * 0`

## Pipeline

```
Settings (YAML)
    → Scraper (python-jobspy: LinkedIn, Indeed, Glassdoor, ZipRecruiter)
    → Processor (clean, geocode, parse salary/YOE, assign regions)
    → NLP Summarizer (skill extraction, extractive summaries)
    → NLP Clustering (sentence-transformers → KMeans → category labels)
    → Visualization (heatmaps, charts, regional breakdowns)
    → PDF Report (professional weekly report)
    → Career Advisor (skill trends, salary benchmarks, recommendations)
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run linter
flake8 src/ tests/
```

## Report Contents

Each weekly PDF report includes:
1. **Executive Summary** - Key metrics and overview
2. **Job Market Overview** - Category distribution charts
3. **Regional Analysis** - Geographic breakdown with heatmaps
4. **Category Breakdown** - Cluster statistics table
5. **Salary Analysis** - Regional and category salary distributions
6. **Experience Requirements** - YOE distribution analysis
7. **Skills & Hiring Trends** - Top skills and companies
8. **Career Suggestions** - Actionable recommendations
