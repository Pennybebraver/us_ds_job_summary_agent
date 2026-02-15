# US Data Science Job Summary Agent

An automated agent that scrapes, analyzes, and reports on Data Science, Machine Learning Engineer, Applied Scientist, and Research Scientist job openings across the US, Singapore, and Hong Kong.

## Features

- **Job scraping**: LinkedIn via `python-jobspy` (additional boards like Indeed, Glassdoor, ZipRecruiter can be enabled in config)
- **NLP-powered clustering**: Automatically categorizes jobs (LLM/GenAI, Analytics, Computer Vision, Fraud Detection, etc.) using sentence-transformers
- **Geographic heatmaps**: US choropleth by state, city-level density maps, international maps
- **Salary & YOE analysis**: Regional salary distributions, experience requirement breakdowns
- **Professional PDF reports**: Weekly reports with charts, tables, and career suggestions
- **Career advisor**: Trending skills, salary benchmarks, and actionable career recommendations
- **Logging**: Full pipeline logging to both console and `agent.log` file

## Requirements

- **Python**: 3.10+ (tested with 3.12.8)
- **OS**: macOS, Linux, or Windows
- **Network**: Internet connection for job scraping and first-run model download

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Pennybebraver/us_ds_job_summary_agent.git
cd us_ds_job_summary_agent

# Create virtual environment (requires Python 3.10+)
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m src.main --full-run
```

## CLI Usage

### Commands

```bash
# Full pipeline: scrape → process → cluster → visualize → PDF report
python -m src.main --full-run

# Only scrape jobs and save raw CSV to data/
python -m src.main --scrape-only

# Generate report from most recent processed data
python -m src.main --report-only

# Generate report from a specific data file
python -m src.main --report-only --data data/processed_jobs_2025-02-15.csv

# Run on automatic weekly schedule (Sunday 10 PM, runs immediately first)
python -m src.main --schedule

# Use a custom config file
python -m src.main --full-run --config config/custom_settings.yaml
```

### CLI Options Reference

| Option | Description |
|---|---|
| `--full-run` | Run the complete pipeline end-to-end |
| `--scrape-only` | Only scrape jobs, save raw data to `data/` |
| `--report-only` | Generate PDF from existing processed data |
| `--schedule` | Run immediately, then repeat weekly on a schedule |
| `--config PATH` | Path to YAML config file (default: `config/settings.yaml`) |
| `--data PATH` | Path to CSV data file (for `--report-only` mode) |

### What Each Mode Does

**`--full-run`** (most common):
1. Scrapes jobs from all configured boards and locations
2. Cleans, deduplicates, geocodes, and normalizes data
3. Extracts skills and summarizes job descriptions
4. Clusters jobs into categories using NLP embeddings
5. Generates geographic heatmaps and analytics charts
6. Produces a PDF report in `weekly_reports/`
7. Saves raw and processed CSVs in `data/`

**`--scrape-only`**:
- Runs only the scraping step
- Saves raw data to `data/raw_jobs_YYYY-MM-DD.csv`
- Useful for collecting data without processing

**`--report-only`**:
- Skips scraping entirely
- Loads the most recent `data/processed_jobs_*.csv` (or a file you specify with `--data`)
- Regenerates charts and PDF report
- Useful for re-running analysis on existing data

**`--schedule`**:
- Runs `--full-run` immediately
- Then waits and re-runs every Sunday at 10 PM
- Must keep terminal open (or run via cron/launchd instead)

## Estimated Running Time

Benchmarked on Apple M4, 16GB RAM, ~300 Mbps download:

| Pipeline Step | Time | Bottleneck |
|---|---|---|
| Scraping | 3.5–6 min | Network + rate-limiting delays (3s between queries) |
| Processing | 3–5 sec | CPU (pandas, regex, geocoding) |
| Summarization | 3–5 sec | CPU (string processing) |
| Clustering | 20–40 sec | CPU (embedding + KMeans). First run adds ~30s for model download |
| Visualization | 8–15 sec | Chart rendering + image export |
| PDF Generation | 1–2 sec | FPDF2 is lightweight |
| **Total (first run)** | **~5–8 min** | Includes one-time model download (~80MB) |
| **Total (subsequent)** | **~4.5–7 min** | Model cached locally |

> ~90% of run time is scraping wait time (the `request_delay` between queries). The compute steps take ~40–60 seconds total.

## Scheduling (Run Frequency)

**Recommended: Weekly** — job boards refresh listings every few days, and weekly runs balance fresh data against rate limits.

### Option 1: Built-in scheduler (simplest, requires terminal open)

```bash
source venv/bin/activate
python -m src.main --schedule
# Runs immediately, then every Sunday at 10 PM
```

### Option 2: macOS cron (runs in background)

```bash
crontab -e
# Add this line (Sunday 10 PM):
0 22 * * 0 cd /Users/penny/Desktop/us_ds_job_summary_agent && /Users/penny/Desktop/us_ds_job_summary_agent/venv/bin/python -m src.main --full-run >> agent.log 2>&1
```

Common cron expressions:
| Schedule | Cron Expression |
|---|---|
| Every Sunday 10 PM | `0 22 * * 0` |
| Every Wednesday 8 AM | `0 8 * * 3` |
| Twice weekly (Mon & Thu 9 AM) | `0 9 * * 1,4` |
| Daily at midnight | `0 0 * * *` |

### Option 3: macOS LaunchAgent (survives sleep/wake/reboot)

Create `~/Library/LaunchAgents/com.ds-job-agent.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ds-job-agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/penny/Desktop/us_ds_job_summary_agent/venv/bin/python</string>
        <string>-m</string>
        <string>src.main</string>
        <string>--full-run</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/penny/Desktop/us_ds_job_summary_agent</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>22</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/penny/Desktop/us_ds_job_summary_agent/agent.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/penny/Desktop/us_ds_job_summary_agent/agent.log</string>
</dict>
</plist>
```

Then load it:
```bash
launchctl load ~/Library/LaunchAgents/com.ds-job-agent.plist

# To unload:
launchctl unload ~/Library/LaunchAgents/com.ds-job-agent.plist
```

## Configuration

All settings are in `config/settings.yaml`.

### Search Settings

```yaml
search:
  job_titles:             # Roles to search for
    - "Data Scientist"
    - "Machine Learning Engineer"
    - "Applied Scientist"
    - "Research Scientist"
    - "ML Engineer"
    - "Data Science Manager"

  locations:
    us:                   # US cities to search
      - "New York, NY"
      - "San Francisco, CA"
      - "Seattle, WA"
      - "Boston, MA"
      # ... add/remove cities as needed
    international:        # International locations
      - "Singapore"
      - "Hong Kong"

  sites:                  # Job boards to scrape (default: LinkedIn only)
    - "linkedin"
    # - "indeed"           # Can be enabled, but may hit rate limits
    # - "glassdoor"        # Often returns 400 errors for location parsing
    # - "zip_recruiter"    # Can be enabled if needed

  results_per_query: 50   # Max results per title+location query
  request_delay: 3        # Seconds between requests (rate limiting)
```

### Tuning `request_delay`

| Delay | Total Run Time | Rate Limit Risk |
|---|---|---|
| 1 sec | ~3–4 min | Higher — may get blocked |
| 3 sec (default) | ~5–8 min | Low — safe for regular use |
| 5 sec | ~9–12 min | Very low — conservative |

### NLP Settings

```yaml
nlp:
  model_name: "all-MiniLM-L6-v2"   # Sentence-transformers model
  min_clusters: 5                    # Minimum clusters to try
  max_clusters: 15                   # Maximum clusters to try
  random_state: 42                   # Reproducibility seed
```

### Logging Settings

```yaml
logging:
  level: "INFO"        # DEBUG, INFO, WARNING, ERROR
  file: "agent.log"    # Log file path
```

Logs are written to both the console and the log file. Set `level: "DEBUG"` for verbose output during troubleshooting.

## Output Files

After a `--full-run`, you'll find:

```
data/
  raw_jobs_2025-02-15.csv           # Raw scraped data
  processed_jobs_2025-02-15.csv     # Cleaned + enriched data

outputs/
  us_choropleth.png                 # US state-level job density map
  city_heatmap.html                 # Interactive city heatmap (open in browser)
  international_heatmap.html        # Singapore/Hong Kong map
  region_comparison.png             # Regional job count bar chart
  jobs_by_category.png              # Category distribution
  salary_by_region.png              # Salary box plots by region
  salary_by_cluster.png             # Salary box plots by category
  yoe_distribution.png              # Years of experience histogram
  top_companies.png                 # Top hiring companies
  skills_demand.png                 # Most in-demand skills
  cluster_distribution.png          # Category pie chart
  regional_breakdown.png            # Stacked bar: categories per region
  heatmap_*.html                    # Per-cluster heatmaps

weekly_reports/
  report_2025-02-15.pdf             # Full PDF report

agent.log                           # Pipeline execution log
```

## Project Structure

```
us_ds_job_summary_agent/
├── config/settings.yaml          # Search parameters, locations, NLP settings
├── src/
│   ├── scraper/job_scraper.py    # Multi-board job scraping via python-jobspy
│   ├── processor/data_processor.py # Data cleaning, geocoding, normalization
│   ├── nlp/
│   │   ├── summarizer.py         # Job description summarization & skill extraction
│   │   └── clustering.py         # Embedding-based job clustering (KMeans)
│   ├── visualization/
│   │   ├── heatmaps.py           # Geographic heatmaps (Plotly + Folium)
│   │   └── charts.py             # Analytics charts (Matplotlib + Seaborn)
│   ├── reports/pdf_generator.py  # PDF report generation (FPDF2)
│   ├── career/advisor.py         # Career trend analysis & suggestions
│   └── main.py                   # CLI entry point & pipeline orchestration
├── tests/                        # Unit and integration tests (76 tests)
├── data/                         # Raw and processed job data (CSV)
├── outputs/                      # Generated charts and heatmaps
├── weekly_reports/               # Generated PDF reports
└── requirements.txt              # Python dependencies
```

## Pipeline

```
Settings (YAML)
    → Scraper (python-jobspy: LinkedIn by default; other boards configurable)
    → Processor (clean, deduplicate, geocode, parse salary/YOE, assign regions)
    → NLP Summarizer (extractive summaries, skill extraction across 5 categories)
    → NLP Clustering (sentence-transformers embeddings → KMeans → category labels)
    → Visualization (heatmaps, 8 chart types, regional breakdowns)
    → PDF Report (8-section professional report with embedded charts)
    → Career Advisor (skill trends, salary benchmarks, career recommendations)
```

## Testing

```bash
# Run all 76 tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_scraper.py -v
python -m pytest tests/test_nlp.py -v
python -m pytest tests/test_integration.py -v

# Run linter
flake8 src/ tests/
```

## Report Contents

Each weekly PDF report includes:
1. **Executive Summary** — Total jobs, companies, regions, median salary, median YOE
2. **Job Market Overview** — Category distribution bar chart and pie chart
3. **Regional Analysis** — Region stats table, US choropleth, regional comparison chart
4. **Category Breakdown** — Cluster summary table with counts, salaries, YOE, top regions
5. **Salary Analysis** — Box plots by region and by job category
6. **Experience Requirements** — YOE histogram with median line
7. **Skills & Hiring Trends** — Top 20 skills bar chart, top 15 companies
8. **Career Suggestions** — Trending skills, salary benchmarks, geographic insights, general advice
9. **Appendix** — Methodology description and run metadata

## Troubleshooting

**Rate limited (429 errors)**:
- Increase `request_delay` in `config/settings.yaml` to 5 or higher
- Reduce the number of locations or job titles
- Wait 15–30 minutes before retrying

**No jobs found**:
- Check your internet connection
- LinkedIn may block scraping from certain IPs — try increasing `request_delay`
- If adding other boards, note that Glassdoor often returns 400 errors for location parsing

**Model download fails**:
- The `all-MiniLM-L6-v2` model (~80MB) downloads on first run
- Ensure stable internet; it caches at `~/.cache/huggingface/`
- If it fails, the agent falls back to TF-IDF embeddings automatically

**Empty PDF report**:
- Check `agent.log` for error details
- Verify `data/processed_jobs_*.csv` has data
- Re-run with `--report-only` after confirming data exists
