"""
PDF report generator using fpdf2.
Creates professional weekly reports with embedded charts and analysis.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from fpdf import FPDF

logger = logging.getLogger(__name__)


class JobReportPDF(FPDF):
    """Custom PDF class for job market reports."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        """Page header."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(128, 128, 128)
        self.cell(
            0, 8,
            "DS/ML Job Market Report",
            align="L",
        )
        self.cell(
            0, 8,
            datetime.now().strftime("%B %d, %Y"),
            align="R",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        """Page footer."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title: str):
        """Add a section title."""
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(33, 37, 41)
        self.ln(5)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 123, 255)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(5)

    def body_text(self, text: str):
        """Add body text."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(33, 37, 41)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def bullet_point(self, text: str):
        """Add a bullet point."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(33, 37, 41)
        indent = 15
        self.set_x(indent)
        self.cell(5, 6, "-")
        self.set_x(indent + 5)
        avail_w = self.w - self.r_margin - indent - 5
        self.multi_cell(avail_w, 6, text)
        self.ln(1)

    def add_chart(self, image_path: str, width: int = 180):
        """Add a chart image to the report."""
        if not os.path.exists(image_path):
            logger.warning(f"Chart not found: {image_path}")
            return

        # Check if we need a new page
        if self.get_y() > 200:
            self.add_page()

        try:
            self.image(image_path, x=15, w=width)
            self.ln(10)
        except Exception as e:
            logger.warning(f"Failed to add chart {image_path}: {e}")

    def add_table(self, headers: List[str], data: List[List[str]], col_widths: Optional[List[int]] = None):
        """Add a simple table."""
        if col_widths is None:
            col_widths = [int(180 / len(headers))] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 123, 255)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            w = col_widths[i] if i < len(col_widths) else col_widths[-1]
            self.cell(w, 8, str(header), border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(33, 37, 41)
        fill = False
        for row in data:
            if self.get_y() > 270:
                self.add_page()
            if fill:
                self.set_fill_color(240, 240, 240)
            else:
                self.set_fill_color(255, 255, 255)

            for i, cell in enumerate(row):
                w = col_widths[i] if i < len(col_widths) else col_widths[-1]
                self.cell(w, 7, str(cell)[:40], border=1, fill=True, align="C")
            self.ln()
            fill = not fill

        self.ln(5)


def generate_report(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    charts: Dict[str, str],
    career_suggestions: List[str],
    output_dir: str = "weekly_reports",
    skill_freq: Optional[Dict] = None,
) -> str:
    """
    Generate a comprehensive PDF report.

    Args:
        df: Processed and clustered job DataFrame.
        cluster_summary: Cluster summary statistics.
        charts: Dict mapping chart name to file path.
        career_suggestions: List of career suggestion strings.
        output_dir: Directory to save the report.
        skill_freq: Optional skill frequency data.

    Returns:
        Path to the generated PDF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, f"report_{date_str}.pdf")

    pdf = JobReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ===== TITLE PAGE =====
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 123, 255)
    pdf.ln(40)
    pdf.cell(0, 15, "Data Science & ML", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 15, "Job Market Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, datetime.now().strftime("%B %d, %Y"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(
        0, 8,
        f"Covering {len(df)} job openings across major job boards",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    # ===== EXECUTIVE SUMMARY =====
    pdf.add_page()
    pdf.chapter_title("1. Executive Summary")

    total_jobs = len(df)
    regions_count = df["region"].nunique() if "region" in df.columns else 0
    companies_count = df["company"].nunique() if "company" in df.columns else 0
    clusters_count = df["cluster_label"].nunique() if "cluster_label" in df.columns else 0

    summary_text = (
        f"This report analyzes {total_jobs} data science and machine learning "
        f"job openings across {regions_count} regions from {companies_count} "
        f"companies. Jobs are categorized into {clusters_count} distinct clusters "
        f"based on NLP analysis of job descriptions."
    )
    pdf.body_text(summary_text)

    # Key metrics
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Key Metrics:", new_x="LMARGIN", new_y="NEXT")

    metrics = [
        f"Total job openings analyzed: {total_jobs}",
        f"Unique companies hiring: {companies_count}",
        f"Geographic regions covered: {regions_count}",
        f"Job categories identified: {clusters_count}",
    ]

    if "salary_max_annual" in df.columns:
        median_sal = df["salary_max_annual"].dropna().median()
        if not pd.isna(median_sal):
            metrics.append(f"Median max salary: ${median_sal:,.0f}")

    if "yoe_min" in df.columns:
        median_yoe = df["yoe_min"].dropna().median()
        if not pd.isna(median_yoe):
            metrics.append(f"Median YOE requirement: {median_yoe:.0f} years")

    for metric in metrics:
        pdf.bullet_point(metric)

    # ===== JOB MARKET OVERVIEW =====
    pdf.add_page()
    pdf.chapter_title("2. Job Market Overview")

    if "jobs_by_category" in charts:
        pdf.body_text("Distribution of job openings across identified categories:")
        pdf.add_chart(charts["jobs_by_category"])

    if "cluster_distribution" in charts:
        pdf.body_text("Proportional breakdown of job categories:")
        pdf.add_chart(charts["cluster_distribution"], width=140)

    # ===== REGIONAL ANALYSIS =====
    pdf.add_page()
    pdf.chapter_title("3. Regional Analysis")

    has_region_data = (
        "region" in df.columns
        and not df.empty
        and "salary_max_annual" in df.columns
        and "yoe_min" in df.columns
    )
    if has_region_data:
        try:
            region_stats = (
                df[df["region"] != "Other"]
                .groupby("region")
                .agg(
                    count=("title", "size"),
                    avg_salary=("salary_max_annual", "mean"),
                    avg_yoe=("yoe_min", "mean"),
                )
                .sort_values("count", ascending=False)
                .head(10)
            )
        except Exception:
            region_stats = pd.DataFrame()

        if not region_stats.empty:
            headers = ["Region", "Jobs", "Avg Salary", "Avg YOE"]
            data = []
            for region, row in region_stats.iterrows():
                sal = f"${row['avg_salary']:,.0f}" if not pd.isna(row["avg_salary"]) else "N/A"
                yoe = f"{row['avg_yoe']:.1f}" if not pd.isna(row["avg_yoe"]) else "N/A"
                data.append([str(region), str(int(row["count"])), sal, yoe])

            pdf.add_table(headers, data, col_widths=[50, 30, 50, 50])

    # Add heatmap charts
    if "us_choropleth" in charts:
        pdf.add_page()
        pdf.body_text("US job distribution by state:")
        pdf.add_chart(charts["us_choropleth"])

    if "region_comparison" in charts:
        pdf.body_text("Regional job count comparison:")
        pdf.add_chart(charts["region_comparison"])

    if "regional_breakdown" in charts:
        pdf.add_page()
        pdf.body_text("Job categories breakdown by region:")
        pdf.add_chart(charts["regional_breakdown"])

    # ===== CATEGORY BREAKDOWN =====
    pdf.add_page()
    pdf.chapter_title("4. Category Breakdown")

    if not cluster_summary.empty:
        headers = list(cluster_summary.columns)
        data = cluster_summary.values.tolist()
        data = [[str(c) for c in row] for row in data]

        # Adjust column widths
        n_cols = len(headers)
        col_widths = [int(180 / n_cols)] * n_cols
        pdf.add_table(headers, data, col_widths)

    # ===== SALARY ANALYSIS =====
    pdf.add_page()
    pdf.chapter_title("5. Salary Analysis")

    if "salary_by_region" in charts:
        pdf.body_text("Salary distribution across regions:")
        pdf.add_chart(charts["salary_by_region"])

    if "salary_by_cluster" in charts:
        pdf.body_text("Salary distribution across job categories:")
        pdf.add_chart(charts["salary_by_cluster"])

    # ===== YOE ANALYSIS =====
    pdf.add_page()
    pdf.chapter_title("6. Experience Requirements")

    if "yoe_distribution" in charts:
        pdf.body_text("Years of experience requirements across all openings:")
        pdf.add_chart(charts["yoe_distribution"])

    # ===== SKILLS & COMPANIES =====
    pdf.add_page()
    pdf.chapter_title("7. Skills & Hiring Trends")

    if "skills_demand" in charts:
        pdf.body_text("Most in-demand skills across all job openings:")
        pdf.add_chart(charts["skills_demand"])

    if "top_companies" in charts:
        pdf.body_text("Top companies by number of openings:")
        pdf.add_chart(charts["top_companies"])

    # ===== CAREER SUGGESTIONS =====
    pdf.add_page()
    pdf.chapter_title("8. Career Enhancement Suggestions")

    pdf.body_text(
        "Based on analysis of current job market trends, here are "
        "actionable recommendations for DS/ML professionals:"
    )

    for suggestion in career_suggestions:
        pdf.bullet_point(suggestion)

    # ===== APPENDIX =====
    pdf.add_page()
    pdf.chapter_title("Appendix: Methodology")
    pdf.body_text(
        "This report was generated using automated job scraping from major "
        "job boards (LinkedIn, Indeed, Glassdoor, ZipRecruiter) followed by "
        "NLP-based clustering using sentence-transformers embeddings and "
        "KMeans clustering. Salary data is normalized to annual USD amounts. "
        "Geographic data is geocoded using a combination of cached coordinates "
        "and the Nominatim geocoding service."
    )

    pdf.body_text(
        f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        f"Recommended run frequency: Weekly (every Sunday at 10 PM). "
        f"Estimated scraping time per run: 10-15 minutes."
    )

    # Save
    pdf.output(output_path)
    logger.info(f"PDF report saved to {output_path}")
    return output_path
