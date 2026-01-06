#!/usr/bin/env python3
"""
Coverage Analyzer CLI

Command-line interface for the Verification Coverage Analyzer.
Provides easy access to all analyzer features.

Usage:
    python main.py analyze coverage_report.txt
    python main.py analyze coverage_report.txt --output results.json
    python main.py parse coverage_report.txt
    python main.py predict coverage_report.txt --deadline 24

Author: ML Engineer
Date: 2025-01-06
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.parser.coverage_parser import CoverageParser
from src.analyzer.coverage_analyzer import CoverageAnalyzer, AnalysisResult
from src.analyzer.prioritization import PrioritizationEngine
from src.predictor.closure_predictor import ClosurePredictor
from src.llm.llm_client import LLMProvider

# Initialize Typer app and Rich console
app = typer.Typer(
    name="coverage-analyzer",
    help="Verification Coverage Analyzer with LLM Integration",
    add_completion=False
)
console = Console()


def print_header():
    """Print the application header."""
    console.print(Panel.fit(
        "[bold blue]Coverage Analyzer[/bold blue]\n"
        "[dim]Verification Coverage Analysis with LLM Integration[/dim]",
        border_style="blue"
    ))


def load_coverage_file(filepath: str) -> str:
    """Load coverage report from file."""
    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]Error: File not found: {filepath}[/red]")
        raise typer.Exit(1)
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


@app.command()
def analyze(
    filepath: str = typer.Argument(..., help="Path to coverage report file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (JSON)"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider (openai/anthropic/ollama)"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Disable LLM integration (use mock suggestions)"),
    batch: bool = typer.Option(True, "--batch/--no-batch", help="Use batch mode for LLM requests"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Analyze a coverage report and generate test suggestions.
    
    This is the main command that performs complete analysis including:
    - Parsing the coverage report
    - Generating LLM-based test suggestions
    - Prioritizing suggestions
    - Generating a comprehensive report
    """
    print_header()
    
    # Load the coverage file
    console.print(f"\n[cyan]Loading coverage report from:[/cyan] {filepath}")
    coverage_text = load_coverage_file(filepath)
    
    # Initialize analyzer
    llm_provider = LLMProvider(provider.lower()) if provider else None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Parse
        task = progress.add_task("Parsing coverage report...", total=None)
        analyzer = CoverageAnalyzer(
            llm_provider=llm_provider,
            use_llm=not no_llm
        )
        progress.update(task, completed=True)
        
        # Analyze
        task = progress.add_task("Analyzing coverage and generating suggestions...", total=None)
        result = analyzer.analyze(coverage_text, batch_mode=batch)
        progress.update(task, completed=True)
    
    # Display results
    display_analysis_results(result, verbose)
    
    # Save output if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output)
        console.print(f"\n[green]✓ Results saved to:[/green] {output}")


@app.command()
def parse(
    filepath: str = typer.Argument(..., help="Path to coverage report file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (JSON)"),
    pretty: bool = typer.Option(True, "--pretty/--compact", help="Pretty print JSON output")
):
    """
    Parse a coverage report and output structured JSON.
    
    This command only parses the report without LLM integration.
    Useful for validating report format and extracting coverage data.
    """
    print_header()
    
    # Load and parse
    console.print(f"\n[cyan]Parsing coverage report:[/cyan] {filepath}")
    coverage_text = load_coverage_file(filepath)
    
    parser = CoverageParser()
    report = parser.parse(coverage_text)
    
    # Output JSON
    indent = 2 if pretty else None
    json_output = report.model_dump_json(indent=indent)
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(json_output)
        console.print(f"[green]✓ Parsed output saved to:[/green] {output}")
    else:
        console.print("\n[bold]Parsed Coverage Report:[/bold]")
        syntax = Syntax(json_output, "json", theme="monokai", line_numbers=True)
        console.print(syntax)
    
    # Print summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Design: {report.design}")
    console.print(f"  Overall Coverage: {report.overall_coverage}%")
    console.print(f"  Total Bins: {report.total_bins}")
    console.print(f"  Uncovered Bins: {report.uncovered_count}")


@app.command()
def predict(
    filepath: str = typer.Argument(..., help="Path to coverage report file"),
    deadline: Optional[float] = typer.Option(None, "--deadline", "-d", help="Deadline in hours"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (JSON)")
):
    """
    Predict coverage closure metrics.
    
    Provides estimates for:
    - Time to closure
    - Closure probability
    - Blocking bins identification
    - Risk assessment
    """
    print_header()
    
    # Load and parse
    console.print(f"\n[cyan]Loading coverage report:[/cyan] {filepath}")
    coverage_text = load_coverage_file(filepath)
    
    parser = CoverageParser()
    report = parser.parse(coverage_text)
    
    # Generate prediction
    console.print("[cyan]Generating closure prediction...[/cyan]")
    predictor = ClosurePredictor(report, deadline_hours=deadline)
    prediction = predictor.predict()
    
    # Display prediction
    display_prediction_results(prediction)
    
    # Save if requested
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(prediction.model_dump_json(indent=2))
        console.print(f"\n[green]✓ Prediction saved to:[/green] {output}")


@app.command()
def suggest(
    filepath: str = typer.Argument(..., help="Path to coverage report file"),
    bin_path: Optional[str] = typer.Option(None, "--bin", "-b", help="Specific bin to get suggestions for"),
    top: int = typer.Option(5, "--top", "-t", help="Number of top suggestions to show"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider")
):
    """
    Generate test suggestions for uncovered bins.
    
    Can generate suggestions for all uncovered bins or a specific bin.
    """
    print_header()
    
    # Load and parse
    coverage_text = load_coverage_file(filepath)
    parser = CoverageParser()
    report = parser.parse(coverage_text)
    
    # Generate suggestions
    llm_provider = LLMProvider(provider.lower()) if provider else None
    analyzer = CoverageAnalyzer(llm_provider=llm_provider)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating suggestions...", total=None)
        result = analyzer.analyze(coverage_text)
        progress.update(task, completed=True)
    
    # Filter if specific bin requested
    suggestions = result.suggestions
    if bin_path:
        suggestions = [s for s in suggestions if bin_path.lower() in s.target_bin.lower()]
    
    # Display suggestions
    display_suggestions(suggestions[:top])


@app.command()
def demo():
    """
    Run a demo analysis with the sample coverage report.
    
    This command demonstrates all features using the built-in sample report.
    """
    print_header()
    console.print("\n[bold yellow]Running Demo Analysis[/bold yellow]\n")
    
    # Use sample report
    sample_report = """
=======================================================
Functional Coverage Report
Design: dma_controller
Date: 2025-01-02
=======================================================

Covergroup: cg_transfer_size
  Coverage: 75.00% (6/8 bins)
  
  Coverpoint: cp_size
    bin small[0:255]          hits: 1523    covered
    bin medium[256:1023]      hits: 892     covered  
    bin large[1024:4095]      hits: 445     covered
    bin max[4096]             hits: 0       UNCOVERED
    
  Coverpoint: cp_burst_type
    bin single                hits: 2341    covered
    bin incr                  hits: 1822    covered
    bin wrap                  hits: 0       UNCOVERED
    bin fixed                 hits: 234     covered

-------------------------------------------------------
Covergroup: cg_channel_arbitration
  Coverage: 60.00% (3/5 bins)
  
  Coverpoint: cp_active_channels
    bin one_channel           hits: 5000    covered
    bin two_channels          hits: 1200    covered
    bin three_channels        hits: 45      covered
    bin four_channels         hits: 0       UNCOVERED
    bin all_eight             hits: 0       UNCOVERED

-------------------------------------------------------
Covergroup: cg_error_scenarios
  Coverage: 33.33% (2/6 bins)
  
  Coverpoint: cp_error_type
    bin no_error              hits: 10000   covered
    bin slave_error           hits: 234     covered
    bin decode_error          hits: 0       UNCOVERED
    bin timeout               hits: 0       UNCOVERED
    
  Coverpoint: cp_error_recovery
    bin retry_success         hits: 0       UNCOVERED
    bin abort                 hits: 0       UNCOVERED

-------------------------------------------------------
Cross Coverage: cross_size_burst
  Coverage: 50.00% (6/12 bins)
  
  <small, single>            hits: 500     covered
  <small, incr>              hits: 400     covered
  <small, wrap>              hits: 0       UNCOVERED
  <small, fixed>             hits: 100     covered
  <medium, single>           hits: 300     covered
  <medium, incr>             hits: 250     covered
  <medium, wrap>             hits: 0       UNCOVERED
  <medium, fixed>            hits: 0       UNCOVERED
  <large, single>            hits: 200     covered
  <large, incr>              hits: 0       UNCOVERED
  <large, wrap>              hits: 0       UNCOVERED
  <large, fixed>             hits: 0       UNCOVERED

=======================================================
Overall Coverage: 54.84%
=======================================================
"""
    
    # Run analysis
    analyzer = CoverageAnalyzer(use_llm=False)  # Use mock suggestions for demo
    result = analyzer.analyze(sample_report)
    
    # Display results
    display_analysis_results(result, verbose=True)
    
    # Also show prediction
    console.print("\n" + "=" * 60)
    predictor = ClosurePredictor(result.report, result.suggestions)
    prediction = predictor.predict()
    display_prediction_results(prediction)


def display_analysis_results(result: AnalysisResult, verbose: bool = False):
    """Display analysis results in a formatted way."""
    report = result.report
    
    # Coverage Summary Table
    console.print("\n[bold]Coverage Summary[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    table.add_row("Design", report.design)
    table.add_row("Overall Coverage", f"{report.overall_coverage}%")
    table.add_row("Total Bins", str(report.total_bins))
    table.add_row("Covered Bins", str(report.covered_bins))
    table.add_row("Uncovered Bins", str(report.uncovered_count))
    table.add_row("Gap to 100%", f"{100 - report.overall_coverage:.2f}%")
    
    console.print(table)
    
    # Covergroup Breakdown
    console.print("\n[bold]Covergroup Breakdown[/bold]")
    cg_table = Table(show_header=True, header_style="bold green")
    cg_table.add_column("Covergroup")
    cg_table.add_column("Coverage", justify="right")
    cg_table.add_column("Uncovered", justify="right")
    cg_table.add_column("Status")
    
    for cg in report.covergroups:
        status = "✓" if cg.coverage >= 80 else "⚠" if cg.coverage >= 50 else "✗"
        color = "green" if cg.coverage >= 80 else "yellow" if cg.coverage >= 50 else "red"
        cg_table.add_row(
            cg.name,
            f"{cg.coverage}%",
            str(cg.uncovered_count),
            f"[{color}]{status}[/{color}]"
        )
    
    console.print(cg_table)
    
    # Top Suggestions
    if result.suggestions:
        console.print("\n[bold]Top Priority Suggestions[/bold]")
        display_suggestions(result.suggestions[:5])
    
    # Recommendations
    if result.summary.get("recommendations"):
        console.print("\n[bold]Recommendations[/bold]")
        for rec in result.summary["recommendations"]:
            console.print(f"  • {rec}")
    
    # Performance
    if verbose:
        console.print("\n[bold]Performance Metrics[/bold]")
        perf = result.performance
        console.print(f"  Parse Time: {perf.get('parse_time_seconds', 0):.3f}s")
        console.print(f"  LLM Time: {perf.get('llm_time_seconds', 0):.3f}s")
        console.print(f"  Total Time: {perf.get('total_time_seconds', 0):.3f}s")


def display_suggestions(suggestions: list):
    """Display suggestions in a formatted table."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Target Bin", width=40)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Difficulty", width=10)
    table.add_column("Priority", width=8)
    
    for i, s in enumerate(suggestions, 1):
        diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(s.difficulty.lower(), "white")
        pri_color = {"high": "red", "medium": "yellow", "low": "green"}.get(s.priority.lower(), "white")
        
        table.add_row(
            str(i),
            s.target_bin[:40],
            f"{s.priority_score:.3f}",
            f"[{diff_color}]{s.difficulty}[/{diff_color}]",
            f"[{pri_color}]{s.priority}[/{pri_color}]"
        )
    
    console.print(table)
    
    # Show details for top suggestion
    if suggestions:
        top = suggestions[0]
        console.print(f"\n[bold]Top Suggestion Details:[/bold]")
        console.print(f"  Target: {top.target_bin}")
        console.print(f"  Suggestion: {top.suggestion}")
        console.print(f"  Reasoning: {top.reasoning}")
        if top.test_outline:
            console.print("  Test Outline:")
            for step in top.test_outline:
                console.print(f"    {step}")


def display_prediction_results(prediction):
    """Display prediction results."""
    console.print("\n[bold]Coverage Closure Prediction[/bold]")
    
    # Time estimates
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    
    table.add_row("Est. Time to 100%", f"{prediction.estimated_time_to_closure_hours:.1f} hours")
    table.add_row("Est. Time to 90%", f"{prediction.estimated_time_to_90_percent_hours:.1f} hours")
    table.add_row(
        "Confidence Interval",
        f"{prediction.confidence_interval_hours[0]:.1f} - {prediction.confidence_interval_hours[1]:.1f} hours"
    )
    table.add_row("Closure Probability", f"{prediction.closure_probability * 100:.1f}%")
    table.add_row("Max Achievable Coverage", f"{prediction.achievable_coverage:.1f}%")
    table.add_row("Coverage Velocity", f"{prediction.current_coverage_velocity:.2f} bins/hour")
    
    console.print(table)
    
    # Risk Assessment
    risk_color = {
        "low": "green",
        "medium": "yellow",
        "high": "red",
        "critical": "bold red"
    }.get(prediction.risk_level, "white")
    
    console.print(f"\n[bold]Risk Level:[/bold] [{risk_color}]{prediction.risk_level.upper()}[/{risk_color}]")
    
    if prediction.risk_factors:
        console.print("\n[bold]Risk Factors:[/bold]")
        for rf in prediction.risk_factors:
            console.print(f"  ⚠ {rf}")
    
    # Blocking bins
    if prediction.total_blocking_bins > 0:
        console.print(f"\n[bold]Potentially Blocking Bins:[/bold] {prediction.total_blocking_bins}")
        blocking = [b for b in prediction.blocking_bins if b.is_potentially_blocking][:3]
        for b in blocking:
            console.print(f"  • {b.bin_path}: {b.recommendation}")
    
    # Recommendations
    if prediction.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in prediction.recommendations:
            console.print(f"  → {rec}")


@app.callback()
def main():
    """
    Verification Coverage Analyzer with LLM Integration.
    
    A tool for analyzing functional coverage reports and generating
    AI-powered test suggestions to accelerate coverage closure.
    """
    pass


if __name__ == "__main__":
    app()
