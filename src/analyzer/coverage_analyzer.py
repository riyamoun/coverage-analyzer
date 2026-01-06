"""
Coverage Analyzer Module

The main orchestrator that combines parsing, LLM integration, and prioritization
to provide comprehensive coverage analysis and test suggestions.

Author: ML Engineer
Date: 2025-01-06
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from ..parser.coverage_parser import CoverageParser, CoverageReport
from ..llm.llm_client import LLMClient, LLMResponse, TestSuggestion, LLMProvider
from .prioritization import PrioritizationEngine, PrioritizedSuggestion


class AnalysisResult(BaseModel):
    """Complete analysis result including parsed report, suggestions, and metrics."""
    
    # Metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: str = Field(..., description="Analysis timestamp")
    
    # Parsed coverage data
    report: CoverageReport = Field(..., description="Parsed coverage report")
    
    # Suggestions
    suggestions: list[PrioritizedSuggestion] = Field(
        default_factory=list,
        description="Prioritized test suggestions"
    )
    
    # Summary statistics
    summary: dict = Field(default_factory=dict, description="Analysis summary")
    
    # Performance metrics
    performance: dict = Field(default_factory=dict, description="Performance metrics")
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=indent)
    
    def save(self, filepath: str) -> None:
        """Save analysis result to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


class CoverageAnalyzer:
    """
    Main coverage analyzer that orchestrates the entire analysis pipeline.
    
    This class combines:
    - Coverage report parsing
    - LLM-based suggestion generation
    - Priority scoring and ranking
    - Result aggregation and reporting
    
    Usage:
        analyzer = CoverageAnalyzer()
        result = analyzer.analyze(coverage_text)
        
        # Or from file
        result = analyzer.analyze_file("coverage_report.txt")
        
        # Get prioritized suggestions
        for suggestion in result.suggestions:
            print(f"{suggestion.rank}. {suggestion.target_bin}: {suggestion.priority_score}")
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_llm: bool = True
    ):
        """
        Initialize the coverage analyzer.
        
        Args:
            llm_provider: LLM provider to use
            api_key: API key for the provider
            model: Model name to use
            use_llm: Whether to use LLM for suggestions (set False for testing)
        """
        self.parser = CoverageParser()
        self.use_llm = use_llm
        
        if use_llm:
            self.llm_client = LLMClient(
                provider=llm_provider,
                api_key=api_key,
                model=model
            )
        else:
            self.llm_client = None
        
        self._analysis_count = 0
    
    def analyze(
        self,
        coverage_text: str,
        batch_mode: bool = True,
        generate_id: bool = True
    ) -> AnalysisResult:
        """
        Perform complete analysis on a coverage report.
        
        Args:
            coverage_text: Raw coverage report text
            batch_mode: Whether to use batch LLM requests
            generate_id: Whether to generate a unique analysis ID
            
        Returns:
            AnalysisResult with all analysis data
        """
        start_time = datetime.now()
        
        # Generate analysis ID
        if generate_id:
            self._analysis_count += 1
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._analysis_count}"
        else:
            analysis_id = "analysis_manual"
        
        # Step 1: Parse the coverage report
        parse_start = datetime.now()
        report = self.parser.parse(coverage_text)
        parse_time = (datetime.now() - parse_start).total_seconds()
        
        # Step 2: Generate suggestions using LLM
        suggestions: list[TestSuggestion] = []
        llm_time = 0.0
        
        if self.use_llm and self.llm_client:
            llm_start = datetime.now()
            try:
                llm_response = self.llm_client.generate_all_suggestions(
                    report,
                    batch_mode=batch_mode
                )
                suggestions = llm_response.suggestions
            except Exception as e:
                print(f"Warning: LLM suggestion generation failed: {e}")
                # Generate mock suggestions for demo
                suggestions = self._generate_mock_suggestions(report)
            llm_time = (datetime.now() - llm_start).total_seconds()
        else:
            # Generate mock suggestions when LLM is disabled
            suggestions = self._generate_mock_suggestions(report)
        
        # Step 3: Prioritize suggestions
        prioritization_start = datetime.now()
        engine = PrioritizationEngine(report)
        prioritized = engine.prioritize(suggestions)
        prioritization_time = (datetime.now() - prioritization_start).total_seconds()
        
        # Step 4: Generate summary
        summary = self._generate_summary(report, prioritized, engine)
        
        # Calculate total time
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Build performance metrics
        performance = {
            "parse_time_seconds": round(parse_time, 3),
            "llm_time_seconds": round(llm_time, 3),
            "prioritization_time_seconds": round(prioritization_time, 3),
            "total_time_seconds": round(total_time, 3),
            "suggestions_generated": len(suggestions),
            "llm_stats": self.llm_client.get_stats() if self.llm_client else {}
        }
        
        return AnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            report=report,
            suggestions=prioritized,
            summary=summary,
            performance=performance
        )
    
    def analyze_file(self, filepath: str, **kwargs) -> AnalysisResult:
        """
        Analyze a coverage report from a file.
        
        Args:
            filepath: Path to the coverage report file
            **kwargs: Additional arguments passed to analyze()
            
        Returns:
            AnalysisResult
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.analyze(text, **kwargs)
    
    def _generate_summary(
        self,
        report: CoverageReport,
        prioritized: list[PrioritizedSuggestion],
        engine: PrioritizationEngine
    ) -> dict:
        """Generate comprehensive analysis summary."""
        prioritization_summary = engine.generate_summary(prioritized)
        
        return {
            "design": report.design,
            "analysis_date": datetime.now().isoformat(),
            "coverage": {
                "overall": report.overall_coverage,
                "gap_to_100": round(100 - report.overall_coverage, 2),
                "total_bins": report.total_bins,
                "covered_bins": report.covered_bins,
                "uncovered_bins": report.uncovered_count
            },
            "covergroups": {
                cg.name: {
                    "coverage": cg.coverage,
                    "uncovered": cg.uncovered_count
                }
                for cg in report.covergroups
            },
            "cross_coverage": {
                xc.name: {
                    "coverage": xc.coverage,
                    "uncovered": len(xc.uncovered)
                }
                for xc in report.cross_coverage
            },
            "suggestions": prioritization_summary,
            "recommendations": self._generate_recommendations(report, prioritized)
        }
    
    def _generate_recommendations(
        self,
        report: CoverageReport,
        prioritized: list[PrioritizedSuggestion]
    ) -> list[str]:
        """Generate high-level recommendations based on analysis."""
        recommendations = []
        
        # Coverage-based recommendations
        if report.overall_coverage < 50:
            recommendations.append(
                "CRITICAL: Overall coverage is below 50%. Focus on increasing basic functional coverage first."
            )
        elif report.overall_coverage < 80:
            recommendations.append(
                "Coverage is moderate. Focus on quick wins to rapidly improve coverage."
            )
        else:
            recommendations.append(
                "Coverage is good. Focus on closing the remaining gaps, which may require more complex tests."
            )
        
        # Identify problem areas
        for cg in report.covergroups:
            if cg.coverage < 50:
                recommendations.append(
                    f"Covergroup '{cg.name}' has low coverage ({cg.coverage}%). "
                    f"Prioritize this area."
                )
        
        # Error coverage check
        for cg in report.covergroups:
            if "error" in cg.name.lower():
                if cg.coverage < 60:
                    recommendations.append(
                        f"Error scenario coverage ({cg.coverage}%) is low. "
                        f"Consider adding error injection tests."
                    )
        
        # Cross-coverage recommendations
        for xc in report.cross_coverage:
            if xc.coverage < 50:
                recommendations.append(
                    f"Cross-coverage '{xc.name}' ({xc.coverage}%) needs attention. "
                    f"Create tests that exercise multiple feature combinations."
                )
        
        # Suggestion-based recommendations
        if prioritized:
            easy_count = sum(1 for p in prioritized if p.difficulty.lower() == "easy")
            if easy_count > 0:
                recommendations.append(
                    f"There are {easy_count} easy-difficulty suggestions. "
                    f"Start with these for quick coverage gains."
                )
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_mock_suggestions(
        self,
        report: CoverageReport
    ) -> list[TestSuggestion]:
        """
        Generate mock suggestions when LLM is not available.
        
        This is useful for testing and demo purposes.
        """
        suggestions = []
        
        # DMA-specific knowledge base
        knowledge_base = {
            "wrap": {
                "suggestion": "Configure DMA for wrap burst mode with address near boundary",
                "test_outline": [
                    "1. Configure DMA channel with wrap burst type",
                    "2. Set base address near wrap boundary (e.g., 0xFFC)",
                    "3. Start transfer and verify wrap behavior"
                ],
                "dependencies": ["AXI slave must support wrap bursts"],
                "difficulty": "medium"
            },
            "max": {
                "suggestion": "Configure DMA with maximum transfer size of 4096 bytes",
                "test_outline": [
                    "1. Configure DMA for 4KB transfer",
                    "2. Ensure sufficient buffer space",
                    "3. Verify complete transfer at max size"
                ],
                "dependencies": [],
                "difficulty": "easy"
            },
            "four_channels": {
                "suggestion": "Enable and start transfers on exactly 4 DMA channels simultaneously",
                "test_outline": [
                    "1. Configure channels 0-3 with valid parameters",
                    "2. Enable all 4 channels simultaneously",
                    "3. Start transfers and monitor"
                ],
                "dependencies": ["Sufficient memory for 4 concurrent transfers"],
                "difficulty": "easy"
            },
            "all_eight": {
                "suggestion": "Configure all 8 DMA channels for concurrent operation",
                "test_outline": [
                    "1. Configure all 8 channels",
                    "2. Enable and start all channels",
                    "3. Verify arbitration between channels"
                ],
                "dependencies": ["8-channel DMA support"],
                "difficulty": "medium"
            },
            "decode_error": {
                "suggestion": "Program DMA to access unmapped address region to trigger DECERR",
                "test_outline": [
                    "1. Identify unmapped address range",
                    "2. Configure DMA to access unmapped region",
                    "3. Verify decode error response"
                ],
                "dependencies": ["Unmapped address region in memory map"],
                "difficulty": "hard"
            },
            "timeout": {
                "suggestion": "Inject bus stall to trigger DMA timeout condition",
                "test_outline": [
                    "1. Configure timeout threshold",
                    "2. Inject bus stall or slow response",
                    "3. Verify timeout detection"
                ],
                "dependencies": ["Testbench stall injection capability"],
                "difficulty": "hard"
            },
            "retry_success": {
                "suggestion": "Configure retry mechanism and trigger transient error followed by success",
                "test_outline": [
                    "1. Enable retry on error",
                    "2. Inject transient error (first attempt fails)",
                    "3. Allow retry to succeed"
                ],
                "dependencies": ["Error injection and retry configuration"],
                "difficulty": "hard"
            },
            "abort": {
                "suggestion": "Trigger non-recoverable error to exercise abort path",
                "test_outline": [
                    "1. Configure abort condition",
                    "2. Trigger non-recoverable error",
                    "3. Verify abort handling"
                ],
                "dependencies": ["Abort mechanism enabled"],
                "difficulty": "medium"
            }
        }
        
        # Generate suggestions for each uncovered bin
        for ub in report.uncovered_bins:
            bin_name = ub.bin.split('[')[0].lower()  # Remove range for matching
            
            if bin_name in knowledge_base:
                kb = knowledge_base[bin_name]
            else:
                # Default suggestion
                kb = {
                    "suggestion": f"Create test to cover {ub.bin} in {ub.coverpoint}",
                    "test_outline": [
                        f"1. Configure testbench for {ub.bin}",
                        "2. Execute test sequence",
                        "3. Verify coverage hit"
                    ],
                    "dependencies": [],
                    "difficulty": "medium"
                }
            
            suggestions.append(TestSuggestion(
                target_bin=ub.full_path,
                priority="high" if "error" in ub.covergroup.lower() else "medium",
                difficulty=kb["difficulty"],
                suggestion=kb["suggestion"],
                test_outline=kb["test_outline"],
                dependencies=kb["dependencies"],
                reasoning=f"This bin is uncovered in {ub.covergroup}.{ub.coverpoint}. "
                          f"Similar bins are covered, suggesting the infrastructure works."
            ))
        
        # Add cross-coverage suggestions
        for xc in report.cross_coverage:
            for uncovered in xc.uncovered[:3]:  # Limit to first 3
                suggestions.append(TestSuggestion(
                    target_bin=f"{xc.name}.{uncovered}",
                    priority="medium",
                    difficulty="medium",
                    suggestion=f"Create test that exercises the combination {uncovered}",
                    test_outline=[
                        f"1. Configure for cross-coverage combination {uncovered}",
                        "2. Execute combined scenario",
                        "3. Verify both dimensions are active simultaneously"
                    ],
                    dependencies=["Both cross dimensions must be exercised together"],
                    reasoning=f"Cross-coverage requires simultaneous activation of "
                              f"multiple coverpoints."
                ))
        
        return suggestions
    
    def get_quick_report(self, result: AnalysisResult) -> str:
        """
        Generate a quick text report of the analysis.
        
        Args:
            result: The analysis result
            
        Returns:
            Formatted text report
        """
        lines = [
            "=" * 60,
            f"COVERAGE ANALYSIS REPORT",
            f"Design: {result.report.design}",
            f"Analysis ID: {result.analysis_id}",
            "=" * 60,
            "",
            f"Overall Coverage: {result.report.overall_coverage}%",
            f"Gap to 100%: {100 - result.report.overall_coverage:.2f}%",
            f"Total Bins: {result.report.total_bins}",
            f"Uncovered Bins: {result.report.uncovered_count}",
            "",
            "COVERGROUP BREAKDOWN:",
            "-" * 40,
        ]
        
        for cg in result.report.covergroups:
            lines.append(f"  {cg.name}: {cg.coverage}% ({cg.uncovered_count} uncovered)")
        
        if result.report.cross_coverage:
            lines.extend([
                "",
                "CROSS-COVERAGE:",
                "-" * 40,
            ])
            for xc in result.report.cross_coverage:
                lines.append(f"  {xc.name}: {xc.coverage}% ({len(xc.uncovered)} uncovered)")
        
        lines.extend([
            "",
            "TOP PRIORITY SUGGESTIONS:",
            "-" * 40,
        ])
        
        for suggestion in result.suggestions[:5]:
            lines.extend([
                f"  #{suggestion.rank} [{suggestion.priority_score:.3f}] {suggestion.target_bin}",
                f"      Difficulty: {suggestion.difficulty}, Priority: {suggestion.priority}",
                f"      {suggestion.suggestion[:80]}...",
                ""
            ])
        
        lines.extend([
            "",
            "RECOMMENDATIONS:",
            "-" * 40,
        ])
        
        for rec in result.summary.get("recommendations", []):
            lines.append(f"  • {rec}")
        
        lines.extend([
            "",
            "=" * 60,
            f"Analysis completed in {result.performance.get('total_time_seconds', 0):.2f} seconds",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def create_analyzer(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    use_llm: bool = True
) -> CoverageAnalyzer:
    """
    Factory function to create a coverage analyzer.
    
    Args:
        provider: LLM provider name
        api_key: API key
        use_llm: Whether to enable LLM integration
        
    Returns:
        Configured CoverageAnalyzer
    """
    provider_enum = LLMProvider(provider.lower()) if provider else None
    return CoverageAnalyzer(llm_provider=provider_enum, api_key=api_key, use_llm=use_llm)
