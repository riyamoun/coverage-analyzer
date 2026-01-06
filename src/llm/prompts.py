"""
Prompt Engineering Module

Provides carefully crafted prompts for generating test suggestions
based on coverage analysis. Uses few-shot learning with domain-specific
examples for optimal results.

Author: ML Engineer
Date: 2025-01-06
"""

from typing import Optional
from ..parser.coverage_parser import CoverageReport, UncoveredBinInfo, CrossCoverageInfo


class PromptBuilder:
    """
    Builds optimized prompts for LLM-based test suggestion generation.
    
    Features:
    - Few-shot learning with verification domain examples
    - Context-aware prompt construction
    - Structured output formatting
    - IP-specific context injection
    
    Usage:
        builder = PromptBuilder()
        prompt = builder.build_suggestion_prompt(report, uncovered_bin)
    """
    
    SYSTEM_PROMPT = """You are an expert chip verification engineer specializing in functional coverage closure. Your role is to analyze coverage reports and suggest specific, actionable test scenarios to hit uncovered bins.

You have deep knowledge of:
- DMA controllers and their protocols (AXI, AHB, APB)
- Bus protocols and burst types (INCR, WRAP, FIXED)
- Error injection and recovery mechanisms
- Coverage-driven verification methodologies
- SystemVerilog testbench architecture

When suggesting tests, you should:
1. Provide specific, implementable test scenarios
2. Include step-by-step test outlines
3. Identify dependencies and prerequisites
4. Explain the reasoning behind your suggestions
5. Consider the relationship between covered and uncovered bins

Always respond with valid JSON matching the specified format."""

    FEW_SHOT_EXAMPLES = """
Here are examples of good test suggestions:

Example 1 - Burst Type Coverage:
Input: Uncovered bin "wrap" in cp_burst_type, while "single", "incr", "fixed" are covered
Output:
{
  "target_bin": "cg_transfer_size.cp_burst_type.wrap",
  "priority": "high",
  "difficulty": "medium",
  "suggestion": "Create a test sequence that configures the DMA for wrap burst mode with a burst length that causes address wrapping. Set base address near a wrap boundary (e.g., 0xFFC for 4KB boundary) with burst length of 4.",
  "test_outline": [
    "1. Configure DMA channel with wrap burst type (AWBURST=2'b10)",
    "2. Set transfer size to 4 beats to trigger address wrapping",
    "3. Set base address near 4KB boundary (0xFFC)",
    "4. Start transfer and verify wrap behavior at boundary",
    "5. Check that address wraps back to boundary-aligned address"
  ],
  "dependencies": ["Ensure AXI slave supports wrap bursts", "Verify memory controller wrap support"],
  "reasoning": "Wrap bursts are used for cache-line fills in processors. The coverage shows INCR and SINGLE work, indicating basic DMA transfer functionality is correct. Wrap mode requires specific burst configuration and proper address alignment."
}

Example 2 - Error Scenario Coverage:
Input: Uncovered bin "decode_error" in cp_error_type, while "no_error" and "slave_error" are covered
Output:
{
  "target_bin": "cg_error_scenarios.cp_error_type.decode_error",
  "priority": "medium",
  "difficulty": "hard",
  "suggestion": "Inject a decode error by programming DMA to access an unmapped address region. Configure the address decoder to return DECERR for accesses to reserved address space.",
  "test_outline": [
    "1. Review memory map to identify unmapped/reserved address ranges",
    "2. Configure DMA source or destination to access unmapped region (e.g., 0xDEAD_0000)",
    "3. Ensure interconnect is configured to return DECERR (not SLVERR)",
    "4. Start DMA transfer",
    "5. Verify DECERR response on AXI bus",
    "6. Check DMA error status register for decode error indication",
    "7. Verify error interrupt is generated if enabled"
  ],
  "dependencies": ["Testbench must have unmapped address region", "Need to verify AXI interconnect returns DECERR vs SLVERR", "May need address decoder configuration"],
  "reasoning": "Decode errors require accessing addresses that no slave claims. Since slave_error is already covered, the error handling path works. The key is to trigger specifically a DECERR response, which requires accessing truly unmapped addresses in the memory map."
}

Example 3 - Multi-Channel Coverage:
Input: Uncovered bin "four_channels" in cp_active_channels, while one/two/three channels are covered
Output:
{
  "target_bin": "cg_channel_arbitration.cp_active_channels.four_channels",
  "priority": "high",
  "difficulty": "easy",
  "suggestion": "Create a test that simultaneously enables and starts transfers on exactly 4 DMA channels. Use different transfer configurations to ensure sustained activity.",
  "test_outline": [
    "1. Configure channels 0-3 with valid source, destination, and transfer size",
    "2. Use varying transfer sizes (e.g., Ch0=1KB, Ch1=2KB, Ch2=512B, Ch3=4KB)",
    "3. Enable all 4 channels simultaneously",
    "4. Start transfers on all channels in quick succession",
    "5. Monitor channel busy/active status register",
    "6. Verify arbitration between channels",
    "7. Wait for all transfers to complete"
  ],
  "dependencies": ["Sufficient memory regions for 4 concurrent transfers", "No channel dependencies that prevent parallel operation"],
  "reasoning": "Since one, two, and three channels work correctly, the multi-channel infrastructure is functional. Four channels is a straightforward extension - just need to ensure all 4 channels are configured and active simultaneously when coverage is sampled."
}
"""

    SUGGESTION_PROMPT_TEMPLATE = """
Analyze the following coverage information and generate a test suggestion to cover the uncovered bin.

=== DESIGN CONTEXT ===
Design: {design_name}
Overall Coverage: {overall_coverage}%
Total Uncovered Bins: {total_uncovered}

=== TARGET UNCOVERED BIN ===
Covergroup: {covergroup}
Coverpoint: {coverpoint}
Bin: {bin_name}
Bin Hits: {bin_hits}

=== RELATED COVERED BINS (same coverpoint) ===
{covered_bins_info}

=== COVERAGE CONTEXT ===
{coverage_context}

=== INSTRUCTIONS ===
Generate a detailed test suggestion to cover the target bin. Consider:
1. What makes this bin different from the covered bins?
2. What specific configuration or scenario would hit this bin?
3. Are there any dependencies or prerequisites?
4. What is the difficulty level (easy/medium/hard)?

Respond with a JSON object in this exact format:
{{
  "target_bin": "{full_bin_path}",
  "priority": "high|medium|low",
  "difficulty": "easy|medium|hard",
  "suggestion": "Detailed description of the test scenario",
  "test_outline": ["Step 1...", "Step 2...", "Step 3..."],
  "dependencies": ["Dependency 1", "Dependency 2"],
  "reasoning": "Explanation of why this approach will work"
}}
"""

    CROSS_COVERAGE_PROMPT_TEMPLATE = """
Analyze the following cross-coverage information and generate test suggestions for uncovered cross bins.

=== DESIGN CONTEXT ===
Design: {design_name}
Overall Coverage: {overall_coverage}%

=== CROSS COVERAGE ===
Name: {cross_name}
Coverage: {cross_coverage}%

Covered combinations:
{covered_crosses}

Uncovered combinations:
{uncovered_crosses}

=== INSTRUCTIONS ===
For each uncovered cross combination, determine:
1. What specific configuration would hit this combination?
2. Why might it be harder than the covered combinations?
3. Are there dependencies between the cross dimensions?

Respond with a JSON array of suggestions, one for each uncovered combination:
{{
  "suggestions": [
    {{
      "target_bin": "<cross_values>",
      "priority": "high|medium|low",
      "difficulty": "easy|medium|hard",
      "suggestion": "Detailed test scenario",
      "test_outline": ["Step 1...", "Step 2..."],
      "dependencies": ["..."],
      "reasoning": "..."
    }}
  ]
}}
"""

    BATCH_SUGGESTION_PROMPT_TEMPLATE = """
Analyze the following coverage report and generate test suggestions for all uncovered bins.

=== COMPLETE COVERAGE REPORT ===
{coverage_report_json}

=== INSTRUCTIONS ===
Generate prioritized test suggestions for the uncovered bins. Focus on:
1. High-impact bins that will significantly improve coverage
2. Easy wins that can be achieved quickly
3. Related bins that might be covered by the same test

Respond with a JSON object containing all suggestions:
{{
  "suggestions": [
    {{
      "target_bin": "covergroup.coverpoint.bin",
      "priority": "high|medium|low",
      "difficulty": "easy|medium|hard",
      "suggestion": "...",
      "test_outline": ["..."],
      "dependencies": ["..."],
      "reasoning": "..."
    }}
  ],
  "grouped_suggestions": [
    {{
      "target_bins": ["bin1", "bin2"],
      "shared_test": "A single test that can hit multiple bins",
      "test_outline": ["..."],
      "reasoning": "..."
    }}
  ]
}}
"""

    def __init__(self):
        """Initialize the prompt builder."""
        pass
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return self.SYSTEM_PROMPT
    
    def build_suggestion_prompt(
        self,
        report: CoverageReport,
        uncovered_bin: UncoveredBinInfo
    ) -> str:
        """
        Build a prompt for generating a suggestion for a single uncovered bin.
        
        Args:
            report: The parsed coverage report
            uncovered_bin: The uncovered bin to target
            
        Returns:
            Formatted prompt string
        """
        # Find the covergroup and coverpoint
        covergroup = next(
            (cg for cg in report.covergroups if cg.name == uncovered_bin.covergroup),
            None
        )
        coverpoint = None
        if covergroup:
            coverpoint = next(
                (cp for cp in covergroup.coverpoints if cp.name == uncovered_bin.coverpoint),
                None
            )
        
        # Build covered bins info
        covered_bins_info = "None found"
        if coverpoint:
            covered = [b for b in coverpoint.bins if b.covered]
            if covered:
                covered_bins_info = "\n".join([
                    f"  - {b.name}{b.range or ''}: {b.hits} hits"
                    for b in covered
                ])
        
        # Build coverage context
        context_lines = []
        for cg in report.covergroups:
            context_lines.append(f"  {cg.name}: {cg.coverage}% ({cg.covered_bins}/{cg.total_bins} bins)")
        coverage_context = "\n".join(context_lines)
        
        full_bin_path = f"{uncovered_bin.covergroup}.{uncovered_bin.coverpoint}.{uncovered_bin.bin}"
        
        return self.FEW_SHOT_EXAMPLES + "\n\n" + self.SUGGESTION_PROMPT_TEMPLATE.format(
            design_name=report.design,
            overall_coverage=report.overall_coverage,
            total_uncovered=report.uncovered_count,
            covergroup=uncovered_bin.covergroup,
            coverpoint=uncovered_bin.coverpoint,
            bin_name=uncovered_bin.bin,
            bin_hits=uncovered_bin.hits,
            covered_bins_info=covered_bins_info,
            coverage_context=coverage_context,
            full_bin_path=full_bin_path
        )
    
    def build_cross_coverage_prompt(
        self,
        report: CoverageReport,
        cross_coverage: CrossCoverageInfo
    ) -> str:
        """
        Build a prompt for generating suggestions for cross-coverage holes.
        
        Args:
            report: The parsed coverage report
            cross_coverage: The cross coverage to analyze
            
        Returns:
            Formatted prompt string
        """
        covered_crosses = "\n".join([
            f"  {b.values}: {b.hits} hits"
            for b in cross_coverage.bins if b.covered
        ])
        
        uncovered_crosses = "\n".join([
            f"  {v}"
            for v in cross_coverage.uncovered
        ])
        
        return self.FEW_SHOT_EXAMPLES + "\n\n" + self.CROSS_COVERAGE_PROMPT_TEMPLATE.format(
            design_name=report.design,
            overall_coverage=report.overall_coverage,
            cross_name=cross_coverage.name,
            cross_coverage=cross_coverage.coverage,
            covered_crosses=covered_crosses or "  None",
            uncovered_crosses=uncovered_crosses or "  None"
        )
    
    def build_batch_prompt(self, report: CoverageReport) -> str:
        """
        Build a prompt for generating suggestions for all uncovered bins.
        
        Args:
            report: The parsed coverage report
            
        Returns:
            Formatted prompt string
        """
        import json
        
        # Create a condensed version of the report for the prompt
        report_dict = {
            "design": report.design,
            "overall_coverage": report.overall_coverage,
            "covergroups": [
                {
                    "name": cg.name,
                    "coverage": cg.coverage,
                    "coverpoints": [
                        {
                            "name": cp.name,
                            "bins": [
                                {
                                    "name": b.name,
                                    "range": b.range,
                                    "hits": b.hits,
                                    "covered": b.covered
                                }
                                for b in cp.bins
                            ]
                        }
                        for cp in cg.coverpoints
                    ]
                }
                for cg in report.covergroups
            ],
            "uncovered_bins": [
                {"covergroup": ub.covergroup, "coverpoint": ub.coverpoint, "bin": ub.bin}
                for ub in report.uncovered_bins
            ],
            "cross_coverage": [
                {
                    "name": xc.name,
                    "coverage": xc.coverage,
                    "uncovered": xc.uncovered
                }
                for xc in report.cross_coverage
            ]
        }
        
        return self.FEW_SHOT_EXAMPLES + "\n\n" + self.BATCH_SUGGESTION_PROMPT_TEMPLATE.format(
            coverage_report_json=json.dumps(report_dict, indent=2)
        )
    
    def build_analysis_prompt(self, report: CoverageReport) -> str:
        """
        Build a prompt for overall coverage analysis.
        
        Args:
            report: The parsed coverage report
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Analyze the following coverage report and provide insights:

Design: {report.design}
Overall Coverage: {report.overall_coverage}%
Total Bins: {report.total_bins}
Covered Bins: {report.covered_bins}
Uncovered Bins: {report.uncovered_count}

Coverage by Covergroup:
"""
        for cg in report.covergroups:
            prompt += f"  {cg.name}: {cg.coverage}%\n"
        
        prompt += """
Provide:
1. Summary of coverage status
2. Identification of critical gaps
3. Recommended focus areas
4. Risk assessment for uncovered functionality
"""
        return prompt


# Convenience function
def create_prompt_builder() -> PromptBuilder:
    """Create a new prompt builder instance."""
    return PromptBuilder()
