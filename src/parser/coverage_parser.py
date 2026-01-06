"""
Coverage Report Parser Module

This module provides a robust parser for functional coverage reports commonly
used in chip verification. It extracts structured data from coverage reports
including covergroups, coverpoints, bins, and cross-coverage information.

Author: ML Engineer
Date: 2025-01-06
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, Field


class BinInfo(BaseModel):
    """Represents a single coverage bin."""
    name: str = Field(..., description="Name of the bin")
    range: Optional[str] = Field(None, description="Range specification (e.g., '[0:255]')")
    hits: int = Field(..., description="Number of hits recorded")
    covered: bool = Field(..., description="Whether the bin is covered")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "small",
                "range": "[0:255]",
                "hits": 1523,
                "covered": True
            }
        }


class CoverpointInfo(BaseModel):
    """Represents a coverpoint containing multiple bins."""
    name: str = Field(..., description="Name of the coverpoint")
    bins: list[BinInfo] = Field(default_factory=list, description="List of bins in this coverpoint")
    
    @property
    def coverage_percentage(self) -> float:
        """Calculate coverage percentage for this coverpoint."""
        if not self.bins:
            return 0.0
        covered = sum(1 for b in self.bins if b.covered)
        return (covered / len(self.bins)) * 100
    
    @property
    def uncovered_bins(self) -> list[BinInfo]:
        """Get list of uncovered bins."""
        return [b for b in self.bins if not b.covered]


class CovergroupInfo(BaseModel):
    """Represents a covergroup containing multiple coverpoints."""
    name: str = Field(..., description="Name of the covergroup")
    coverage: float = Field(..., description="Coverage percentage")
    covered_bins: int = Field(0, description="Number of covered bins")
    total_bins: int = Field(0, description="Total number of bins")
    coverpoints: list[CoverpointInfo] = Field(
        default_factory=list, 
        description="List of coverpoints in this covergroup"
    )
    
    @property
    def uncovered_count(self) -> int:
        """Get count of uncovered bins across all coverpoints."""
        return sum(len(cp.uncovered_bins) for cp in self.coverpoints)


class CrossCoverageBin(BaseModel):
    """Represents a single cross-coverage bin."""
    values: str = Field(..., description="Cross values (e.g., '<small, wrap>')")
    hits: int = Field(..., description="Number of hits")
    covered: bool = Field(..., description="Whether this cross bin is covered")


class CrossCoverageInfo(BaseModel):
    """Represents cross-coverage information."""
    name: str = Field(..., description="Name of the cross coverage")
    coverage: float = Field(..., description="Coverage percentage")
    covered_bins: int = Field(0, description="Number of covered bins")
    total_bins: int = Field(0, description="Total number of bins")
    bins: list[CrossCoverageBin] = Field(
        default_factory=list, 
        description="List of cross coverage bins"
    )
    uncovered: list[str] = Field(
        default_factory=list, 
        description="List of uncovered cross value combinations"
    )


class UncoveredBinInfo(BaseModel):
    """Represents an uncovered bin with full path information."""
    covergroup: str = Field(..., description="Parent covergroup name")
    coverpoint: str = Field(..., description="Parent coverpoint name")
    bin: str = Field(..., description="Bin name with range")
    hits: int = Field(0, description="Number of hits (should be 0 for uncovered)")
    
    @property
    def full_path(self) -> str:
        """Get full hierarchical path to this bin."""
        return f"{self.covergroup}.{self.coverpoint}.{self.bin}"


class CoverageReport(BaseModel):
    """
    Complete parsed coverage report with all extracted information.
    
    This is the main output structure of the parser, containing:
    - Design metadata (name, date)
    - Overall coverage metrics
    - Detailed covergroup information
    - List of all uncovered bins
    - Cross-coverage information
    """
    design: str = Field(..., description="Design name")
    date: Optional[str] = Field(None, description="Report generation date")
    overall_coverage: float = Field(..., description="Overall coverage percentage")
    covergroups: list[CovergroupInfo] = Field(
        default_factory=list, 
        description="List of all covergroups"
    )
    uncovered_bins: list[UncoveredBinInfo] = Field(
        default_factory=list, 
        description="Flattened list of all uncovered bins"
    )
    cross_coverage: list[CrossCoverageInfo] = Field(
        default_factory=list, 
        description="List of cross-coverage information"
    )
    
    # Additional metadata for analysis
    total_bins: int = Field(0, description="Total number of bins across all covergroups")
    covered_bins: int = Field(0, description="Total number of covered bins")
    
    @property
    def coverage_gap(self) -> float:
        """Calculate the coverage gap (percentage to reach 100%)."""
        return 100.0 - self.overall_coverage
    
    @property
    def uncovered_count(self) -> int:
        """Get total count of uncovered bins including cross-coverage."""
        cross_uncovered = sum(len(xc.uncovered) for xc in self.cross_coverage)
        return len(self.uncovered_bins) + cross_uncovered
    
    def get_uncovered_by_covergroup(self) -> dict[str, list[UncoveredBinInfo]]:
        """Group uncovered bins by covergroup."""
        result: dict[str, list[UncoveredBinInfo]] = {}
        for ub in self.uncovered_bins:
            if ub.covergroup not in result:
                result[ub.covergroup] = []
            result[ub.covergroup].append(ub)
        return result
    
    def get_coverage_by_category(self) -> dict[str, float]:
        """Get coverage breakdown by covergroup category."""
        return {cg.name: cg.coverage for cg in self.covergroups}


class CoverageParser:
    """
    Parser for functional coverage reports.
    
    This parser handles the standard coverage report format used in chip
    verification, extracting all relevant information into structured
    Pydantic models for downstream processing.
    
    Features:
    - Robust regex-based parsing
    - Handles multiple covergroups and coverpoints
    - Extracts cross-coverage information
    - Provides both raw and structured output formats
    - Error handling for malformed reports
    
    Usage:
        parser = CoverageParser()
        report = parser.parse(coverage_text)
        print(report.model_dump_json(indent=2))
    """
    
    # Regex patterns for parsing
    DESIGN_PATTERN = re.compile(r"Design:\s*(\w+)")
    DATE_PATTERN = re.compile(r"Date:\s*([\d-]+)")
    OVERALL_COVERAGE_PATTERN = re.compile(r"Overall Coverage:\s*([\d.]+)%")
    
    COVERGROUP_PATTERN = re.compile(
        r"Covergroup:\s*(\w+)\s*\n\s*Coverage:\s*([\d.]+)%\s*\((\d+)/(\d+)\s*bins\)"
    )
    COVERPOINT_PATTERN = re.compile(r"Coverpoint:\s*(\w+)")
    
    # Bin patterns - handles various formats
    BIN_PATTERN = re.compile(
        r"bin\s+(\w+)(\[[\d:]+\])?\s+hits:\s*(\d+)\s+(covered|UNCOVERED)"
    )
    
    # Cross coverage patterns
    CROSS_HEADER_PATTERN = re.compile(
        r"Cross Coverage:\s*(\w+)\s*\n\s*Coverage:\s*([\d.]+)%\s*\((\d+)/(\d+)\s*bins\)"
    )
    CROSS_BIN_PATTERN = re.compile(
        r"<([^>]+)>\s+hits:\s*(\d+)\s+(covered|UNCOVERED)"
    )
    
    def __init__(self):
        """Initialize the parser."""
        self._current_covergroup: Optional[str] = None
        self._current_coverpoint: Optional[str] = None
    
    def parse(self, text: str) -> CoverageReport:
        """
        Parse a coverage report text into a structured CoverageReport object.
        
        Args:
            text: The raw coverage report text
            
        Returns:
            CoverageReport object with all extracted information
            
        Note:
            If a coverpoint block is incomplete or malformed, the parser
            records a warning and continues parsing remaining sections.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Extract basic metadata with graceful defaults
        try:
            design = self._extract_design(text)
        except ValueError:
            logger.warning("Could not extract design name, using 'unknown'")
            design = "unknown"
        
        date = self._extract_date(text)
        
        try:
            overall_coverage = self._extract_overall_coverage(text)
        except ValueError:
            logger.warning("Could not extract overall coverage, defaulting to 0.0")
            overall_coverage = 0.0
        
        # Parse covergroups with error handling for individual groups
        covergroups = []
        try:
            covergroups = self._parse_covergroups(text)
        except Exception as e:
            logger.warning(f"Error parsing covergroups: {e}. Continuing with partial data.")
        
        # Parse cross-coverage with error handling
        cross_coverage = []
        try:
            cross_coverage = self._parse_cross_coverage(text)
        except Exception as e:
            logger.warning(f"Error parsing cross-coverage: {e}. Continuing with partial data.")
        
        # Build uncovered bins list
        uncovered_bins = self._build_uncovered_list(covergroups)
        
        # Calculate totals
        total_bins = sum(cg.total_bins for cg in covergroups)
        covered_bins = sum(cg.covered_bins for cg in covergroups)
        
        # Add cross-coverage bins to totals
        for xc in cross_coverage:
            total_bins += xc.total_bins
            covered_bins += xc.covered_bins
        
        return CoverageReport(
            design=design,
            date=date,
            overall_coverage=overall_coverage,
            covergroups=covergroups,
            uncovered_bins=uncovered_bins,
            cross_coverage=cross_coverage,
            total_bins=total_bins,
            covered_bins=covered_bins
        )
    
    def parse_file(self, filepath: str) -> CoverageReport:
        """
        Parse a coverage report from a file.
        
        Args:
            filepath: Path to the coverage report file
            
        Returns:
            CoverageReport object
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.parse(text)
    
    def _extract_design(self, text: str) -> str:
        """Extract design name from report."""
        match = self.DESIGN_PATTERN.search(text)
        if match:
            return match.group(1)
        raise ValueError("Could not find design name in coverage report")
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from report."""
        match = self.DATE_PATTERN.search(text)
        return match.group(1) if match else None
    
    def _extract_overall_coverage(self, text: str) -> float:
        """Extract overall coverage percentage."""
        match = self.OVERALL_COVERAGE_PATTERN.search(text)
        if match:
            return float(match.group(1))
        raise ValueError("Could not find overall coverage in report")
    
    def _parse_covergroups(self, text: str) -> list[CovergroupInfo]:
        """Parse all covergroups from the report."""
        covergroups: list[CovergroupInfo] = []
        
        # Split text into sections by separator
        sections = re.split(r'-{40,}', text)
        
        for section in sections:
            # Check if this section contains a covergroup
            cg_match = self.COVERGROUP_PATTERN.search(section)
            if cg_match:
                cg_name = cg_match.group(1)
                cg_coverage = float(cg_match.group(2))
                cg_covered = int(cg_match.group(3))
                cg_total = int(cg_match.group(4))
                
                # Parse coverpoints within this covergroup
                coverpoints = self._parse_coverpoints(section, cg_name)
                
                covergroup = CovergroupInfo(
                    name=cg_name,
                    coverage=cg_coverage,
                    covered_bins=cg_covered,
                    total_bins=cg_total,
                    coverpoints=coverpoints
                )
                covergroups.append(covergroup)
        
        return covergroups
    
    def _parse_coverpoints(self, section: str, covergroup_name: str) -> list[CoverpointInfo]:
        """Parse coverpoints within a covergroup section."""
        coverpoints: list[CoverpointInfo] = []
        
        # Find all coverpoint declarations
        cp_matches = list(self.COVERPOINT_PATTERN.finditer(section))
        
        for i, cp_match in enumerate(cp_matches):
            cp_name = cp_match.group(1)
            
            # Get the text between this coverpoint and the next (or end of section)
            start = cp_match.end()
            end = cp_matches[i + 1].start() if i + 1 < len(cp_matches) else len(section)
            cp_text = section[start:end]
            
            # Parse bins within this coverpoint
            bins = self._parse_bins(cp_text)
            
            coverpoint = CoverpointInfo(
                name=cp_name,
                bins=bins
            )
            coverpoints.append(coverpoint)
        
        return coverpoints
    
    def _parse_bins(self, text: str) -> list[BinInfo]:
        """Parse bins from a coverpoint section."""
        bins: list[BinInfo] = []
        
        for match in self.BIN_PATTERN.finditer(text):
            bin_name = match.group(1)
            bin_range = match.group(2)  # May be None
            hits = int(match.group(3))
            is_covered = match.group(4).lower() == "covered"
            
            bin_info = BinInfo(
                name=bin_name,
                range=bin_range,
                hits=hits,
                covered=is_covered
            )
            bins.append(bin_info)
        
        return bins
    
    def _parse_cross_coverage(self, text: str) -> list[CrossCoverageInfo]:
        """Parse cross-coverage sections."""
        cross_coverages: list[CrossCoverageInfo] = []
        
        # Find cross coverage headers
        for header_match in self.CROSS_HEADER_PATTERN.finditer(text):
            xc_name = header_match.group(1)
            xc_coverage = float(header_match.group(2))
            xc_covered = int(header_match.group(3))
            xc_total = int(header_match.group(4))
            
            # Get the section after this header
            start = header_match.end()
            # Find the next section break or end of file
            next_section = re.search(r'={40,}|-{40,}', text[start:])
            end = start + next_section.start() if next_section else len(text)
            xc_text = text[start:end]
            
            # Parse cross bins
            bins: list[CrossCoverageBin] = []
            uncovered: list[str] = []
            
            for bin_match in self.CROSS_BIN_PATTERN.finditer(xc_text):
                values = f"<{bin_match.group(1)}>"
                hits = int(bin_match.group(2))
                is_covered = bin_match.group(3).lower() == "covered"
                
                bins.append(CrossCoverageBin(
                    values=values,
                    hits=hits,
                    covered=is_covered
                ))
                
                if not is_covered:
                    uncovered.append(values)
            
            cross_coverage = CrossCoverageInfo(
                name=xc_name,
                coverage=xc_coverage,
                covered_bins=xc_covered,
                total_bins=xc_total,
                bins=bins,
                uncovered=uncovered
            )
            cross_coverages.append(cross_coverage)
        
        return cross_coverages
    
    def _build_uncovered_list(
        self, 
        covergroups: list[CovergroupInfo]
    ) -> list[UncoveredBinInfo]:
        """Build a flattened list of all uncovered bins."""
        uncovered: list[UncoveredBinInfo] = []
        
        for cg in covergroups:
            for cp in cg.coverpoints:
                for bin_info in cp.bins:
                    if not bin_info.covered:
                        bin_str = bin_info.name
                        if bin_info.range:
                            bin_str += bin_info.range
                        
                        uncovered.append(UncoveredBinInfo(
                            covergroup=cg.name,
                            coverpoint=cp.name,
                            bin=bin_str,
                            hits=bin_info.hits
                        ))
        
        return uncovered
    
    def to_dict(self, report: CoverageReport) -> dict:
        """
        Convert a CoverageReport to a dictionary matching the expected output format.
        
        This method ensures the output matches the exact format specified in the
        assignment requirements.
        """
        return report.model_dump(mode='json')
    
    def to_json(self, report: CoverageReport, indent: int = 2) -> str:
        """Convert a CoverageReport to a JSON string."""
        return report.model_dump_json(indent=indent)


# Convenience function for quick parsing
def parse_coverage_report(text: str) -> CoverageReport:
    """
    Quick function to parse a coverage report.
    
    Args:
        text: Raw coverage report text
        
    Returns:
        Parsed CoverageReport object
    """
    parser = CoverageParser()
    return parser.parse(text)


if __name__ == "__main__":
    # Test with sample input
    sample = """
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
    
    parser = CoverageParser()
    report = parser.parse(sample)
    print(parser.to_json(report))
