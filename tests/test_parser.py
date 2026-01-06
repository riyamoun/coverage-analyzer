"""
Unit tests for the Coverage Parser module.
Tests cover parsing of covergroups, coverpoints, bins, and cross-coverage.
"""

import pytest
from src.parser.coverage_parser import (
    CoverageParser,
    CoverageReport,
    CovergroupInfo,
    CoverpointInfo,
    BinInfo,
    CrossCoverageInfo
)


class TestBinInfo:
    """Tests for the BinInfo model."""
    
    def test_bin_info_creation(self):
        """Test creating a BinInfo object."""
        bin_info = BinInfo(
            name="test_bin",
            range="[0:15]",
            hits=100,
            is_covered=True
        )
        assert bin_info.name == "test_bin"
        assert bin_info.range == "[0:15]"
        assert bin_info.hits == 100
        assert bin_info.is_covered is True
    
    def test_bin_info_uncovered(self):
        """Test uncovered bin."""
        bin_info = BinInfo(
            name="uncovered_bin",
            range=None,
            hits=0,
            is_covered=False
        )
        assert bin_info.is_covered is False
        assert bin_info.hits == 0


class TestCoverpointInfo:
    """Tests for the CoverpointInfo model."""
    
    def test_coverpoint_creation(self):
        """Test creating a CoverpointInfo object."""
        bins = [
            BinInfo(name="bin1", hits=10, is_covered=True),
            BinInfo(name="bin2", hits=0, is_covered=False)
        ]
        coverpoint = CoverpointInfo(
            name="cp_test",
            bins=bins
        )
        assert coverpoint.name == "cp_test"
        assert len(coverpoint.bins) == 2
        assert coverpoint.covered_bins == 1
        assert coverpoint.uncovered_bins == 1
        assert coverpoint.coverage_percentage == 50.0
    
    def test_coverpoint_all_covered(self):
        """Test coverpoint with all bins covered."""
        bins = [
            BinInfo(name="bin1", hits=10, is_covered=True),
            BinInfo(name="bin2", hits=5, is_covered=True),
            BinInfo(name="bin3", hits=20, is_covered=True)
        ]
        coverpoint = CoverpointInfo(name="cp_full", bins=bins)
        assert coverpoint.coverage_percentage == 100.0
        assert coverpoint.uncovered_bins == 0


class TestCovergroupInfo:
    """Tests for the CovergroupInfo model."""
    
    def test_covergroup_creation(self):
        """Test creating a CovergroupInfo object."""
        coverpoints = [
            CoverpointInfo(
                name="cp1",
                bins=[BinInfo(name="b1", hits=10, is_covered=True)]
            )
        ]
        covergroup = CovergroupInfo(
            name="cg_test",
            coverage=85.5,
            covered_bins=17,
            total_bins=20,
            coverpoints=coverpoints
        )
        assert covergroup.name == "cg_test"
        assert covergroup.coverage == 85.5
        assert covergroup.covered_bins == 17
        assert covergroup.total_bins == 20


class TestCrossCoverageInfo:
    """Tests for the CrossCoverageInfo model."""
    
    def test_cross_coverage_creation(self):
        """Test creating a CrossCoverageInfo object."""
        bins = [
            BinInfo(name="<read, low>", hits=50, is_covered=True),
            BinInfo(name="<write, high>", hits=0, is_covered=False)
        ]
        cross = CrossCoverageInfo(
            name="cross_op_x_addr",
            sources=["cp_operation", "cp_address"],
            coverage=50.0,
            covered_bins=1,
            total_bins=2,
            bins=bins
        )
        assert cross.name == "cross_op_x_addr"
        assert len(cross.sources) == 2
        assert cross.coverage == 50.0


class TestCoverageParser:
    """Tests for the CoverageParser class."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return CoverageParser()
    
    @pytest.fixture
    def sample_report(self):
        """Sample coverage report text."""
        return """
=======================================================
Functional Coverage Report
Design: test_controller
Date: 2024-01-15
=======================================================

Covergroup: cg_basic
  Coverage: 75.00% (3/4 bins)
  
  Coverpoint: cp_state
    bin idle                      hits: 1000    covered
    bin active                    hits: 500     covered
    bin error                     hits: 0       UNCOVERED
    bin complete                  hits: 250     covered

-------------------------------------------------------

Covergroup: cg_transfers
  Coverage: 50.00% (4/8 bins)
  
  Coverpoint: cp_size
    bin small[0:255]              hits: 100     covered
    bin medium[256:1023]          hits: 50      covered
    bin large[1024:4095]          hits: 0       UNCOVERED
    bin max[4096]                 hits: 0       UNCOVERED

  Coverpoint: cp_direction
    bin read                      hits: 200     covered
    bin write                     hits: 150     covered
    bin read_write                hits: 0       UNCOVERED
    bin idle                      hits: 0       UNCOVERED

-------------------------------------------------------
Cross Coverage: cross_size_x_dir
  Coverage: 25.00% (2/8 bins)
  
  <small, read>                   hits: 50      covered
  <small, write>                  hits: 30      covered
  <medium, read>                  hits: 0       UNCOVERED
  <medium, write>                 hits: 0       UNCOVERED
  <large, read>                   hits: 0       UNCOVERED
  <large, write>                  hits: 0       UNCOVERED
  <max, read>                     hits: 0       UNCOVERED
  <max, write>                    hits: 0       UNCOVERED

=======================================================
Overall Coverage: 45.00%
=======================================================
"""
    
    def test_parse_design_name(self, parser, sample_report):
        """Test parsing the design name."""
        report = parser.parse(sample_report)
        assert report.design == "test_controller"
    
    def test_parse_date(self, parser, sample_report):
        """Test parsing the date."""
        report = parser.parse(sample_report)
        assert report.date == "2024-01-15"
    
    def test_parse_overall_coverage(self, parser, sample_report):
        """Test parsing overall coverage percentage."""
        report = parser.parse(sample_report)
        assert report.overall_coverage == 45.0
    
    def test_parse_covergroups(self, parser, sample_report):
        """Test parsing covergroups."""
        report = parser.parse(sample_report)
        assert len(report.covergroups) == 2
        
        # Check first covergroup
        cg1 = report.covergroups[0]
        assert cg1.name == "cg_basic"
        assert cg1.coverage == 75.0
        assert cg1.covered_bins == 3
        assert cg1.total_bins == 4
    
    def test_parse_coverpoints(self, parser, sample_report):
        """Test parsing coverpoints within covergroups."""
        report = parser.parse(sample_report)
        
        # Check cg_basic coverpoints
        cg_basic = report.covergroups[0]
        assert len(cg_basic.coverpoints) == 1
        assert cg_basic.coverpoints[0].name == "cp_state"
        
        # Check cg_transfers coverpoints
        cg_transfers = report.covergroups[1]
        assert len(cg_transfers.coverpoints) == 2
    
    def test_parse_bins(self, parser, sample_report):
        """Test parsing bins within coverpoints."""
        report = parser.parse(sample_report)
        
        cp_state = report.covergroups[0].coverpoints[0]
        assert len(cp_state.bins) == 4
        
        # Check covered bin
        idle_bin = cp_state.bins[0]
        assert idle_bin.name == "idle"
        assert idle_bin.hits == 1000
        assert idle_bin.is_covered is True
        
        # Check uncovered bin
        error_bin = cp_state.bins[2]
        assert error_bin.name == "error"
        assert error_bin.hits == 0
        assert error_bin.is_covered is False
    
    def test_parse_bins_with_ranges(self, parser, sample_report):
        """Test parsing bins with ranges."""
        report = parser.parse(sample_report)
        
        cp_size = report.covergroups[1].coverpoints[0]
        
        # Check bin with range
        small_bin = cp_size.bins[0]
        assert small_bin.name == "small"
        assert small_bin.range == "[0:255]"
    
    def test_parse_cross_coverage(self, parser, sample_report):
        """Test parsing cross coverage."""
        report = parser.parse(sample_report)
        
        assert len(report.cross_coverage) == 1
        cross = report.cross_coverage[0]
        assert cross.name == "cross_size_x_dir"
        assert cross.coverage == 25.0
        assert cross.covered_bins == 2
        assert cross.total_bins == 8
    
    def test_parse_cross_bins(self, parser, sample_report):
        """Test parsing cross coverage bins."""
        report = parser.parse(sample_report)
        
        cross = report.cross_coverage[0]
        assert len(cross.bins) == 8
        
        # Check covered cross bin
        covered_bin = cross.bins[0]
        assert covered_bin.name == "<small, read>"
        assert covered_bin.hits == 50
        assert covered_bin.is_covered is True
    
    def test_uncovered_bins_list(self, parser, sample_report):
        """Test the uncovered bins helper property."""
        report = parser.parse(sample_report)
        uncovered = report.uncovered_bins
        
        # Should have uncovered bins from coverpoints and cross coverage
        assert len(uncovered) > 0
        
        # All should be uncovered
        for bin_info in uncovered:
            assert bin_info.is_covered is False
    
    def test_covered_count(self, parser, sample_report):
        """Test the covered count property."""
        report = parser.parse(sample_report)
        assert report.covered_count > 0
    
    def test_uncovered_count(self, parser, sample_report):
        """Test the uncovered count property."""
        report = parser.parse(sample_report)
        assert report.uncovered_count > 0
    
    def test_parse_empty_report(self, parser):
        """Test parsing an empty report."""
        report = parser.parse("")
        assert report.design == ""
        assert report.overall_coverage == 0.0
        assert len(report.covergroups) == 0
    
    def test_parse_minimal_report(self, parser):
        """Test parsing a minimal report."""
        minimal_report = """
Functional Coverage Report
Design: minimal
Overall Coverage: 0.00%
"""
        report = parser.parse(minimal_report)
        assert report.design == "minimal"
        assert report.overall_coverage == 0.0
    
    def test_to_dict(self, parser, sample_report):
        """Test converting report to dictionary."""
        report = parser.parse(sample_report)
        data = report.model_dump()
        
        assert isinstance(data, dict)
        assert "design" in data
        assert "overall_coverage" in data
        assert "covergroups" in data
    
    def test_to_json(self, parser, sample_report):
        """Test converting report to JSON string."""
        report = parser.parse(sample_report)
        json_str = report.model_dump_json(indent=2)
        
        assert isinstance(json_str, str)
        assert "test_controller" in json_str


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def parser(self):
        return CoverageParser()
    
    def test_report_with_special_characters(self, parser):
        """Test parsing report with special characters in names."""
        report_text = """
Functional Coverage Report
Design: dma_controller_v2.0
Date: 2024-01-15

Covergroup: cg_special_test_123
  Coverage: 50.00% (1/2 bins)
  
  Coverpoint: cp_mode_v2
    bin mode_a_1                  hits: 100     covered
    bin mode_b_2                  hits: 0       UNCOVERED

Overall Coverage: 50.00%
"""
        report = parser.parse(report_text)
        assert report.design == "dma_controller_v2.0"
        assert report.covergroups[0].name == "cg_special_test_123"
    
    def test_report_with_large_hit_counts(self, parser):
        """Test parsing report with large hit counts."""
        report_text = """
Functional Coverage Report
Design: stress_test

Covergroup: cg_stress
  Coverage: 100.00% (2/2 bins)
  
  Coverpoint: cp_hits
    bin high_hits                 hits: 1000000  covered
    bin very_high                 hits: 999999999 covered

Overall Coverage: 100.00%
"""
        report = parser.parse(report_text)
        assert report.covergroups[0].coverpoints[0].bins[0].hits == 1000000
    
    def test_report_with_percentage_edge_cases(self, parser):
        """Test parsing report with edge case percentages."""
        report_text = """
Functional Coverage Report
Design: edge_case

Covergroup: cg_zero
  Coverage: 0.00% (0/10 bins)
  
Covergroup: cg_full
  Coverage: 100.00% (5/5 bins)

Overall Coverage: 33.33%
"""
        report = parser.parse(report_text)
        assert report.covergroups[0].coverage == 0.0
        assert report.covergroups[1].coverage == 100.0
        assert report.overall_coverage == 33.33


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
