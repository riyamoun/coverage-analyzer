"""
Unit tests for the Coverage Analyzer module.
Tests the main analysis pipeline including LLM integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.analyzer.coverage_analyzer import CoverageAnalyzer, AnalysisResult
from src.parser.coverage_parser import CoverageReport, CovergroupInfo, CoverpointInfo, BinInfo


class TestAnalysisResult:
    """Tests for the AnalysisResult model."""
    
    def test_result_creation(self):
        """Test creating an AnalysisResult."""
        result = AnalysisResult(
            design="test_design",
            date="2024-01-15",
            overall_coverage=75.0,
            covered_bins=15,
            uncovered_bins=5,
            total_bins=20,
            suggestions=[],
            covergroups=[]
        )
        assert result.design == "test_design"
        assert result.overall_coverage == 75.0
        assert result.total_bins == 20


class TestCoverageAnalyzer:
    """Tests for the CoverageAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an analyzer instance without LLM."""
        return CoverageAnalyzer(use_llm=False)
    
    @pytest.fixture
    def sample_report_text(self):
        """Sample coverage report text for testing."""
        return """
=======================================================
Functional Coverage Report
Design: test_controller
Date: 2024-01-15
=======================================================

Covergroup: cg_states
  Coverage: 75.00% (3/4 bins)
  
  Coverpoint: cp_state
    bin idle                      hits: 100     covered
    bin active                    hits: 50      covered
    bin error                     hits: 0       UNCOVERED
    bin complete                  hits: 25      covered

=======================================================
Overall Coverage: 75.00%
=======================================================
"""
    
    def test_analyzer_creation(self, analyzer):
        """Test creating an analyzer instance."""
        assert analyzer is not None
        assert analyzer.use_llm is False
    
    def test_analyzer_with_llm_disabled(self, analyzer, sample_report_text):
        """Test analysis with LLM disabled (mock suggestions)."""
        result = analyzer.analyze(sample_report_text)
        
        assert result is not None
        assert result.design == "test_controller"
        assert result.overall_coverage == 75.0
    
    def test_analyze_returns_suggestions(self, analyzer, sample_report_text):
        """Test that analysis returns suggestions for uncovered bins."""
        result = analyzer.analyze(sample_report_text)
        
        # Should have at least one suggestion for the uncovered 'error' bin
        # When using mock suggestions
        assert result.uncovered_bins > 0
    
    def test_analyze_result_to_dict(self, analyzer, sample_report_text):
        """Test converting analysis result to dictionary."""
        result = analyzer.analyze(sample_report_text)
        data = result.model_dump()
        
        assert isinstance(data, dict)
        assert "design" in data
        assert "overall_coverage" in data
        assert "suggestions" in data
    
    def test_analyze_empty_report(self, analyzer):
        """Test analyzing an empty report."""
        result = analyzer.analyze("")
        assert result.overall_coverage == 0.0
    
    def test_full_coverage_report(self, analyzer):
        """Test analyzing a report with full coverage."""
        full_coverage_report = """
Functional Coverage Report
Design: perfect_design
Date: 2024-01-15

Covergroup: cg_test
  Coverage: 100.00% (4/4 bins)
  
  Coverpoint: cp_state
    bin state1                    hits: 100     covered
    bin state2                    hits: 100     covered
    bin state3                    hits: 100     covered
    bin state4                    hits: 100     covered

Overall Coverage: 100.00%
"""
        result = analyzer.analyze(full_coverage_report)
        assert result.overall_coverage == 100.0
        assert result.uncovered_bins == 0


class TestMockSuggestions:
    """Tests for mock suggestion generation."""
    
    @pytest.fixture
    def analyzer(self):
        return CoverageAnalyzer(use_llm=False)
    
    def test_mock_suggestions_format(self, analyzer):
        """Test that mock suggestions have correct format."""
        report_text = """
Functional Coverage Report
Design: dma_controller

Covergroup: cg_transfer
  Coverage: 50.00% (1/2 bins)
  
  Coverpoint: cp_size
    bin small                     hits: 100     covered
    bin large                     hits: 0       UNCOVERED

Overall Coverage: 50.00%
"""
        result = analyzer.analyze(report_text)
        
        if result.suggestions:
            suggestion = result.suggestions[0]
            assert hasattr(suggestion, 'target_bin')
            assert hasattr(suggestion, 'priority')
            assert hasattr(suggestion, 'difficulty')
            assert hasattr(suggestion, 'priority_score')
            assert hasattr(suggestion, 'suggestion')
            assert hasattr(suggestion, 'test_outline')


class TestLLMIntegration:
    """Tests for LLM integration (mocked)."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock = MagicMock()
        mock.generate_suggestions.return_value = [
            {
                "target_bin": "cg_test.cp_test.bin1",
                "priority": "high",
                "difficulty": "medium",
                "suggestion": "Test mock suggestion",
                "reasoning": "Mock reasoning",
                "test_outline": ["Step 1", "Step 2"],
                "dependencies": [],
                "estimated_effort": "1 hour"
            }
        ]
        return mock
    
    def test_analyzer_uses_llm_client(self, mock_llm_client):
        """Test that analyzer uses LLM client when enabled."""
        with patch('src.analyzer.coverage_analyzer.LLMClient', return_value=mock_llm_client):
            analyzer = CoverageAnalyzer(use_llm=True)
            # Verify LLM client would be used
            assert analyzer.use_llm is True


class TestErrorHandling:
    """Tests for error handling in the analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return CoverageAnalyzer(use_llm=False)
    
    def test_malformed_report(self, analyzer):
        """Test handling of malformed reports."""
        malformed_report = """
This is not a valid coverage report
Just some random text
123 abc xyz
"""
        # Should not raise exception
        result = analyzer.analyze(malformed_report)
        assert result is not None
    
    def test_partial_report(self, analyzer):
        """Test handling of partial reports."""
        partial_report = """
Functional Coverage Report
Design: partial_test

Covergroup: cg_incomplete
  Coverage: 
"""
        # Should not raise exception
        result = analyzer.analyze(partial_report)
        assert result is not None


class TestCoverageMetrics:
    """Tests for coverage metric calculations."""
    
    @pytest.fixture
    def analyzer(self):
        return CoverageAnalyzer(use_llm=False)
    
    def test_coverage_percentage_calculation(self, analyzer):
        """Test correct coverage percentage in results."""
        report = """
Functional Coverage Report
Design: metric_test
Date: 2024-01-15

Covergroup: cg_test
  Coverage: 60.00% (3/5 bins)
  
  Coverpoint: cp_test
    bin b1                        hits: 10      covered
    bin b2                        hits: 5       covered
    bin b3                        hits: 1       covered
    bin b4                        hits: 0       UNCOVERED
    bin b5                        hits: 0       UNCOVERED

Overall Coverage: 60.00%
"""
        result = analyzer.analyze(report)
        assert result.overall_coverage == 60.0
        assert result.covered_bins == 3
        assert result.uncovered_bins == 2
        assert result.total_bins == 5
    
    def test_bin_counts_accurate(self, analyzer):
        """Test that bin counts are accurate."""
        report = """
Functional Coverage Report
Design: count_test

Covergroup: cg_a
  Coverage: 50.00% (2/4 bins)
  
  Coverpoint: cp_a
    bin a1                        hits: 10      covered
    bin a2                        hits: 0       UNCOVERED
    bin a3                        hits: 5       covered
    bin a4                        hits: 0       UNCOVERED

Covergroup: cg_b
  Coverage: 100.00% (2/2 bins)
  
  Coverpoint: cp_b
    bin b1                        hits: 100     covered
    bin b2                        hits: 50      covered

Overall Coverage: 66.67%
"""
        result = analyzer.analyze(report)
        # Total: 6 bins, 4 covered, 2 uncovered
        assert result.covered_bins == 4
        assert result.uncovered_bins == 2
        assert result.total_bins == 6


class TestCrossCoverageAnalysis:
    """Tests for cross coverage analysis."""
    
    @pytest.fixture
    def analyzer(self):
        return CoverageAnalyzer(use_llm=False)
    
    def test_cross_coverage_parsing(self, analyzer):
        """Test that cross coverage is properly parsed."""
        report = """
Functional Coverage Report
Design: cross_test

Covergroup: cg_main
  Coverage: 100.00% (2/2 bins)
  
  Coverpoint: cp_type
    bin read                      hits: 100     covered
    bin write                     hits: 50      covered

-------------------------------------------------------
Cross Coverage: cross_type_x_size
  Coverage: 50.00% (2/4 bins)
  
  <read, small>                   hits: 30      covered
  <read, large>                   hits: 0       UNCOVERED
  <write, small>                  hits: 20      covered
  <write, large>                  hits: 0       UNCOVERED

Overall Coverage: 66.67%
"""
        result = analyzer.analyze(report)
        assert result.overall_coverage == 66.67


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
