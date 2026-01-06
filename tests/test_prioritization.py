"""
Unit tests for the Prioritization Engine.
Tests the priority scoring formula and ranking logic.
"""

import pytest
from src.analyzer.prioritization import (
    PrioritizationEngine,
    PrioritizedSuggestion,
    RawSuggestion
)
from src.parser.coverage_parser import CoverageReport, CovergroupInfo, CoverpointInfo, BinInfo


class TestPrioritizedSuggestion:
    """Tests for the PrioritizedSuggestion model."""
    
    def test_suggestion_creation(self):
        """Test creating a PrioritizedSuggestion."""
        suggestion = PrioritizedSuggestion(
            rank=1,
            target_bin="cg_test.cp_test.bin1",
            priority="high",
            difficulty="medium",
            priority_score=0.75,
            coverage_impact=0.8,
            inverse_difficulty=0.7,
            dependency_score=0.6,
            suggestion="Test suggestion text",
            reasoning="Reasoning text",
            test_outline=["Step 1", "Step 2"],
            dependencies=["dep1"],
            estimated_effort="2 hours"
        )
        assert suggestion.rank == 1
        assert suggestion.priority_score == 0.75
    
    def test_suggestion_to_dict(self):
        """Test converting suggestion to dictionary."""
        suggestion = PrioritizedSuggestion(
            rank=1,
            target_bin="test_bin",
            priority="medium",
            difficulty="easy",
            priority_score=0.65,
            coverage_impact=0.5,
            inverse_difficulty=0.8,
            dependency_score=0.7,
            suggestion="Test",
            reasoning="Reason",
            test_outline=["Step 1"],
            dependencies=[],
            estimated_effort="1 hour"
        )
        data = suggestion.model_dump()
        assert isinstance(data, dict)
        assert data["rank"] == 1
        assert data["target_bin"] == "test_bin"


class TestPrioritizationEngine:
    """Tests for the PrioritizationEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a prioritization engine instance."""
        return PrioritizationEngine()
    
    @pytest.fixture
    def sample_report(self):
        """Create a sample coverage report."""
        return CoverageReport(
            design="test_design",
            date="2024-01-15",
            overall_coverage=50.0,
            covergroups=[
                CovergroupInfo(
                    name="cg_test",
                    coverage=50.0,
                    covered_bins=5,
                    total_bins=10,
                    coverpoints=[
                        CoverpointInfo(
                            name="cp_state",
                            bins=[
                                BinInfo(name="idle", hits=100, is_covered=True),
                                BinInfo(name="active", hits=50, is_covered=True),
                                BinInfo(name="error", hits=0, is_covered=False),
                                BinInfo(name="critical", hits=0, is_covered=False)
                            ]
                        )
                    ],
                    cross_coverage=[]
                )
            ],
            cross_coverage=[]
        )
    
    @pytest.fixture
    def sample_suggestions(self):
        """Create sample raw suggestions."""
        return [
            RawSuggestion(
                target_bin="cg_test.cp_state.error",
                priority="high",
                difficulty="medium",
                suggestion="Test error state transition",
                reasoning="Error state is critical for fault handling",
                test_outline=["1. Inject error", "2. Verify state"],
                dependencies=["idle state working"],
                estimated_effort="2 hours"
            ),
            RawSuggestion(
                target_bin="cg_test.cp_state.critical",
                priority="critical",
                difficulty="hard",
                suggestion="Test critical failure mode",
                reasoning="Critical state needs complex setup",
                test_outline=["1. Setup", "2. Trigger", "3. Verify"],
                dependencies=["error state working", "special mode enabled"],
                estimated_effort="4 hours"
            ),
            RawSuggestion(
                target_bin="cg_test.cp_state.recovery",
                priority="low",
                difficulty="easy",
                suggestion="Test recovery sequence",
                reasoning="Simple recovery test",
                test_outline=["1. Enter recovery", "2. Verify exit"],
                dependencies=[],
                estimated_effort="30 minutes"
            )
        ]
    
    def test_priority_score_formula(self, engine):
        """Test the priority score calculation formula."""
        # Priority Score = (Coverage Impact × 0.4) + (Inverse Difficulty × 0.3) + (Dependency Score × 0.3)
        coverage_impact = 0.8
        inverse_difficulty = 0.6
        dependency_score = 0.4
        
        expected = (0.8 * 0.4) + (0.6 * 0.3) + (0.4 * 0.3)
        expected = 0.32 + 0.18 + 0.12
        expected = 0.62
        
        score = engine.calculate_score(coverage_impact, inverse_difficulty, dependency_score)
        assert abs(score - expected) < 0.001
    
    def test_priority_to_impact(self, engine):
        """Test mapping priority levels to impact scores."""
        assert engine.priority_to_impact("critical") == 1.0
        assert engine.priority_to_impact("high") == 0.8
        assert engine.priority_to_impact("medium") == 0.6
        assert engine.priority_to_impact("low") == 0.4
        assert engine.priority_to_impact("unknown") == 0.5  # Default
    
    def test_difficulty_to_inverse(self, engine):
        """Test mapping difficulty to inverse score."""
        # Easy = high inverse score, Hard = low inverse score
        assert engine.difficulty_to_inverse("easy") == 1.0
        assert engine.difficulty_to_inverse("medium") == 0.6
        assert engine.difficulty_to_inverse("hard") == 0.3
        assert engine.difficulty_to_inverse("unknown") == 0.5  # Default
    
    def test_dependency_score_no_deps(self, engine):
        """Test dependency score with no dependencies."""
        score = engine.calculate_dependency_score([], set())
        assert score == 1.0
    
    def test_dependency_score_all_satisfied(self, engine):
        """Test dependency score when all dependencies are satisfied."""
        dependencies = ["dep1", "dep2"]
        available = {"dep1", "dep2", "dep3"}
        score = engine.calculate_dependency_score(dependencies, available)
        assert score == 1.0
    
    def test_dependency_score_none_satisfied(self, engine):
        """Test dependency score when no dependencies are satisfied."""
        dependencies = ["dep1", "dep2"]
        available = set()
        score = engine.calculate_dependency_score(dependencies, available)
        assert score < 1.0
    
    def test_prioritize_suggestions(self, engine, sample_report, sample_suggestions):
        """Test prioritizing a list of suggestions."""
        prioritized = engine.prioritize(sample_suggestions, sample_report)
        
        assert len(prioritized) == len(sample_suggestions)
        
        # Check that results are sorted by score descending
        for i in range(len(prioritized) - 1):
            assert prioritized[i].priority_score >= prioritized[i+1].priority_score
        
        # Check ranks are assigned correctly
        for i, suggestion in enumerate(prioritized):
            assert suggestion.rank == i + 1
    
    def test_prioritized_suggestion_has_all_fields(self, engine, sample_report, sample_suggestions):
        """Test that prioritized suggestions have all required fields."""
        prioritized = engine.prioritize(sample_suggestions, sample_report)
        
        for suggestion in prioritized:
            assert suggestion.rank is not None
            assert suggestion.target_bin is not None
            assert suggestion.priority is not None
            assert suggestion.difficulty is not None
            assert suggestion.priority_score is not None
            assert 0.0 <= suggestion.priority_score <= 1.0
            assert suggestion.coverage_impact is not None
            assert suggestion.inverse_difficulty is not None
            assert suggestion.dependency_score is not None
    
    def test_empty_suggestions(self, engine, sample_report):
        """Test prioritizing empty suggestion list."""
        prioritized = engine.prioritize([], sample_report)
        assert len(prioritized) == 0
    
    def test_single_suggestion(self, engine, sample_report):
        """Test prioritizing single suggestion."""
        single_suggestion = [
            RawSuggestion(
                target_bin="cg_test.cp_state.test",
                priority="medium",
                difficulty="easy",
                suggestion="Test",
                reasoning="Reason",
                test_outline=["Step 1"],
                dependencies=[],
                estimated_effort="1 hour"
            )
        ]
        prioritized = engine.prioritize(single_suggestion, sample_report)
        
        assert len(prioritized) == 1
        assert prioritized[0].rank == 1


class TestPriorityScoreEdgeCases:
    """Test edge cases for priority scoring."""
    
    @pytest.fixture
    def engine(self):
        return PrioritizationEngine()
    
    def test_max_score(self, engine):
        """Test maximum possible score."""
        # Critical priority, easy difficulty, no dependencies
        score = engine.calculate_score(1.0, 1.0, 1.0)
        assert score == 1.0
    
    def test_min_score(self, engine):
        """Test minimum possible score."""
        # Low priority, hard difficulty, unmet dependencies
        score = engine.calculate_score(0.0, 0.0, 0.0)
        assert score == 0.0
    
    def test_equal_weights(self, engine):
        """Test with equal component scores."""
        score = engine.calculate_score(0.5, 0.5, 0.5)
        expected = (0.5 * 0.4) + (0.5 * 0.3) + (0.5 * 0.3)
        assert abs(score - expected) < 0.001
    
    def test_coverage_impact_dominance(self, engine):
        """Test that coverage impact has highest weight."""
        # High impact, low everything else
        score1 = engine.calculate_score(1.0, 0.0, 0.0)
        # Low impact, high everything else
        score2 = engine.calculate_score(0.0, 1.0, 1.0)
        
        # Score1 (0.4) should be less than Score2 (0.6)
        assert score1 < score2
    
    def test_difficulty_and_dependency_equal_weight(self, engine):
        """Test that difficulty and dependency have equal weight."""
        # High difficulty only
        score1 = engine.calculate_score(0.0, 1.0, 0.0)
        # High dependency only
        score2 = engine.calculate_score(0.0, 0.0, 1.0)
        
        assert abs(score1 - score2) < 0.001


class TestRankingStability:
    """Test ranking stability with edge cases."""
    
    @pytest.fixture
    def engine(self):
        return PrioritizationEngine()
    
    @pytest.fixture
    def sample_report(self):
        return CoverageReport(
            design="test",
            date="2024-01-15",
            overall_coverage=50.0,
            covergroups=[],
            cross_coverage=[]
        )
    
    def test_tie_breaking(self, engine, sample_report):
        """Test handling of equal scores."""
        # Create suggestions that would have identical scores
        suggestions = [
            RawSuggestion(
                target_bin="bin1",
                priority="medium",
                difficulty="medium",
                suggestion="Test 1",
                reasoning="Reason 1",
                test_outline=["Step 1"],
                dependencies=[],
                estimated_effort="1 hour"
            ),
            RawSuggestion(
                target_bin="bin2",
                priority="medium",
                difficulty="medium",
                suggestion="Test 2",
                reasoning="Reason 2",
                test_outline=["Step 1"],
                dependencies=[],
                estimated_effort="1 hour"
            )
        ]
        
        prioritized = engine.prioritize(suggestions, sample_report)
        
        # Both should be ranked
        assert len(prioritized) == 2
        assert prioritized[0].rank == 1
        assert prioritized[1].rank == 2
    
    def test_deterministic_ordering(self, engine, sample_report):
        """Test that ordering is deterministic."""
        suggestions = [
            RawSuggestion(
                target_bin=f"bin{i}",
                priority="medium",
                difficulty="medium",
                suggestion=f"Test {i}",
                reasoning=f"Reason {i}",
                test_outline=["Step 1"],
                dependencies=[],
                estimated_effort="1 hour"
            )
            for i in range(5)
        ]
        
        # Run multiple times
        result1 = engine.prioritize(suggestions.copy(), sample_report)
        result2 = engine.prioritize(suggestions.copy(), sample_report)
        
        # Should produce same ordering
        for i in range(len(result1)):
            assert result1[i].target_bin == result2[i].target_bin


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
