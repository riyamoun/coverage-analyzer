"""
Prioritization Engine Module

Implements the priority scoring system for test suggestions:
Priority Score = (Coverage Impact × 0.4) + (Inverse Difficulty × 0.3) + (Dependency Score × 0.3)

Author: ML Engineer
Date: 2025-01-06
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, computed_field

from ..llm.llm_client import TestSuggestion
from ..parser.coverage_parser import CoverageReport


class DifficultyLevel(str, Enum):
    """Difficulty levels for test implementation."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PriorityLevel(str, Enum):
    """Priority levels for suggestions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PrioritizedSuggestion(BaseModel):
    """
    A test suggestion with computed priority score.
    
    Extends the base TestSuggestion with prioritization metrics.
    """
    # Original suggestion fields
    target_bin: str
    priority: str
    difficulty: str
    suggestion: str
    test_outline: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    reasoning: str
    
    # Prioritization metrics
    coverage_impact: float = Field(
        ..., 
        description="Estimated coverage improvement (0-1)",
        ge=0.0,
        le=1.0
    )
    inverse_difficulty: float = Field(
        ..., 
        description="Inverse difficulty score (higher = easier)",
        ge=0.0,
        le=1.0
    )
    dependency_score: float = Field(
        ..., 
        description="Dependency score (1 = no deps, 0.5 = has deps)",
        ge=0.0,
        le=1.0
    )
    priority_score: float = Field(
        ..., 
        description="Final computed priority score",
        ge=0.0,
        le=1.0
    )
    
    # Additional metadata
    rank: int = Field(0, description="Rank among all suggestions")
    estimated_effort_hours: Optional[float] = Field(
        None, 
        description="Estimated implementation effort in hours"
    )
    related_bins: list[str] = Field(
        default_factory=list,
        description="Other bins that might be covered by the same test"
    )
    
    @classmethod
    def from_suggestion(
        cls,
        suggestion: TestSuggestion,
        coverage_impact: float,
        priority_score: float,
        rank: int = 0
    ) -> "PrioritizedSuggestion":
        """
        Create a prioritized suggestion from a base suggestion.
        
        Args:
            suggestion: The original TestSuggestion
            coverage_impact: Calculated coverage impact
            priority_score: Calculated priority score
            rank: Rank among all suggestions
            
        Returns:
            PrioritizedSuggestion instance
        """
        # Calculate inverse difficulty
        difficulty_map = {"easy": 1.0, "medium": 0.5, "hard": 0.333}
        inverse_difficulty = difficulty_map.get(suggestion.difficulty.lower(), 0.5)
        
        # Calculate dependency score
        dependency_score = 0.5 if suggestion.dependencies else 1.0
        
        # Estimate effort based on difficulty
        effort_map = {"easy": 1.0, "medium": 3.0, "hard": 8.0}
        estimated_effort = effort_map.get(suggestion.difficulty.lower(), 3.0)
        
        return cls(
            target_bin=suggestion.target_bin,
            priority=suggestion.priority,
            difficulty=suggestion.difficulty,
            suggestion=suggestion.suggestion,
            test_outline=suggestion.test_outline,
            dependencies=suggestion.dependencies,
            reasoning=suggestion.reasoning,
            coverage_impact=coverage_impact,
            inverse_difficulty=inverse_difficulty,
            dependency_score=dependency_score,
            priority_score=priority_score,
            rank=rank,
            estimated_effort_hours=estimated_effort
        )


class PrioritizationEngine:
    """
    Engine for prioritizing test suggestions based on multiple factors.
    
    Implements the scoring formula:
    Priority Score = (Coverage Impact × 0.4) + (Inverse Difficulty × 0.3) + (Dependency Score × 0.3)
    
    Features:
    - Coverage impact estimation
    - Difficulty-based scoring
    - Dependency analysis
    - Related bin detection
    - Effort estimation
    
    Usage:
        engine = PrioritizationEngine(report)
        prioritized = engine.prioritize(suggestions)
    """
    
    # Scoring weights (as specified in requirements)
    WEIGHT_COVERAGE_IMPACT = 0.4
    WEIGHT_INVERSE_DIFFICULTY = 0.3
    WEIGHT_DEPENDENCY_SCORE = 0.3
    
    # Difficulty mappings
    DIFFICULTY_SCORES = {
        "easy": 1.0,      # 1/1 = 1.0
        "medium": 0.5,    # 1/2 = 0.5
        "hard": 0.333     # 1/3 ≈ 0.333
    }
    
    # Priority boost factors
    PRIORITY_BOOST = {
        "high": 1.2,
        "medium": 1.0,
        "low": 0.8
    }
    
    def __init__(self, report: CoverageReport):
        """
        Initialize the prioritization engine.
        
        Args:
            report: The parsed coverage report for context
        """
        self.report = report
        self._bin_to_covergroup = self._build_bin_mapping()
    
    def _build_bin_mapping(self) -> dict[str, tuple[str, int]]:
        """Build a mapping from bin names to their covergroup and total bins."""
        mapping = {}
        
        for cg in self.report.covergroups:
            for cp in cg.coverpoints:
                for bin_info in cp.bins:
                    full_path = f"{cg.name}.{cp.name}.{bin_info.name}"
                    mapping[full_path] = (cg.name, cg.total_bins)
        
        return mapping
    
    def calculate_coverage_impact(self, target_bin: str) -> float:
        """
        Calculate the coverage impact of hitting a specific bin.
        
        The impact is the percentage increase in overall coverage if this
        bin were to be covered.
        
        Args:
            target_bin: Full path to the target bin
            
        Returns:
            Coverage impact as a value between 0 and 1
        """
        total_bins = self.report.total_bins
        
        if total_bins == 0:
            return 0.0
        
        # Each bin contributes equally to coverage
        # Impact = 1 / total_bins, normalized to 0-1 scale
        base_impact = 1.0 / total_bins
        
        # Boost impact for bins in covergroups with lower coverage
        # (hitting them has more relative impact)
        covergroup_name = target_bin.split('.')[0] if '.' in target_bin else target_bin
        
        for cg in self.report.covergroups:
            if cg.name == covergroup_name:
                # Lower coverage = higher boost
                coverage_boost = (100 - cg.coverage) / 100
                base_impact *= (1 + coverage_boost)
                break
        
        # Normalize to 0-1 range
        return min(base_impact * 10, 1.0)  # Scale up for visibility
    
    def calculate_inverse_difficulty(self, difficulty: str) -> float:
        """
        Calculate the inverse difficulty score.
        
        Args:
            difficulty: Difficulty level ("easy", "medium", "hard")
            
        Returns:
            Inverse difficulty score (higher = easier)
        """
        return self.DIFFICULTY_SCORES.get(difficulty.lower(), 0.5)
    
    def calculate_dependency_score(self, dependencies: list[str]) -> float:
        """
        Calculate the dependency score.
        
        Args:
            dependencies: List of dependencies
            
        Returns:
            1.0 if no dependencies, 0.5 if dependencies exist
        """
        if not dependencies:
            return 1.0
        
        # Could be extended to analyze dependency complexity
        num_deps = len(dependencies)
        
        if num_deps == 0:
            return 1.0
        elif num_deps <= 2:
            return 0.7
        else:
            return 0.5
    
    def calculate_priority_score(
        self,
        coverage_impact: float,
        inverse_difficulty: float,
        dependency_score: float,
        priority_level: str = "medium"
    ) -> float:
        """
        Calculate the final priority score using the formula:
        Score = (Coverage Impact × 0.4) + (Inverse Difficulty × 0.3) + (Dependency Score × 0.3)
        
        Args:
            coverage_impact: Coverage impact (0-1)
            inverse_difficulty: Inverse difficulty (0-1)
            dependency_score: Dependency score (0-1)
            priority_level: Original priority from LLM
            
        Returns:
            Final priority score (0-1)
        """
        base_score = (
            coverage_impact * self.WEIGHT_COVERAGE_IMPACT +
            inverse_difficulty * self.WEIGHT_INVERSE_DIFFICULTY +
            dependency_score * self.WEIGHT_DEPENDENCY_SCORE
        )
        
        # Apply priority boost from LLM's assessment
        boost = self.PRIORITY_BOOST.get(priority_level.lower(), 1.0)
        
        return min(base_score * boost, 1.0)
    
    def find_related_bins(self, target_bin: str) -> list[str]:
        """
        Find bins that might be covered by the same test.
        
        Args:
            target_bin: Full path to the target bin
            
        Returns:
            List of related bin paths
        """
        related = []
        parts = target_bin.split('.')
        
        if len(parts) < 2:
            return related
        
        covergroup_name = parts[0]
        
        # Find other uncovered bins in the same covergroup
        for ub in self.report.uncovered_bins:
            if ub.covergroup == covergroup_name and ub.full_path != target_bin:
                related.append(ub.full_path)
        
        return related[:5]  # Limit to 5 related bins
    
    def prioritize(
        self,
        suggestions: list[TestSuggestion]
    ) -> list[PrioritizedSuggestion]:
        """
        Prioritize a list of test suggestions.
        
        Args:
            suggestions: List of TestSuggestion objects
            
        Returns:
            Sorted list of PrioritizedSuggestion objects
        """
        prioritized = []
        
        for suggestion in suggestions:
            # Calculate all scores
            coverage_impact = self.calculate_coverage_impact(suggestion.target_bin)
            inverse_difficulty = self.calculate_inverse_difficulty(suggestion.difficulty)
            dependency_score = self.calculate_dependency_score(suggestion.dependencies)
            
            priority_score = self.calculate_priority_score(
                coverage_impact=coverage_impact,
                inverse_difficulty=inverse_difficulty,
                dependency_score=dependency_score,
                priority_level=suggestion.priority
            )
            
            # Create prioritized suggestion
            prioritized_suggestion = PrioritizedSuggestion.from_suggestion(
                suggestion=suggestion,
                coverage_impact=coverage_impact,
                priority_score=priority_score
            )
            
            # Add related bins
            prioritized_suggestion.related_bins = self.find_related_bins(suggestion.target_bin)
            
            prioritized.append(prioritized_suggestion)
        
        # Sort by priority score (descending)
        prioritized.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Assign ranks
        for i, p in enumerate(prioritized, 1):
            p.rank = i
        
        return prioritized
    
    def get_quick_wins(
        self,
        prioritized: list[PrioritizedSuggestion],
        max_effort_hours: float = 2.0
    ) -> list[PrioritizedSuggestion]:
        """
        Get suggestions that are quick wins (high impact, low effort).
        
        Args:
            prioritized: List of prioritized suggestions
            max_effort_hours: Maximum effort threshold
            
        Returns:
            Filtered list of quick win suggestions
        """
        return [
            p for p in prioritized
            if (p.estimated_effort_hours or 0) <= max_effort_hours
            and p.difficulty.lower() in ["easy", "medium"]
        ]
    
    def get_high_impact(
        self,
        prioritized: list[PrioritizedSuggestion],
        min_impact: float = 0.3
    ) -> list[PrioritizedSuggestion]:
        """
        Get high-impact suggestions regardless of difficulty.
        
        Args:
            prioritized: List of prioritized suggestions
            min_impact: Minimum coverage impact threshold
            
        Returns:
            Filtered list of high-impact suggestions
        """
        return [
            p for p in prioritized
            if p.coverage_impact >= min_impact
        ]
    
    def generate_summary(
        self,
        prioritized: list[PrioritizedSuggestion]
    ) -> dict:
        """
        Generate a summary of the prioritization results.
        
        Args:
            prioritized: List of prioritized suggestions
            
        Returns:
            Summary dictionary with statistics
        """
        if not prioritized:
            return {
                "total_suggestions": 0,
                "by_difficulty": {},
                "by_priority": {},
                "average_score": 0,
                "estimated_total_effort": 0
            }
        
        # Count by difficulty
        by_difficulty = {"easy": 0, "medium": 0, "hard": 0}
        for p in prioritized:
            diff = p.difficulty.lower()
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
        
        # Count by priority
        by_priority = {"high": 0, "medium": 0, "low": 0}
        for p in prioritized:
            pri = p.priority.lower()
            by_priority[pri] = by_priority.get(pri, 0) + 1
        
        # Calculate averages
        avg_score = sum(p.priority_score for p in prioritized) / len(prioritized)
        total_effort = sum(p.estimated_effort_hours or 0 for p in prioritized)
        
        return {
            "total_suggestions": len(prioritized),
            "by_difficulty": by_difficulty,
            "by_priority": by_priority,
            "average_score": round(avg_score, 3),
            "estimated_total_effort_hours": round(total_effort, 1),
            "quick_wins": len(self.get_quick_wins(prioritized)),
            "top_3_bins": [p.target_bin for p in prioritized[:3]]
        }
