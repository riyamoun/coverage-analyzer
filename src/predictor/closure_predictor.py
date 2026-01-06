"""
Coverage Closure Prediction Module (BONUS FEATURE)

Provides predictive analytics for coverage closure including:
- Estimated time to closure
- Closure probability assessment
- Blocking bins identification
- Coverage velocity analysis

Author: ML Engineer
Date: 2025-01-06
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from ..parser.coverage_parser import CoverageReport, UncoveredBinInfo
from ..analyzer.prioritization import PrioritizedSuggestion


class BlockingReason(str, Enum):
    """Reasons why a bin might be blocking (impossible to cover)."""
    HARDWARE_MISSING = "hardware_not_in_testbench"
    DEPENDENCY_CHAIN = "complex_dependency_chain"
    TIMING_CONSTRAINT = "timing_constraint_impossible"
    CONFIG_IMPOSSIBLE = "configuration_impossible"
    CROSS_COVERAGE_EXPLOSION = "cross_coverage_dimension_explosion"
    UNKNOWN = "unknown"


class BinBlockingAssessment(BaseModel):
    """Assessment of whether a bin is potentially blocking."""
    bin_path: str = Field(..., description="Full path to the bin")
    is_potentially_blocking: bool = Field(..., description="Whether bin might be impossible to cover")
    blocking_reason: Optional[BlockingReason] = Field(None, description="Reason for blocking")
    confidence: float = Field(..., description="Confidence in assessment (0-1)")
    recommendation: str = Field(..., description="Recommended action")


class ClosurePrediction(BaseModel):
    """
    Complete coverage closure prediction.
    
    Contains all predictive metrics for coverage closure including
    time estimates, probability assessments, and blocking bin analysis.
    """
    # Time estimates
    estimated_time_to_closure_hours: float = Field(
        ..., 
        description="Estimated hours to reach 100% coverage"
    )
    estimated_time_to_90_percent_hours: float = Field(
        ...,
        description="Estimated hours to reach 90% coverage"
    )
    confidence_interval_hours: tuple[float, float] = Field(
        ...,
        description="Confidence interval for time estimate (min, max)"
    )
    
    # Probability assessments
    closure_probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Probability of reaching 100% with current approach"
    )
    achievable_coverage: float = Field(
        ...,
        description="Estimated maximum achievable coverage percentage"
    )
    
    # Velocity metrics
    current_coverage_velocity: float = Field(
        ...,
        description="Current coverage points per hour (estimated)"
    )
    required_velocity_for_deadline: Optional[float] = Field(
        None,
        description="Required velocity to meet deadline (if provided)"
    )
    
    # Blocking bins analysis
    blocking_bins: list[BinBlockingAssessment] = Field(
        default_factory=list,
        description="List of potentially blocking bins"
    )
    total_blocking_bins: int = Field(0, description="Count of potentially blocking bins")
    
    # Risk assessment
    risk_level: str = Field(..., description="Overall risk level: low/medium/high/critical")
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Identified risk factors"
    )
    
    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations"
    )
    
    def to_summary(self) -> str:
        """Generate a text summary of the prediction."""
        lines = [
            "=== CLOSURE PREDICTION SUMMARY ===",
            "",
            f"Estimated Time to 100% Coverage: {self.estimated_time_to_closure_hours:.1f} hours",
            f"  Confidence Interval: {self.confidence_interval_hours[0]:.1f} - {self.confidence_interval_hours[1]:.1f} hours",
            f"Estimated Time to 90% Coverage: {self.estimated_time_to_90_percent_hours:.1f} hours",
            "",
            f"Closure Probability: {self.closure_probability * 100:.1f}%",
            f"Maximum Achievable Coverage: {self.achievable_coverage:.1f}%",
            "",
            f"Coverage Velocity: {self.current_coverage_velocity:.2f} bins/hour",
            "",
            f"Risk Level: {self.risk_level.upper()}",
            f"Potentially Blocking Bins: {self.total_blocking_bins}",
            "",
            "Risk Factors:"
        ]
        
        for rf in self.risk_factors:
            lines.append(f"  • {rf}")
        
        lines.extend(["", "Recommendations:"])
        for rec in self.recommendations:
            lines.append(f"  → {rec}")
        
        return "\n".join(lines)


class ClosurePredictor:
    """
    Predicts coverage closure metrics based on current state and historical patterns.
    
    This is a BONUS FEATURE that provides:
    1. Time-to-closure estimation based on difficulty distribution
    2. Closure probability based on blocking bin analysis
    3. Blocking bin identification using heuristics
    
    The model uses a combination of:
    - Difficulty-based effort estimation
    - Dependency chain analysis
    - Cross-coverage complexity assessment
    - Historical velocity assumptions
    
    Usage:
        predictor = ClosurePredictor(report, suggestions)
        prediction = predictor.predict()
        print(prediction.to_summary())
    """
    
    # Effort multipliers by difficulty (hours per bin)
    EFFORT_PER_BIN = {
        "easy": 1.0,
        "medium": 3.0,
        "hard": 8.0
    }
    
    # Default velocity (bins covered per hour)
    DEFAULT_VELOCITY = 0.5
    
    # Blocking patterns - keywords that suggest blocking bins
    BLOCKING_PATTERNS = {
        "error_injection": ["decode_error", "parity_error", "ecc_error"],
        "corner_case": ["max", "min", "boundary", "overflow", "underflow"],
        "multi_channel": ["all_eight", "all_channels", "full_capacity"],
        "timing_related": ["timeout", "latency", "jitter"],
        "recovery": ["retry", "recovery", "abort", "reset"]
    }
    
    def __init__(
        self,
        report: CoverageReport,
        suggestions: list[PrioritizedSuggestion] = None,
        deadline_hours: Optional[float] = None,
        historical_velocity: Optional[float] = None
    ):
        """
        Initialize the closure predictor.
        
        Args:
            report: Parsed coverage report
            suggestions: Prioritized suggestions (for difficulty info)
            deadline_hours: Optional deadline in hours
            historical_velocity: Historical coverage velocity (bins/hour)
        """
        self.report = report
        self.suggestions = suggestions or []
        self.deadline_hours = deadline_hours
        self.velocity = historical_velocity or self.DEFAULT_VELOCITY
    
    def predict(self) -> ClosurePrediction:
        """
        Generate complete closure prediction.
        
        Returns:
            ClosurePrediction with all metrics
        """
        # Calculate time estimates
        time_estimates = self._estimate_time_to_closure()
        
        # Identify blocking bins
        blocking_bins = self._identify_blocking_bins()
        
        # Calculate closure probability
        closure_prob = self._calculate_closure_probability(blocking_bins)
        
        # Calculate achievable coverage
        achievable = self._calculate_achievable_coverage(blocking_bins)
        
        # Assess risk
        risk_level, risk_factors = self._assess_risk(blocking_bins, time_estimates)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            blocking_bins, risk_level, time_estimates
        )
        
        # Calculate required velocity for deadline
        required_velocity = None
        if self.deadline_hours:
            uncovered = self.report.uncovered_count
            required_velocity = uncovered / self.deadline_hours if self.deadline_hours > 0 else float('inf')
        
        return ClosurePrediction(
            estimated_time_to_closure_hours=time_estimates["total"],
            estimated_time_to_90_percent_hours=time_estimates["to_90"],
            confidence_interval_hours=time_estimates["confidence_interval"],
            closure_probability=closure_prob,
            achievable_coverage=achievable,
            current_coverage_velocity=self.velocity,
            required_velocity_for_deadline=required_velocity,
            blocking_bins=blocking_bins,
            total_blocking_bins=len([b for b in blocking_bins if b.is_potentially_blocking]),
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations
        )
    
    def _estimate_time_to_closure(self) -> dict:
        """
        Estimate time to reach various coverage targets.
        
        Uses difficulty-weighted effort estimation.
        """
        # Count suggestions by difficulty
        difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
        
        if self.suggestions:
            for s in self.suggestions:
                diff = s.difficulty.lower()
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        else:
            # Estimate based on uncovered bins without difficulty info
            total_uncovered = self.report.uncovered_count
            # Assume distribution: 30% easy, 50% medium, 20% hard
            difficulty_counts["easy"] = int(total_uncovered * 0.3)
            difficulty_counts["medium"] = int(total_uncovered * 0.5)
            difficulty_counts["hard"] = total_uncovered - difficulty_counts["easy"] - difficulty_counts["medium"]
        
        # Calculate total effort
        total_effort = sum(
            count * self.EFFORT_PER_BIN[diff]
            for diff, count in difficulty_counts.items()
        )
        
        # Estimate time to 90% coverage (typically easier bins first)
        # Assume 90% can be achieved with easy + medium bins
        easy_medium_effort = (
            difficulty_counts["easy"] * self.EFFORT_PER_BIN["easy"] +
            difficulty_counts["medium"] * self.EFFORT_PER_BIN["medium"]
        )
        
        # Current coverage and gap
        current = self.report.overall_coverage
        gap_to_100 = 100 - current
        gap_to_90 = max(0, 90 - current)
        
        # Scale estimates by remaining gap
        scale_factor = gap_to_100 / 100 if gap_to_100 > 0 else 0
        
        # Confidence interval (±30%)
        min_time = total_effort * 0.7
        max_time = total_effort * 1.5
        
        return {
            "total": round(total_effort * scale_factor, 1),
            "to_90": round(min(easy_medium_effort * scale_factor, total_effort * 0.6), 1),
            "confidence_interval": (round(min_time * scale_factor, 1), round(max_time * scale_factor, 1)),
            "by_difficulty": {
                diff: round(count * self.EFFORT_PER_BIN[diff], 1)
                for diff, count in difficulty_counts.items()
            }
        }
    
    def _identify_blocking_bins(self) -> list[BinBlockingAssessment]:
        """
        Identify bins that might be impossible to cover.
        
        Uses pattern matching and heuristics to identify potential blockers.
        """
        blocking_assessments = []
        
        for ub in self.report.uncovered_bins:
            bin_name = ub.bin.lower()
            covergroup_name = ub.covergroup.lower()
            full_path = ub.full_path
            
            is_blocking = False
            reason = None
            confidence = 0.0
            recommendation = ""
            
            # Check for error injection patterns
            if any(pattern in bin_name for pattern in self.BLOCKING_PATTERNS["error_injection"]):
                # Error injection might require special testbench infrastructure
                is_blocking = self._check_error_injection_feasibility(ub)
                if is_blocking:
                    reason = BlockingReason.HARDWARE_MISSING
                    confidence = 0.7
                    recommendation = "Verify testbench has error injection capability"
            
            # Check for max capacity patterns
            elif any(pattern in bin_name for pattern in self.BLOCKING_PATTERNS["multi_channel"]):
                is_blocking = self._check_max_capacity_feasibility(ub)
                if is_blocking:
                    reason = BlockingReason.CONFIG_IMPOSSIBLE
                    confidence = 0.6
                    recommendation = "Verify DUT supports maximum channel configuration"
            
            # Check for timing-related patterns
            elif any(pattern in bin_name for pattern in self.BLOCKING_PATTERNS["timing_related"]):
                is_blocking = self._check_timing_feasibility(ub)
                if is_blocking:
                    reason = BlockingReason.TIMING_CONSTRAINT
                    confidence = 0.5
                    recommendation = "Review timing constraints and testbench capabilities"
            
            # Check for recovery patterns (often depend on error scenarios)
            elif any(pattern in bin_name for pattern in self.BLOCKING_PATTERNS["recovery"]):
                # Recovery depends on error injection
                is_blocking = "error" in covergroup_name and self._check_error_injection_feasibility(ub)
                if is_blocking:
                    reason = BlockingReason.DEPENDENCY_CHAIN
                    confidence = 0.65
                    recommendation = "Error scenarios must be covered first"
            
            # Create assessment
            if not recommendation:
                recommendation = "Standard test development required"
            
            blocking_assessments.append(BinBlockingAssessment(
                bin_path=full_path,
                is_potentially_blocking=is_blocking,
                blocking_reason=reason,
                confidence=confidence,
                recommendation=recommendation
            ))
        
        # Also check cross-coverage for combinatorial explosion
        for xc in self.report.cross_coverage:
            if len(xc.uncovered) > 10:
                # Many uncovered cross bins might indicate combinatorial issues
                for uncovered_val in xc.uncovered[:5]:  # Sample first 5
                    blocking_assessments.append(BinBlockingAssessment(
                        bin_path=f"{xc.name}.{uncovered_val}",
                        is_potentially_blocking=True,
                        blocking_reason=BlockingReason.CROSS_COVERAGE_EXPLOSION,
                        confidence=0.4,
                        recommendation="Review if all cross combinations are valid/achievable"
                    ))
        
        return blocking_assessments
    
    def _check_error_injection_feasibility(self, bin_info: UncoveredBinInfo) -> bool:
        """Check if error injection is likely feasible."""
        # If no error bins are covered at all, injection might not be working
        for cg in self.report.covergroups:
            if "error" in cg.name.lower():
                # Check if ANY error bins are covered
                for cp in cg.coverpoints:
                    for b in cp.bins:
                        if b.covered and "error" in b.name.lower():
                            return False  # Error injection works
                return True  # No error bins covered - might be blocking
        return False
    
    def _check_max_capacity_feasibility(self, bin_info: UncoveredBinInfo) -> bool:
        """Check if max capacity configuration is likely feasible."""
        # Look for patterns - if lower capacity works but max doesn't
        covergroup = bin_info.covergroup
        coverpoint = bin_info.coverpoint
        
        for cg in self.report.covergroups:
            if cg.name == covergroup:
                for cp in cg.coverpoints:
                    if cp.name == coverpoint:
                        covered = [b for b in cp.bins if b.covered]
                        uncovered = [b for b in cp.bins if not b.covered]
                        
                        # If we have progression (1,2,3 covered but 4+ not)
                        # it might be a configuration limit
                        if len(covered) >= 2 and len(uncovered) >= 2:
                            return True
        return False
    
    def _check_timing_feasibility(self, bin_info: UncoveredBinInfo) -> bool:
        """Check if timing-related bin is likely achievable."""
        # Timeout bins often require long simulations
        if "timeout" in bin_info.bin.lower():
            return True  # Timeouts often need special handling
        return False
    
    def _calculate_closure_probability(
        self,
        blocking_bins: list[BinBlockingAssessment]
    ) -> float:
        """
        Calculate probability of reaching 100% coverage.
        
        Based on blocking bins and current coverage trends.
        """
        total_uncovered = self.report.uncovered_count
        if total_uncovered == 0:
            return 1.0
        
        blocking_count = len([b for b in blocking_bins if b.is_potentially_blocking])
        
        # Base probability from coverage level
        current_coverage = self.report.overall_coverage
        base_prob = min(current_coverage / 100, 0.95)  # Cap at 95%
        
        # Reduce by blocking bin ratio
        blocking_ratio = blocking_count / total_uncovered if total_uncovered > 0 else 0
        blocking_penalty = blocking_ratio * 0.5  # Up to 50% reduction
        
        # Calculate final probability
        probability = base_prob * (1 - blocking_penalty)
        
        # Adjust for coverage difficulty
        if current_coverage > 90:
            # Last 10% is always hardest
            probability *= 0.8
        elif current_coverage > 80:
            probability *= 0.9
        
        return round(max(0.1, min(probability, 0.99)), 3)
    
    def _calculate_achievable_coverage(
        self,
        blocking_bins: list[BinBlockingAssessment]
    ) -> float:
        """Calculate maximum achievable coverage percentage."""
        total_bins = self.report.total_bins
        if total_bins == 0:
            return 100.0
        
        # Count high-confidence blocking bins
        high_confidence_blocking = len([
            b for b in blocking_bins
            if b.is_potentially_blocking and b.confidence >= 0.6
        ])
        
        # Calculate achievable
        achievable_bins = total_bins - high_confidence_blocking
        achievable_coverage = (achievable_bins / total_bins) * 100
        
        return round(min(achievable_coverage, 100.0), 2)
    
    def _assess_risk(
        self,
        blocking_bins: list[BinBlockingAssessment],
        time_estimates: dict
    ) -> tuple[str, list[str]]:
        """
        Assess overall risk level and identify risk factors.
        
        Returns:
            Tuple of (risk_level, risk_factors)
        """
        risk_factors = []
        risk_score = 0
        
        # Check coverage level
        current = self.report.overall_coverage
        if current < 50:
            risk_factors.append("Coverage is below 50%")
            risk_score += 3
        elif current < 70:
            risk_factors.append("Coverage is below 70%")
            risk_score += 2
        elif current < 85:
            risk_factors.append("Coverage is below 85%")
            risk_score += 1
        
        # Check blocking bins
        blocking_count = len([b for b in blocking_bins if b.is_potentially_blocking])
        if blocking_count > 5:
            risk_factors.append(f"{blocking_count} potentially blocking bins identified")
            risk_score += 3
        elif blocking_count > 2:
            risk_factors.append(f"{blocking_count} potentially blocking bins identified")
            risk_score += 2
        elif blocking_count > 0:
            risk_factors.append(f"{blocking_count} potentially blocking bin(s) identified")
            risk_score += 1
        
        # Check time estimate
        total_time = time_estimates.get("total", 0)
        if total_time > 40:
            risk_factors.append(f"Estimated {total_time:.0f} hours needed for closure")
            risk_score += 2
        elif total_time > 20:
            risk_factors.append(f"Estimated {total_time:.0f} hours needed for closure")
            risk_score += 1
        
        # Check deadline feasibility
        if self.deadline_hours and total_time > self.deadline_hours:
            risk_factors.append(f"Estimated time ({total_time:.0f}h) exceeds deadline ({self.deadline_hours:.0f}h)")
            risk_score += 3
        
        # Check error coverage
        for cg in self.report.covergroups:
            if "error" in cg.name.lower() and cg.coverage < 50:
                risk_factors.append(f"Error scenario coverage is low ({cg.coverage}%)")
                risk_score += 2
                break
        
        # Determine risk level
        if risk_score >= 7:
            risk_level = "critical"
        elif risk_score >= 5:
            risk_level = "high"
        elif risk_score >= 3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return risk_level, risk_factors
    
    def _generate_recommendations(
        self,
        blocking_bins: list[BinBlockingAssessment],
        risk_level: str,
        time_estimates: dict
    ) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Risk-based recommendations
        if risk_level in ["critical", "high"]:
            recommendations.append(
                "URGENT: Schedule immediate triage meeting to review coverage gaps"
            )
        
        # Blocking bin recommendations
        blocking = [b for b in blocking_bins if b.is_potentially_blocking]
        if blocking:
            by_reason = {}
            for b in blocking:
                reason = b.blocking_reason.value if b.blocking_reason else "unknown"
                by_reason.setdefault(reason, []).append(b)
            
            for reason, bins in by_reason.items():
                if reason == "hardware_not_in_testbench":
                    recommendations.append(
                        f"Review testbench infrastructure for {len(bins)} bins requiring "
                        f"hardware features (e.g., error injection)"
                    )
                elif reason == "complex_dependency_chain":
                    recommendations.append(
                        f"Create prerequisite tests for {len(bins)} bins with dependencies"
                    )
                elif reason == "cross_coverage_dimension_explosion":
                    recommendations.append(
                        f"Review {len(bins)} cross-coverage combinations for validity"
                    )
        
        # Time-based recommendations
        total_time = time_estimates.get("total", 0)
        by_difficulty = time_estimates.get("by_difficulty", {})
        
        if by_difficulty.get("easy", 0) > 0:
            recommendations.append(
                f"Start with easy bins ({by_difficulty.get('easy', 0):.0f}h estimated) for quick wins"
            )
        
        if by_difficulty.get("hard", 0) > total_time * 0.5:
            recommendations.append(
                "Consider parallel test development for hard bins to reduce critical path"
            )
        
        # Coverage-specific recommendations
        if self.report.overall_coverage < 80:
            recommendations.append(
                "Focus on systematic coverage improvement before targeting specific holes"
            )
        else:
            recommendations.append(
                "Target remaining holes systematically using prioritized suggestions"
            )
        
        return recommendations[:6]  # Limit to 6 recommendations


def create_predictor(
    report: CoverageReport,
    suggestions: list[PrioritizedSuggestion] = None,
    deadline_hours: float = None
) -> ClosurePredictor:
    """Factory function to create a closure predictor."""
    return ClosurePredictor(
        report=report,
        suggestions=suggestions,
        deadline_hours=deadline_hours
    )
