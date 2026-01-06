#!/usr/bin/env python3
"""
Coverage Analyzer Web UI

A Streamlit-based web interface for the Verification Coverage Analyzer.
Provides an interactive dashboard for coverage analysis.

Run with: streamlit run web_ui.py

Author: ML Engineer
Date: 2025-01-06
"""

import json
import sys
from pathlib import Path
from io import StringIO

import streamlit as st
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parser.coverage_parser import CoverageParser
from src.analyzer.coverage_analyzer import CoverageAnalyzer
from src.predictor.closure_predictor import ClosurePredictor

# Page configuration
st.set_page_config(
    page_title="Coverage Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .suggestion-card {
        background-color: #e8f4ea;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-card {
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .error-card {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# Sample coverage report for demo
SAMPLE_REPORT = """
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


def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Coverage Analyzer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666;">Verification Coverage Analysis with LLM Integration</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Provider selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["None (Demo Mode)", "OpenAI", "Anthropic", "Ollama"],
            index=0
        )
        
        if llm_provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password")
        elif llm_provider == "Anthropic":
            api_key = st.text_input("Anthropic API Key", type="password")
        else:
            api_key = None
        
        st.divider()
        
        # Analysis options
        st.subheader("Analysis Options")
        batch_mode = st.checkbox("Batch Mode", value=True, help="Process all bins in a single LLM request")
        deadline_hours = st.number_input("Deadline (hours)", min_value=0, value=0, help="Optional deadline for predictions")
        
        st.divider()
        
        # About section
        st.subheader("About")
        st.markdown("""
        This tool analyzes coverage reports and generates AI-powered test suggestions
        to accelerate coverage closure.
        
        **Features:**
        - 📄 Parse coverage reports
        - 🤖 LLM-based suggestions
        - 📈 Priority scoring
        - 🔮 Closure prediction
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyze", "📊 Results Dashboard", "💡 Suggestions", "🔮 Predictions"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_results_tab()
    
    with tab3:
        render_suggestions_tab()
    
    with tab4:
        render_predictions_tab(deadline_hours if deadline_hours > 0 else None)


def render_upload_tab():
    """Render the upload and analysis tab."""
    st.header("Upload Coverage Report")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        upload_method = st.radio(
            "Input Method",
            ["Upload File", "Paste Text", "Use Sample"],
            horizontal=True
        )
        
        coverage_text = None
        
        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a coverage report file",
                type=['txt', 'rpt', 'log']
            )
            if uploaded_file:
                coverage_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        
        elif upload_method == "Paste Text":
            coverage_text = st.text_area(
                "Paste coverage report here",
                height=300,
                placeholder="Paste your coverage report..."
            )
        
        else:  # Use Sample
            coverage_text = SAMPLE_REPORT
            st.info("Using sample DMA controller coverage report")
            with st.expander("View Sample Report"):
                st.code(SAMPLE_REPORT, language="text")
    
    with col2:
        st.subheader("Analysis Settings")
        
        use_llm = st.checkbox("Enable LLM Suggestions", value=False)
        show_json = st.checkbox("Show Raw JSON Output", value=False)
        
        if st.button("🚀 Analyze", type="primary", use_container_width=True):
            if coverage_text:
                with st.spinner("Analyzing coverage report..."):
                    try:
                        # Run analysis
                        analyzer = CoverageAnalyzer(use_llm=use_llm)
                        result = analyzer.analyze(coverage_text)
                        
                        # Store in session state
                        st.session_state['analysis_result'] = result
                        st.session_state['coverage_text'] = coverage_text
                        
                        st.success("✅ Analysis complete!")
                        
                        if show_json:
                            st.json(json.loads(result.to_json()))
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            else:
                st.warning("Please provide a coverage report to analyze.")


def render_results_tab():
    """Render the results dashboard tab."""
    if 'analysis_result' not in st.session_state:
        st.info("👆 Please upload and analyze a coverage report first.")
        return
    
    result = st.session_state['analysis_result']
    report = result.report
    
    st.header(f"Coverage Analysis: {report.design}")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Coverage",
            f"{report.overall_coverage}%",
            delta=f"{100 - report.overall_coverage:.1f}% to go"
        )
    
    with col2:
        st.metric("Total Bins", report.total_bins)
    
    with col3:
        st.metric("Covered Bins", report.covered_bins)
    
    with col4:
        st.metric("Uncovered Bins", report.uncovered_count)
    
    st.divider()
    
    # Covergroup breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Covergroup Breakdown")
        
        cg_data = []
        for cg in report.covergroups:
            cg_data.append({
                "Covergroup": cg.name,
                "Coverage": cg.coverage,
                "Covered": cg.covered_bins,
                "Total": cg.total_bins,
                "Uncovered": cg.uncovered_count
            })
        
        df = pd.DataFrame(cg_data)
        
        # Color code coverage
        def color_coverage(val):
            if val >= 80:
                return 'background-color: #c8e6c9'
            elif val >= 50:
                return 'background-color: #fff9c4'
            else:
                return 'background-color: #ffcdd2'
        
        styled_df = df.style.applymap(color_coverage, subset=['Coverage'])
        st.dataframe(styled_df, use_container_width=True)
    
    with col2:
        st.subheader("Coverage Distribution")
        
        # Create chart data
        chart_data = pd.DataFrame({
            'Covergroup': [cg.name for cg in report.covergroups],
            'Coverage': [cg.coverage for cg in report.covergroups]
        })
        
        st.bar_chart(chart_data.set_index('Covergroup'))
    
    # Cross-coverage
    if report.cross_coverage:
        st.subheader("Cross-Coverage Status")
        
        for xc in report.cross_coverage:
            with st.expander(f"{xc.name} - {xc.coverage}%"):
                st.write(f"**Covered:** {xc.covered_bins}/{xc.total_bins} bins")
                
                if xc.uncovered:
                    st.write("**Uncovered combinations:**")
                    for u in xc.uncovered:
                        st.code(u)
    
    # Uncovered bins list
    st.subheader("Uncovered Bins")
    
    uncovered_data = []
    for ub in report.uncovered_bins:
        uncovered_data.append({
            "Covergroup": ub.covergroup,
            "Coverpoint": ub.coverpoint,
            "Bin": ub.bin,
            "Full Path": ub.full_path
        })
    
    if uncovered_data:
        st.dataframe(pd.DataFrame(uncovered_data), use_container_width=True)


def render_suggestions_tab():
    """Render the suggestions tab."""
    if 'analysis_result' not in st.session_state:
        st.info("👆 Please upload and analyze a coverage report first.")
        return
    
    result = st.session_state['analysis_result']
    suggestions = result.suggestions
    
    st.header("Test Suggestions")
    
    if not suggestions:
        st.warning("No suggestions generated. Try enabling LLM integration.")
        return
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        difficulty_filter = st.multiselect(
            "Filter by Difficulty",
            ["easy", "medium", "hard"],
            default=["easy", "medium", "hard"]
        )
    
    with col2:
        priority_filter = st.multiselect(
            "Filter by Priority",
            ["high", "medium", "low"],
            default=["high", "medium", "low"]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Priority Score", "Difficulty", "Target Bin"]
        )
    
    # Filter suggestions
    filtered = [
        s for s in suggestions
        if s.difficulty.lower() in difficulty_filter
        and s.priority.lower() in priority_filter
    ]
    
    # Sort
    if sort_by == "Priority Score":
        filtered.sort(key=lambda x: x.priority_score, reverse=True)
    elif sort_by == "Difficulty":
        order = {"easy": 0, "medium": 1, "hard": 2}
        filtered.sort(key=lambda x: order.get(x.difficulty.lower(), 1))
    else:
        filtered.sort(key=lambda x: x.target_bin)
    
    st.write(f"Showing {len(filtered)} of {len(suggestions)} suggestions")
    
    # Display suggestions
    for i, s in enumerate(filtered, 1):
        difficulty_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(s.difficulty.lower(), "⚪")
        priority_badge = {"high": "🔥", "medium": "📌", "low": "📎"}.get(s.priority.lower(), "")
        
        with st.expander(
            f"{difficulty_color} #{i} {s.target_bin} - Score: {s.priority_score:.3f} {priority_badge}"
        ):
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric("Priority Score", f"{s.priority_score:.3f}")
            with col2:
                st.metric("Difficulty", s.difficulty.capitalize())
            with col3:
                st.metric("Est. Effort", f"{s.estimated_effort_hours or 'N/A'}h")
            
            st.markdown("**Suggestion:**")
            st.info(s.suggestion)
            
            st.markdown("**Reasoning:**")
            st.write(s.reasoning)
            
            if s.test_outline:
                st.markdown("**Test Outline:**")
                for step in s.test_outline:
                    st.write(f"  {step}")
            
            if s.dependencies:
                st.markdown("**Dependencies:**")
                for dep in s.dependencies:
                    st.write(f"  ⚠️ {dep}")


def render_predictions_tab(deadline_hours=None):
    """Render the predictions tab."""
    if 'analysis_result' not in st.session_state:
        st.info("👆 Please upload and analyze a coverage report first.")
        return
    
    result = st.session_state['analysis_result']
    
    st.header("Coverage Closure Prediction")
    
    # Generate prediction
    predictor = ClosurePredictor(
        result.report,
        result.suggestions,
        deadline_hours=deadline_hours
    )
    prediction = predictor.predict()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Time to 100%",
            f"{prediction.estimated_time_to_closure_hours:.1f}h"
        )
    
    with col2:
        st.metric(
            "Time to 90%",
            f"{prediction.estimated_time_to_90_percent_hours:.1f}h"
        )
    
    with col3:
        prob_percent = prediction.closure_probability * 100
        st.metric(
            "Closure Probability",
            f"{prob_percent:.1f}%"
        )
    
    with col4:
        st.metric(
            "Achievable Coverage",
            f"{prediction.achievable_coverage:.1f}%"
        )
    
    st.divider()
    
    # Risk Assessment
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Assessment")
        
        risk_colors = {
            "low": "🟢 Low",
            "medium": "🟡 Medium",
            "high": "🟠 High",
            "critical": "🔴 Critical"
        }
        
        st.markdown(f"### {risk_colors.get(prediction.risk_level, prediction.risk_level)}")
        
        if prediction.risk_factors:
            st.markdown("**Risk Factors:**")
            for rf in prediction.risk_factors:
                st.warning(rf)
    
    with col2:
        st.subheader("Blocking Bins Analysis")
        
        st.metric("Potentially Blocking", prediction.total_blocking_bins)
        
        blocking = [b for b in prediction.blocking_bins if b.is_potentially_blocking]
        
        if blocking:
            for b in blocking[:5]:
                reason = b.blocking_reason.value if b.blocking_reason else "Unknown"
                st.error(f"**{b.bin_path}**\n\nReason: {reason}\n\n{b.recommendation}")
        else:
            st.success("No blocking bins detected!")
    
    # Recommendations
    st.subheader("Recommendations")
    
    for rec in prediction.recommendations:
        st.info(f"💡 {rec}")
    
    # Confidence interval
    st.subheader("Time Estimate Confidence")
    
    min_time, max_time = prediction.confidence_interval_hours
    
    st.write(f"Estimated time to closure: **{prediction.estimated_time_to_closure_hours:.1f}** hours")
    st.write(f"Confidence interval: {min_time:.1f} - {max_time:.1f} hours")
    
    # Visual progress bar
    progress = (max_time - prediction.estimated_time_to_closure_hours) / (max_time - min_time) if max_time != min_time else 0.5
    st.progress(progress)


if __name__ == "__main__":
    main()
