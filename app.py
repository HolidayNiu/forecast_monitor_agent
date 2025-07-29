"""
Streamlit app for forecast monitoring agent.
"""
# Load environment variables first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path

from modules.loader import load_data, get_item_data, get_all_item_ids, get_recent_actuals, get_early_forecast
from modules.diagnostics import run_all_diagnostics
from modules.visualizer import plot_forecast_analysis, create_summary_stats_table
from modules.explainer_simple import (prepare_analysis_summary, generate_explanation, 
                                     format_explanation_report, check_llm_availability, 
                                     get_preferred_provider)
from modules.reporter import create_detailed_report

# Import batch processing modules
try:
    from src.detector.batch_detector import BatchDetector
    from src.monitoring.csv_exporter import CSVExporter
    from src.monitoring.json_summarizer import JSONSummarizer
    import config
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False


@st.cache_data
def run_batch_diagnostics(df):
    """Run batch diagnostics and cache the results."""
    if not BATCH_AVAILABLE:
        return None, None
    
    try:
        detector = BatchDetector(df, max_workers=1)  # Single worker for Streamlit
        results = detector.process_all_parts()
        summary_stats = detector.get_summary_stats()
        return results, summary_stats
    except Exception as e:
        st.error(f"Batch processing failed: {e}")
        return None, None


def create_batch_summary_visualizations(results, summary_stats):
    """Create visualizations for batch summary."""
    if not results or not summary_stats:
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Parts", summary_stats.get('total_parts', 0))
    
    with col2:
        flagged = summary_stats.get('flagged_parts', 0)
        total = summary_stats.get('total_parts', 1)
        st.metric("Parts with Issues", flagged, f"{flagged/total*100:.1f}%")
    
    with col3:
        st.metric("Average Risk Score", f"{summary_stats.get('avg_risk_score', 0):.3f}")
    
    with col4:
        st.metric("Avg Issues/Part", f"{summary_stats.get('avg_issues_per_part', 0):.1f}")
    
    # Issue breakdown chart
    st.subheader("📊 Issue Breakdown")
    issues = summary_stats.get('issues', {})
    
    if any(len(parts) > 0 for parts in issues.values()):
        # Create issue counts
        issue_data = []
        for issue_type, part_list in issues.items():
            if part_list:
                issue_data.append({
                    'Issue Type': issue_type.replace('_', ' ').title(),
                    'Count': len(part_list),
                    'Parts': ', '.join(part_list[:5]) + ('...' if len(part_list) > 5 else '')
                })
        
        if issue_data:
            issue_df = pd.DataFrame(issue_data)
            
            # Bar chart
            fig = px.bar(issue_df, x='Issue Type', y='Count', 
                        title="Number of Parts by Issue Type",
                        color='Count',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            
            # Issue details table
            st.dataframe(issue_df, use_container_width=True)
    
    # Severity breakdown
    st.subheader("⚠️ Severity Distribution")
    severity_breakdown = summary_stats.get('severity_breakdown', {})
    
    if severity_breakdown:
        severity_data = []
        colors = {'high': '#FF4B4B', 'medium': '#FFA500', 'low': '#FFFF00', 'none': '#00FF00'}
        
        for severity, count in severity_breakdown.items():
            if count > 0:
                severity_data.append({
                    'Severity': severity.title(),
                    'Count': count,
                    'Color': colors.get(severity, '#808080')
                })
        
        if severity_data:
            severity_df = pd.DataFrame(severity_data)
            
            # Pie chart
            fig = px.pie(severity_df, values='Count', names='Severity',
                        title="Parts by Severity Level",
                        color='Severity',
                        color_discrete_map={row['Severity']: row['Color'] for _, row in severity_df.iterrows()})
            st.plotly_chart(fig, use_container_width=True)


def create_top_issues_table(results):
    """Create a table of top problematic parts."""
    if not results:
        return
    
    # Filter parts with issues and sort by risk score
    parts_with_issues = [r for r in results if r.get('total_issues', 0) > 0]
    parts_with_issues.sort(key=lambda x: x.get('risk_score', 0), reverse=True)
    
    if not parts_with_issues:
        st.info("No parts with issues detected.")
        return
    
    # Create display data
    display_data = []
    for part in parts_with_issues[:10]:  # Top 10
        display_data.append({
            'Part ID': part['part_id'],
            'Risk Score': f"{part.get('risk_score', 0):.3f}",
            'Issues': part.get('total_issues', 0),
            'Severity': part.get('severity', 'none').title(),
            'Problems': part.get('comment', 'No description')
        })
    
    if display_data:
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True)


def display_batch_summary(df):
    """Display the complete batch summary section."""
    st.header("📋 Batch Analysis Summary")
    
    if not BATCH_AVAILABLE:
        st.error("Batch processing modules not available. Please check your installation.")
        return
    
    # Run batch processing
    with st.spinner("Running batch diagnostics..."):
        results, summary_stats = run_batch_diagnostics(df)
    
    if results is None:
        st.error("Failed to run batch diagnostics.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "🔥 Top Issues", "📁 Downloads"])
    
    with tab1:
        create_batch_summary_visualizations(results, summary_stats)
    
    with tab2:
        st.subheader("🔥 Most Problematic Parts")
        create_top_issues_table(results)
    
    with tab3:
        st.subheader("📁 Export Reports")
        
        # Generate and offer downloads
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate CSV Report"):
                with st.spinner("Generating CSV..."):
                    try:
                        csv_exporter = CSVExporter()
                        csv_df = csv_exporter.format_results_for_csv(results)
                        
                        csv = csv_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download CSV Report",
                            data=csv,
                            file_name="forecast_diagnostics.csv",
                            mime="text/csv"
                        )
                        st.success("CSV report ready for download!")
                    except Exception as e:
                        st.error(f"Failed to generate CSV: {e}")
        
        with col2:
            if st.button("Generate JSON Summary"):
                with st.spinner("Generating JSON..."):
                    try:
                        json_summarizer = JSONSummarizer()
                        payload = json_summarizer.create_summary_payload(results, summary_stats)
                        
                        json_str = json.dumps(payload, indent=2)
                        st.download_button(
                            label="🤖 Download JSON Summary",
                            data=json_str,
                            file_name="summary_payload.json",
                            mime="application/json"
                        )
                        st.success("JSON summary ready for download!")
                    except Exception as e:
                        st.error(f"Failed to generate JSON: {e}")


def display_individual_analysis(df, item_ids):
    """Display the individual part analysis interface."""
    st.header("🔍 Individual Part Analysis")
    
    st.sidebar.header("Configuration")
    selected_item = st.sidebar.selectbox("Select Item ID", item_ids)
    
    # LLM Configuration
    st.sidebar.subheader("🤖 AI Settings")
    
    # Import LLM client and get available providers
    import os
    import sys
    sys.path.append('modules')
    
    try:
        from modules.llm_client_robust import get_available_providers as direct_get_providers
        available_providers = direct_get_providers()
    except Exception as e:
        st.sidebar.error(f"AI services unavailable: {e}")
        available_providers = {"claude": False, "openai": False}
    
    if any(available_providers.values()):
        # Show available providers
        available_list = [p for p, available in available_providers.items() if available]
        preferred = "claude" if available_providers.get("claude", False) else available_list[0]
        default_idx = available_list.index(preferred) if preferred in available_list else 0
        
        llm_provider = st.sidebar.selectbox(
            "AI Provider", 
            available_list, 
            index=default_idx,
            help="Select AI provider for generating explanations"
        )
        st.sidebar.success(f"✅ {llm_provider.title()} AI ready")
        use_llm = True
    else:
        st.sidebar.error("❌ No AI providers configured")
        st.sidebar.info("💡 Configure API keys to enable AI explanations")
        use_llm = False
        llm_provider = None
    
    if selected_item:
        # Load item data
        historical_data, forecast_data = get_item_data(df, selected_item)
        recent_actuals = get_recent_actuals(historical_data)
        early_forecast = get_early_forecast(forecast_data)
        
        # Run diagnostics
        diagnostics = run_all_diagnostics(historical_data, forecast_data, recent_actuals, early_forecast)
        
        # Create explanation
        analysis_summary = prepare_analysis_summary(selected_item, diagnostics, historical_data, forecast_data)
        
        if use_llm and llm_provider:
            try:
                # Use AI to generate explanation
                from modules.llm_client_robust import get_explanation
                explanation = get_explanation(analysis_summary, provider=llm_provider)
                used_provider = llm_provider
            except Exception as e:
                st.error(f"❌ AI Error: {e}")
                explanation = "Unable to generate AI explanation due to an error."
                used_provider = "error"
        else:
            explanation = "No AI provider available. Please configure your API keys."
            used_provider = "none"
        
        report = format_explanation_report(selected_item, diagnostics, explanation, used_provider)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Time Series Analysis")
            
            # Create and display plot
            fig = plot_forecast_analysis(historical_data, forecast_data, diagnostics, selected_item)
            st.pyplot(fig)
            plt.close()
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            stats_df = create_summary_stats_table(historical_data, forecast_data)
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.subheader("🚨 Issue Detection")
            
            # Risk score
            risk_score = report['risk_score']
            if risk_score > 0.5:
                st.error(f"🔴 High Risk: {risk_score:.3f}")
            elif risk_score > 0.2:
                st.warning(f"🟡 Medium Risk: {risk_score:.3f}")
            else:
                st.success(f"🟢 Low Risk: {risk_score:.3f}")
            
            # Issues summary
            st.metric("Issues Detected", report['total_issues'])
            st.metric("Avg Confidence", f"{report['avg_confidence']:.3f}")
            
            # Individual issues
            issues = diagnostics
            
            if issues['trend_mismatch']['detected']:
                st.error(f"❌ Trend Mismatch ({issues['trend_mismatch']['confidence']:.2f})")
            
            if issues['missing_seasonality']['detected']:
                st.error(f"❌ Missing Seasonality ({issues['missing_seasonality']['confidence']:.2f})")
            
            if issues['volatility_mismatch']['detected']:
                st.error(f"❌ Too Flat ({issues['volatility_mismatch']['confidence']:.2f})")
            
            if issues['magnitude_mismatch']['detected']:
                st.error(f"❌ Magnitude Mismatch ({issues['magnitude_mismatch']['confidence']:.2f})")
            
            if report['total_issues'] == 0:
                st.success("✅ No issues detected")
        
        # Explanation section
        st.subheader("🤖 AI Explanation")
        if used_provider in ["claude", "openai"]:
            st.caption(f"✅ Generated by {used_provider.title()} AI")
            st.info(explanation)
        elif used_provider == "error":
            st.caption("❌ AI generation failed")
            st.error(explanation)
        else:
            st.caption("❌ No AI provider configured")
            st.warning(explanation)
        
        # Detailed diagnostics (expandable)
        with st.expander("🔍 Detailed Diagnostics"):
            detailed_report = create_detailed_report(report)
            st.text(detailed_report)
        
        # Data preview (expandable)
        with st.expander("📋 Data Preview"):
            st.write("**Historical Data (last 10 months):**")
            st.write(historical_data.tail(10))
            st.write("**Forecast Data (first 10 months):**")
            st.write(forecast_data.head(10))


def main():
    st.set_page_config(page_title="Forecast Monitor Agent", layout="wide")
    
    st.title("📊 Forecast Monitor Agent")
    st.markdown("**Portfolio-wide forecast diagnostics with scalable batch analysis**")
    st.markdown("*Switch to Individual Analysis mode for detailed part-by-part inspection*")
    
    # Load data
    try:
        df = load_data("data/data.csv")
        item_ids = get_all_item_ids(df)
        
        # Navigation - Default to Batch Summary
        st.sidebar.header("📋 Navigation")
        analysis_mode = st.sidebar.radio(
            "Choose Analysis Mode",
            ["📊 Batch Summary", "🔍 Individual Part Analysis"],
            index=0,  # Default to batch summary
            help="Switch between portfolio overview and detailed individual part analysis"
        )
        
        # Show individual part analysis if selected
        if analysis_mode == "🔍 Individual Part Analysis":
            # Individual part analysis section
            display_individual_analysis(df, item_ids)
            return
        
        # Batch summary is now the main page - display it by default
        display_batch_summary(df)
    
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/data.csv' exists.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


if __name__ == "__main__":
    main()