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

from modules.loader import load_data, get_item_data, get_all_item_ids
from modules.visualizer import plot_forecast_analysis, create_summary_stats_table
from modules.explainer_simple import (prepare_analysis_summary, generate_explanation, 
                                     format_explanation_report, check_llm_availability, 
                                     get_preferred_provider)
from modules.reporter import create_detailed_report

# Try to import DetectorAgent for agent-based processing
try:
    from agent.detector_agent import DetectorAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# Try to import BasicRetrainAgent
try:
    from agent.basic_retrain_agent import BasicRetrainAgent
    from modules.retrain_visualizer import plot_simple_before_after
    AUTO_RETRAIN_AVAILABLE = True
except ImportError:
    AUTO_RETRAIN_AVAILABLE = False

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






def create_simple_batch_table(results, df):
    """Create a simple table showing all parts with their summary information."""
    if not results:
        st.info("No results to display.")
        return
    
    # Create display data for all parts
    display_data = []
    for result in results:
        part_id = result.get('part_id')
        
        # Get additional data from original dataframe
        part_data = df[df['item_loc_id'] == part_id].iloc[0] if len(df[df['item_loc_id'] == part_id]) > 0 else None
        
        if part_data is not None:
            product_bu = part_data['product_bu']
            location = part_data['location']
            price = part_data['local_price']
            
            display_data.append({
                'Part Number': part_id,
                'Product BU': product_bu,
                'Location': location,
                'Price': f"${price:,.2f}",
                'Detected Issue': result.get('comment', 'No issues detected'),
                'Risk Score': f"{result.get('risk_score', 0):.3f}",
                'Issue Count': result.get('total_issues', 0)
            })
    
    if not display_data:
        st.info("No parts to display.")
        return
    
    # Convert to DataFrame
    display_df = pd.DataFrame(display_data)
    
    # Show the table with clickable part numbers
    event = st.dataframe(
        display_df, 
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Show summary stats
    total_parts = len(display_df)
    parts_with_issues = len(display_df[display_df['Issue Count'] > 0])
    st.caption(f"Showing {total_parts} parts ({parts_with_issues} with issues)")
    
    # Handle row selection for detailed analysis
    if event.selection.rows:
        selected_row = event.selection.rows[0]
        selected_part = display_df.iloc[selected_row]['Part Number']
        
        # Display simple analysis for selected part
        st.divider()
        display_simple_part_analysis(df, selected_part)


def display_simple_part_analysis(df, selected_item):
    """Display simple, clean analysis for selected part."""
    # Get LLM provider for basic explanations
    try:
        from modules.llm_client_robust import get_available_providers
        available_providers = get_available_providers()
        if any(available_providers.values()):
            llm_provider = "claude" if available_providers.get("claude", False) else list(available_providers.keys())[0]
            use_llm = True
        else:
            use_llm = False
            llm_provider = None
    except:
        use_llm = False
        llm_provider = None
    
    # Load data and run diagnostics
    historical_data, forecast_data = get_item_data(df, selected_item)
    
    if AGENT_AVAILABLE:
        try:
            agent = DetectorAgent()
            diagnostics = agent.process_single(historical_data, forecast_data)
        except:
            from modules.diagnostics import run_all_diagnostics
            from modules.loader import get_recent_actuals, get_early_forecast
            recent_actuals = get_recent_actuals(historical_data)
            early_forecast = get_early_forecast(forecast_data)
            diagnostics = run_all_diagnostics(historical_data, forecast_data, recent_actuals, early_forecast)
    else:
        from modules.diagnostics import run_all_diagnostics
        from modules.loader import get_recent_actuals, get_early_forecast
        recent_actuals = get_recent_actuals(historical_data)
        early_forecast = get_early_forecast(forecast_data)
        diagnostics = run_all_diagnostics(historical_data, forecast_data, recent_actuals, early_forecast)
    
    # Generate simple explanation
    if use_llm and llm_provider:
        try:
            from modules.explainer_simple import prepare_analysis_summary, get_explanation
            summary = prepare_analysis_summary(selected_item, diagnostics, historical_data, forecast_data)
            explanation = get_explanation(summary, provider=llm_provider)
            analysis_method = "llm"
        except:
            explanation = "Issues detected. Consider retraining with different models."
            analysis_method = "basic"
    else:
        explanation = "Issues detected. Consider retraining with different models."
        analysis_method = "basic"
    
    # === SECTION 1: ISSUE SUMMARY WITH VISUAL ===
    st.subheader(f"📊 {selected_item} - Issue Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series plot
        fig = plot_forecast_analysis(historical_data, forecast_data, diagnostics, selected_item)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Issue summary
        risk_score = diagnostics['summary']['risk_score']
        total_issues = diagnostics['summary']['total_issues']
        
        # Risk indicator
        if risk_score > 0.5:
            st.error(f"🔴 HIGH RISK: {risk_score:.2f}")
        elif risk_score > 0.2:
            st.warning(f"🟡 MEDIUM RISK: {risk_score:.2f}")
        else:
            st.success(f"🟢 LOW RISK: {risk_score:.2f}")
        
        st.metric("Issues Found", total_issues)
        
        # List specific issues
        if diagnostics['trend_mismatch']['detected']:
            st.write("❌ Trend Mismatch")
        if diagnostics['missing_seasonality']['detected']:
            st.write("❌ Missing Seasonality")
        if diagnostics['volatility_mismatch']['detected']:
            st.write("❌ Too Flat")
        if diagnostics['magnitude_mismatch']['detected']:
            st.write("❌ Magnitude Mismatch")
        
        if total_issues == 0:
            st.write("✅ No issues detected")
    
    # === SECTION 2: ANALYSIS REPORT ===
    analysis_icons = {"llm": "🤖", "basic": "📄"}
    st.subheader(f"{analysis_icons.get(analysis_method, '📄')} Analysis Report")
    st.info(explanation)
    
    # === SECTION 3: AUTO-RETRAINING ===
    if total_issues > 0 and AUTO_RETRAIN_AVAILABLE:
        st.subheader("🔄 Automatic Model Retraining")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🚀 Auto-Retrain Models", key=f"retrain_{selected_item}"):
                with st.spinner("Testing HoltWinters, Holt, and AutoETS..."):
                    try:
                        # Initialize basic retrain agent
                        retrain_agent = BasicRetrainAgent()
                        
                        # Run basic retraining (always tests same 3 models)
                        retrain_results = retrain_agent.retrain_and_select(
                            historical_data=historical_data,
                            original_forecast=forecast_data,
                            item_id=selected_item
                        )
                        
                        # Store results in session state for visualization
                        st.session_state[f'retrain_results_{selected_item}'] = retrain_results
                        
                        if retrain_results['success']:
                            st.success(f"✅ Best model: **{retrain_results['best_model']}** (Accuracy: {retrain_results['best_accuracy']:.3f})")
                            
                            # Show tested models
                            tested_models = retrain_results.get('tested_models', [])
                            st.info(f"🔄 Tested models: {', '.join(tested_models)}")
                        else:
                            error_msg = retrain_results.get('error', 'Unknown error')
                            st.error(f"❌ Retraining failed: {error_msg}")
                            
                            # Show detailed error information
                            with st.expander("🔍 Debug Information"):
                                tested_models = retrain_results.get('tested_models', [])
                                
                                st.write(f"**Models Tested:** {', '.join(tested_models) if tested_models else 'None'}")
                                
                                st.write("**Available Models:**")
                                for model_name in ['HoltWinters', 'Holt', 'AutoETS']:
                                    st.write(f"- {model_name}")
                                
                                st.write("**Test Results:**")
                                for result in retrain_results.get('all_results', []):
                                    model = result.get('model', 'Unknown')
                                    success = result.get('success', False)
                                    error = result.get('error', 'N/A')
                                    st.write(f"- {model}: {'✅' if success else '❌'} {error if not success else 'Success'}")
                            
                    except Exception as e:
                        st.error(f"❌ Retraining error: {str(e)}")
        
        with col2:
            st.info("🤖 This will automatically test HoltWinters, Holt, and AutoETS models and select the most accurate one.")
    
    # === SECTION 4: BEFORE/AFTER VISUALIZATION ===
    if f'retrain_results_{selected_item}' in st.session_state:
        retrain_results = st.session_state[f'retrain_results_{selected_item}']
        
        if retrain_results.get('success', False):
            st.subheader("📊 Before vs After Comparison")
            
            # Create simple before/after plot
            best_forecast = retrain_results.get('best_forecast', [])
            best_model = retrain_results.get('best_model', 'Unknown')
            
            if len(best_forecast) > 0:
                fig = plot_simple_before_after(
                    historical_data=historical_data,
                    original_forecast=forecast_data,
                    new_forecast=np.array(best_forecast),
                    model_name=best_model,
                    item_id=selected_item
                )
                
                st.pyplot(fig)
                plt.close()
            
            # Show simple metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy = retrain_results.get('best_accuracy', 0)
                st.metric("Best Model Accuracy", f"{accuracy:.3f}")
            
            with col2:
                tested_models = retrain_results.get('tested_models', [])
                st.metric("Models Tested", len(tested_models))
            
            with col3:
                best_model = retrain_results.get('best_model', 'Unknown')
                st.metric("Selected Model", best_model)
            
            # Clear results button
            if st.button("🗑️ Clear Results", key=f"clear_{selected_item}"):
                del st.session_state[f'retrain_results_{selected_item}']
                st.rerun()
    
    # === SECTION 5: FEEDBACK ===
    st.subheader("📝 Feedback")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rating = st.slider("Rate this analysis:", 1, 5, 3, key=f"rating_{selected_item}")
    
    with col2:
        preferred_model = st.selectbox(
            "Preferred model:",
            ["None", "HoltWinters", "Holt", "AutoETS"],
            key=f"model_{selected_item}"
        )
    
    with col3:
        if st.button("Submit", key=f"submit_{selected_item}"):
            # Simple feedback collection (could be extended later)
            st.success("✅ Thanks for your feedback!")


def display_batch_summary(df):
    """Display simplified batch summary table."""
    st.header("📊 Part Summary")
    
    if not BATCH_AVAILABLE:
        st.error("Analysis modules not available.")
        return
    
    # Run batch processing
    with st.spinner("Analyzing parts..."):
        results, summary_stats = run_batch_diagnostics(df)
    
    if results is None:
        st.error("Analysis failed.")
        return
    
    # Show simple metrics
    total_parts = summary_stats.get('total_parts', 0)
    flagged_parts = summary_stats.get('flagged_parts', 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parts", total_parts)
    with col2:
        st.metric("Parts with Issues", flagged_parts)
    with col3:
        pct = (flagged_parts / total_parts * 100) if total_parts > 0 else 0
        st.metric("Issue Rate", f"{pct:.1f}%")
    
    st.markdown("---")
    
    # Create simple summary table
    create_simple_batch_table(results, df)


def main():
    st.set_page_config(page_title="Forecast Monitor Agent", layout="wide")
    
    st.title("📊 Forecast Monitor Agent")
    st.markdown("**Simple forecast analysis - Click any part to see details**")
    
    # Minimal sidebar
    st.sidebar.markdown("### Quick Info")
    st.sidebar.info("💡 Click on any part in the table to see detailed analysis")
    
    # Load data
    try:
        df = load_data("data/data.csv")
        
        # Display the simplified batch summary
        display_batch_summary(df)
    
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/data.csv' exists.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


if __name__ == "__main__":
    main()