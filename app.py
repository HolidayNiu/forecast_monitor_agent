"""
Streamlit app for forecast monitoring agent.
"""
# Load environment variables first
try:
    from dotenv import load_dotenv
    load_dotenv()
    # Verify API key is loaded
    import os
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print(f"✅ API key loaded: {api_key[:20]}...{api_key[-10:]}")
    else:
        print("❌ No API key found in environment")
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
<<<<<<< HEAD

from agent.core.loader import load_data, get_item_data, get_all_item_ids
from agent.utils.visualizer import plot_forecast_analysis, create_summary_stats_table
from agent.utils.reporter import create_detailed_report

# Try to import DetectorAgent for agent-based processing
try:
    from agent.agents.detector_agent import DetectorAgent, run_simple_batch_diagnostics
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# Try to import IntelligentExplainerAgent
try:
    from agent.agents.intelligent_explainer_agent import IntelligentExplainerAgent
    INTELLIGENT_EXPLAINER_AVAILABLE = True
except ImportError:
    INTELLIGENT_EXPLAINER_AVAILABLE = False

# Try to import BasicRetrainAgent
try:
    from agent.agents.basic_retrain_agent import BasicRetrainAgent
=======

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
>>>>>>> origin/main
    AUTO_RETRAIN_AVAILABLE = True
except ImportError:
    AUTO_RETRAIN_AVAILABLE = False

<<<<<<< HEAD


def _create_issue_summary(diagnostics):
    """Create human-readable issue summary."""
    issues = []
    if diagnostics['trend_mismatch']['detected']:
        issues.append('Trend mismatch')
    if diagnostics['missing_seasonality']['detected']:
        issues.append('Missing seasonality')
    if diagnostics['volatility_mismatch']['detected']:
        issues.append('Too flat')
    if diagnostics['magnitude_mismatch']['detected']:
        issues.append('Magnitude mismatch')
    
    if not issues:
        return 'No issues detected'
    else:
        return ', '.join(issues)


def create_simple_batch_table(results, df):
    """Create a simple table showing all parts with their summary information."""
    if not results:
        st.info("No results to display.")
        return
    
=======
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
    
>>>>>>> origin/main
    # Create display data for all parts
    display_data = []
    for result in results:
        part_id = result.get('part_id')
        
        # Get additional data from original dataframe
        part_data = df[df['item_loc_id'] == part_id].iloc[0] if len(df[df['item_loc_id'] == part_id]) > 0 else None
<<<<<<< HEAD
        
        if part_data is not None:
            # Convert pandas Series to dict access
            best_model = part_data['best_model'] if 'best_model' in part_data.index else 'Unknown'
            display_data.append({
                'Part Number': part_id,
                'Current Model': best_model,
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
        
        # Display detailed analysis for selected part
        st.divider()
        display_intelligent_part_analysis(df, selected_part)


def display_intelligent_part_analysis(df, selected_item):
    """Display intelligent analysis for selected part using the full agent architecture."""
    
    # Initialize intelligent explainer
    intelligent_explainer = None
    if INTELLIGENT_EXPLAINER_AVAILABLE:
        try:
            intelligent_explainer = IntelligentExplainerAgent()
        except Exception as e:
            st.error(f"Failed to initialize intelligent explainer: {e}")
    
    # Auto-detect LLM provider
    try:
        from agent.llm.llm_client_robust import get_preferred_provider, get_available_providers
        llm_provider = get_preferred_provider()
        if llm_provider:
            use_llm = True
            available_providers = get_available_providers()
            st.sidebar.success(f"🤖 LLM: {llm_provider.title()} (Auto-detected)")
            
            # Test API connection
            if llm_provider == "claude":
                import os
                api_key = os.getenv('ANTHROPIC_API_KEY')
                st.sidebar.info(f"API Key: {api_key[:15]}...{api_key[-5:] if api_key else 'None'}")
                
                # Quick test call
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)
                    test_response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=5,
                        messages=[{"role": "user", "content": "Hi"}]
                    )
                    st.sidebar.success("✅ API connection verified")
                except Exception as test_e:
                    st.sidebar.error(f"❌ API test failed: {str(test_e)}")
            
            # Show available providers for info
            provider_list = [k for k, v in available_providers.items() if v]
            if len(provider_list) > 1:
                st.sidebar.info(f"Available: {', '.join(provider_list)}")
        else:
            use_llm = False
            st.sidebar.warning("🤖 LLM: None available")
            st.sidebar.info("Set ANTHROPIC_API_KEY, DATABRICKS_TOKEN/HOST, or OPENAI_API_KEY")
    except Exception as e:
        st.sidebar.error(f"LLM detection failed: {e}")
        use_llm = False
        llm_provider = None
    
    # Load data and run diagnostics
    historical_data, forecast_data = get_item_data(df, selected_item)
    
    if AGENT_AVAILABLE:
        try:
            agent = DetectorAgent()
            diagnostics = agent.process_single(historical_data, forecast_data)
        except:
            from agent.core.diagnostics import run_all_diagnostics
            from agent.core.loader import get_recent_actuals, get_early_forecast
            recent_actuals = get_recent_actuals(historical_data)
            early_forecast = get_early_forecast(forecast_data)
            diagnostics = run_all_diagnostics(historical_data, forecast_data, recent_actuals, early_forecast)
    else:
        from agent.core.diagnostics import run_all_diagnostics
        from agent.core.loader import get_recent_actuals, get_early_forecast
        recent_actuals = get_recent_actuals(historical_data)
        early_forecast = get_early_forecast(forecast_data)
        diagnostics = run_all_diagnostics(historical_data, forecast_data, recent_actuals, early_forecast)
    
    # Generate intelligent explanation and recommendations
    model_recommendations = []
    analysis_method = "basic"
    
    if intelligent_explainer and use_llm and llm_provider:
        try:
            with st.spinner("🤖 Generating AI-powered analysis..."):
                intelligent_report = intelligent_explainer.generate_intelligent_explanation(
                    selected_item, diagnostics, historical_data, forecast_data, llm_provider, df
                )
            explanation = intelligent_report['explanation']
            model_recommendations = intelligent_report.get('model_recommendations', [])
            analysis_method = "intelligent"
        except Exception as e:
            st.error(f"❌ AI Analysis Error: {str(e)}")
            explanation = "⚠️ **LLM Analysis Failed** - Using basic rule-based analysis instead."
            analysis_method = "basic"
    elif not intelligent_explainer:
        explanation = "⚠️ **Intelligent Explainer Not Available** - Check imports and dependencies."
        analysis_method = "basic"
    elif not use_llm or not llm_provider:
        explanation = "⚠️ **LLM Not Available** - Check API keys (ANTHROPIC_API_KEY, DATABRICKS_TOKEN, OPENAI_API_KEY)."
        analysis_method = "basic"
    else:
        # Generate informative fallback explanation
        detected_issues = []
        if diagnostics['trend_mismatch']['detected']:
            detected_issues.append('trend mismatch')
        if diagnostics['missing_seasonality']['detected']:
            detected_issues.append('missing seasonality')
        if diagnostics['volatility_mismatch']['detected']:
            detected_issues.append('volatility issues (too flat)')
        if diagnostics['magnitude_mismatch']['detected']:
            detected_issues.append('magnitude mismatch')
        
        if detected_issues:
            explanation = f"Issues detected: {', '.join(detected_issues)}. These problems may affect forecast accuracy and business planning. Consider retraining with models that better handle these specific patterns."
        else:
            explanation = "No significant issues detected with the current forecast. The model appears to be performing well."
        
        analysis_method = "basic"
    
    # === SECTION 1: ISSUE SUMMARY WITH VISUAL ===
    st.subheader(f"📊 {selected_item} - Issue Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series plot
        fig = plot_forecast_analysis(historical_data, forecast_data, selected_item, diagnostics)
        st.plotly_chart(fig, use_container_width=True)
    
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
=======
        
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
>>>>>>> origin/main
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
    
<<<<<<< HEAD
    # === SECTION 2: INTELLIGENT ANALYSIS REPORT ===
    analysis_icons = {"intelligent": "🧠", "basic": "📄"}
    st.subheader(f"{analysis_icons.get(analysis_method, '📄')} Analysis Report")
    
    # Show LLM-based intelligent analysis
    if analysis_method == "intelligent" and explanation:
        st.markdown("### 🤖 **AI-Generated Analysis Report**")
        st.info(explanation)
    elif explanation:
        st.markdown("### 📄 **Basic Analysis Report**") 
        st.info(explanation)
    else:
        st.warning("Analysis report generation failed. Please check LLM connection.")
    
    # === SECTION 3: MODEL RECOMMENDATIONS ===
    if model_recommendations:
        st.subheader("🎯 Model Recommendations")
        for i, rec in enumerate(model_recommendations, 1):
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            st.write(f"{i}. {priority_icon.get(rec['priority'], '⚪')} **{rec['model']}** - {rec['rationale']}")
    
    # === SECTION 4: AUTO-RETRAINING (SIMPLIFIED) ===
=======
    # === SECTION 2: ANALYSIS REPORT ===
    analysis_icons = {"llm": "🤖", "basic": "📄"}
    st.subheader(f"{analysis_icons.get(analysis_method, '📄')} Analysis Report")
    st.info(explanation)
    
    # === SECTION 3: AUTO-RETRAINING ===
>>>>>>> origin/main
    if total_issues > 0 and AUTO_RETRAIN_AVAILABLE:
        st.subheader("🔄 Automatic Model Retraining")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🚀 Auto-Retrain Models", key=f"retrain_{selected_item}"):
                with st.spinner("Testing HoltWinters, Holt, and AutoETS..."):
                    try:
<<<<<<< HEAD
                        # Initialize basic retrain agent (always uses same 3 models)
                        retrain_agent = BasicRetrainAgent()
                        
                        # Run retraining
=======
                        # Initialize basic retrain agent
                        retrain_agent = BasicRetrainAgent()
                        
                        # Run basic retraining (always tests same 3 models)
>>>>>>> origin/main
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
                            
<<<<<<< HEAD
                            # Show debug info
                            with st.expander("🔍 Debug Information"):
                                tested_models = retrain_results.get('tested_models', [])
                                st.write(f"**Models Tested:** {', '.join(tested_models) if tested_models else 'None'}")
                                
=======
                            # Show detailed error information
                            with st.expander("🔍 Debug Information"):
                                tested_models = retrain_results.get('tested_models', [])
                                
                                st.write(f"**Models Tested:** {', '.join(tested_models) if tested_models else 'None'}")
                                
                                st.write("**Available Models:**")
                                for model_name in ['HoltWinters', 'Holt', 'AutoETS']:
                                    st.write(f"- {model_name}")
                                
                                st.write("**Test Results:**")
>>>>>>> origin/main
                                for result in retrain_results.get('all_results', []):
                                    model = result.get('model', 'Unknown')
                                    success = result.get('success', False)
                                    error = result.get('error', 'N/A')
                                    st.write(f"- {model}: {'✅' if success else '❌'} {error if not success else 'Success'}")
                            
                    except Exception as e:
                        st.error(f"❌ Retraining error: {str(e)}")
        
        with col2:
            st.info("🤖 This will automatically test HoltWinters, Holt, and AutoETS models and select the most accurate one.")
    
<<<<<<< HEAD
    # === SECTION 5: BEFORE/AFTER VISUALIZATION ===
=======
    # === SECTION 4: BEFORE/AFTER VISUALIZATION ===
>>>>>>> origin/main
    if f'retrain_results_{selected_item}' in st.session_state:
        retrain_results = st.session_state[f'retrain_results_{selected_item}']
        
        if retrain_results.get('success', False):
            st.subheader("📊 Before vs After Comparison")
            
            # Create simple before/after plot
            best_forecast = retrain_results.get('best_forecast', [])
            best_model = retrain_results.get('best_model', 'Unknown')
<<<<<<< HEAD
            
            if len(best_forecast) > 0:
                # Store retrain forecast in session state to show in plot
                if 'retrain_forecast' not in st.session_state:
                    st.session_state.retrain_forecast = {}
                st.session_state.retrain_forecast[selected_item] = best_forecast
                
                # Show updated plot with retrained forecast
                fig_retrain = plot_forecast_analysis(historical_data, forecast_data, selected_item, diagnostics)
                st.plotly_chart(fig_retrain, use_container_width=True)
            
            # Show simple metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy = retrain_results.get('best_accuracy', 0)
                st.metric("Best Model Accuracy", f"{accuracy:.3f}")
            
            with col2:
                tested_models = retrain_results.get('tested_models', [])
                st.metric("Models Tested", len(tested_models))
            
=======
            
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
            
>>>>>>> origin/main
            with col3:
                best_model = retrain_results.get('best_model', 'Unknown')
                st.metric("Selected Model", best_model)
            
            # Clear results button
            if st.button("🗑️ Clear Results", key=f"clear_{selected_item}"):
                del st.session_state[f'retrain_results_{selected_item}']
                st.rerun()
    
<<<<<<< HEAD
    # === SECTION 6: FEEDBACK ===
=======
    # === SECTION 5: FEEDBACK ===
>>>>>>> origin/main
    st.subheader("📝 Feedback")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rating = st.slider("Rate this analysis:", 1, 5, 3, key=f"rating_{selected_item}")
    
    with col2:
<<<<<<< HEAD
        if model_recommendations:
            preferred_model = st.selectbox(
                "Preferred model:",
                ["None"] + [rec['model'] for rec in model_recommendations],
                key=f"model_{selected_item}"
            )
        else:
            preferred_model = st.selectbox(
                "Preferred model:",
                ["None", "HoltWinters", "Holt", "AutoETS"],
                key=f"model_{selected_item}"
            )
    
    with col3:
        if st.button("Submit", key=f"submit_{selected_item}"):
            if intelligent_explainer:
                try:
                    success = intelligent_explainer.collect_user_feedback(
                        selected_item, f"{selected_item}_analysis", 
                        "overall", rating, "", 
                        preferred_model if preferred_model != "None" else None
                    )
                    if success:
                        st.success("✅ Thanks for your feedback!")
                    else:
                        st.error("❌ Failed to save feedback")
                except:
                    st.error("❌ Failed to save feedback")
            else:
                st.success("✅ Thanks for your feedback!")
=======
        preferred_model = st.selectbox(
            "Preferred model:",
            ["None", "HoltWinters", "Holt", "AutoETS"],
            key=f"model_{selected_item}"
        )
    
    with col3:
        if st.button("Submit", key=f"submit_{selected_item}"):
            # Simple feedback collection (could be extended later)
            st.success("✅ Thanks for your feedback!")
>>>>>>> origin/main


def display_batch_summary(df):
    """Display simplified batch summary table."""
    st.header("📊 Part Summary")
    
<<<<<<< HEAD
    # Run simple batch processing
    with st.spinner("Analyzing parts..."):
        results, summary_stats = run_simple_batch_diagnostics(df)
=======
    if not BATCH_AVAILABLE:
        st.error("Analysis modules not available.")
        return
    
    # Run batch processing
    with st.spinner("Analyzing parts..."):
        results, summary_stats = run_batch_diagnostics(df)
>>>>>>> origin/main
    
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
<<<<<<< HEAD
    st.markdown("**Intelligent forecast analysis - Click any part to see details**")
    
    # Minimal sidebar
    st.sidebar.markdown("### System Status")
    st.sidebar.info(f"🔍 Detector: {'✅' if AGENT_AVAILABLE else '❌'}")
    st.sidebar.info(f"🧠 Explainer: {'✅' if INTELLIGENT_EXPLAINER_AVAILABLE else '❌'}")
    st.sidebar.info(f"🔄 Retrain: {'✅' if AUTO_RETRAIN_AVAILABLE else '❌'}")
    st.sidebar.info("📊 Batch: ✅")  # Always available now
=======
    st.markdown("**Simple forecast analysis - Click any part to see details**")
    
    # Minimal sidebar
    st.sidebar.markdown("### Quick Info")
    st.sidebar.info("💡 Click on any part in the table to see detailed analysis")
>>>>>>> origin/main
    
    # Load data
    try:
        df = load_data("data/data.csv")
        
<<<<<<< HEAD
        # Display the batch summary with intelligent analysis
=======
        # Display the simplified batch summary
>>>>>>> origin/main
        display_batch_summary(df)
    
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/data.csv' exists.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


if __name__ == "__main__":
    main()