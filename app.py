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
from modules.loader import load_data, get_item_data, get_all_item_ids, get_recent_actuals, get_early_forecast
from modules.diagnostics import run_all_diagnostics
from modules.visualizer import plot_forecast_analysis, create_summary_stats_table
from modules.explainer_simple import (prepare_analysis_summary, generate_explanation, 
                                     format_explanation_report, check_llm_availability, 
                                     get_preferred_provider)
from modules.reporter import create_detailed_report


def main():
    st.set_page_config(page_title="Forecast Monitor Agent", layout="wide")
    
    st.title("🔍 Forecast Monitor Agent")
    st.markdown("Intelligent detection and explanation of forecast alignment issues")
    
    # Load data
    try:
        df = load_data("data/data.csv")
        item_ids = get_all_item_ids(df)
        
        st.sidebar.header("Configuration")
        selected_item = st.sidebar.selectbox("Select Item ID", item_ids)
        
        # LLM Configuration
        st.sidebar.subheader("🤖 LLM Settings")
        
        # Debug information
        import os
        import sys
        sys.path.append('modules')
        st.sidebar.write("**Debug Info:**")
        
        # Show current working directory and python path
        st.sidebar.write(f"Working Dir: {os.getcwd()}")
        st.sidebar.write(f"Python Path: {len(sys.path)} entries")
        
        # Check API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            st.sidebar.write(f"API Key: Set ({api_key[:8]}...)")
        else:
            st.sidebar.write("API Key: Not set")
        
        # Test dotenv loading explicitly
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key_after_dotenv = os.getenv("ANTHROPIC_API_KEY")
            if api_key_after_dotenv:
                st.sidebar.write(f"After dotenv: Set ({api_key_after_dotenv[:8]}...)")
            else:
                st.sidebar.write("After dotenv: Not set")
        except Exception as e:
            st.sidebar.write(f"Dotenv error: {e}")
        
        # Test anthropic import
        try:
            import anthropic
            st.sidebar.write(f"Anthropic: ✅ v{anthropic.__version__}")
        except ImportError as e:
            st.sidebar.write(f"Anthropic: ❌ {e}")
        
        # Import LLM client with detailed error reporting
        try:
            from modules.llm_client_robust import get_available_providers as direct_get_providers
            available_providers = direct_get_providers()
            st.sidebar.write("LLM Import: ✅")
        except Exception as e:
            st.sidebar.error(f"LLM import error: {e}")
            import traceback
            st.sidebar.text(traceback.format_exc())
            available_providers = {"claude": False, "openai": False}
        
        st.sidebar.write("Providers:", available_providers)
        
        if any(available_providers.values()):
            # Show available providers
            available_list = [p for p, available in available_providers.items() if available]
            preferred = "claude" if available_providers.get("claude", False) else available_list[0]
            default_idx = available_list.index(preferred) if preferred in available_list else 0
            
            llm_provider = st.sidebar.selectbox("AI Provider", available_list, 
                                              index=default_idx,
                                              help="Select AI provider for generating explanations")
            use_llm = True
        else:
            st.sidebar.error("❌ No AI providers available. Please check your API keys.")
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
    
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/data.csv' exists.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


if __name__ == "__main__":
    main()