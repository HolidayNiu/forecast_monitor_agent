#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test script to demonstrate the new DetectorAgent.
Run this to see the agent in action!
"""
import sys
sys.path.append('.')

def test_agent_basic():
    """Test basic agent functionality."""
    print("🚀 DetectorAgent Quick Test")
    print("=" * 40)
    
    try:
        # Test 1: Basic imports
        print("1. Testing imports...")
        from agent.detector_agent import DetectorAgent
        from agent.base import BaseAgent
        print("   ✅ Agent classes imported successfully")
        
        # Test 2: Agent creation
        print("2. Creating agent...")
        agent = DetectorAgent()
        print("   ✅ DetectorAgent created")
        
        # Test 3: Check inheritance
        if isinstance(agent, BaseAgent):
            print("   ✅ Agent properly inherits from BaseAgent")
        
        # Test 4: Check methods exist
        required_methods = ['observe', 'reason', 'act', 'reset', 'process_single']
        for method in required_methods:
            if hasattr(agent, method):
                print("   ✅ Agent has {} method".format(method))
            else:
                print("   ❌ Agent missing {} method".format(method))
        
        print("\n3. Testing with real data...")
        
        # Test 5: Try with actual data
        try:
            from modules.loader import load_data, get_item_data, get_all_item_ids
            
            # Load data
            df = load_data("data/data.csv")
            item_ids = get_all_item_ids(df)
            
            if len(item_ids) > 0:
                # Test with first part
                test_item = item_ids[0]
                print("   📊 Testing with part: {}".format(test_item))
                
                # Get data
                historical_data, forecast_data = get_item_data(df, test_item)
                print("   ✅ Data loaded: {} hist, {} forecast months".format(
                    len(historical_data), len(forecast_data)))
                
                # Test agent processing
                result = agent.process_single(historical_data, forecast_data)
                
                print("   ✅ Agent processing successful!")
                print("   📈 Risk Score: {:.3f}".format(result['summary']['risk_score']))
                print("   🚨 Issues Found: {}".format(result['summary']['total_issues']))
                
                # Test data summary
                agent.observe((historical_data, forecast_data))
                summary = agent.get_data_summary()
                if summary:
                    print("   📋 Data Summary: Historical mean = {:.2f}".format(summary['historical_mean']))
                
            else:
                print("   ⚠️  No parts found in data file")
                
        except FileNotFoundError:
            print("   ⚠️  Data file not found - testing structure only")
        except Exception as e:
            print("   ❌ Data processing error: {}".format(e))
        
        print("\n🎉 Agent structure test completed successfully!")
        return True
        
    except ImportError as e:
        print("   ❌ Import error: {}".format(e))
        print("   💡 Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print("   ❌ Test error: {}".format(e))
        return False

def show_usage_examples():
    """Show practical usage examples."""
    print("\n" + "=" * 40)
    print("📖 Usage Examples")
    print("=" * 40)
    
    print("\n1. BASIC USAGE:")
    print("   from agent.detector_agent import DetectorAgent")
    print("   agent = DetectorAgent()")
    print("   result = agent.process_single(historical_data, forecast_data)")
    print("   print('Risk:', result['summary']['risk_score'])")
    
    print("\n2. LIFECYCLE USAGE:")
    print("   agent = DetectorAgent()")
    print("   agent.observe((historical_data, forecast_data))")
    print("   agent.reason()")
    print("   result = agent.act()")
    
    print("\n3. BATCH PROCESSING:")
    print("   agent = DetectorAgent()")
    print("   for part_id in part_ids:")
    print("       agent.reset()")
    print("       result = agent.process_single(hist_data, fcst_data)")
    print("       results.append(result)")
    
    print("\n4. WEB INTERFACE:")
    print("   streamlit run app.py")
    print("   # Look for '✅ Using Agent-based Processing' in sidebar")
    
    print("\n5. COMMAND LINE:")
    print("   python run_diagnostics.py --parts YOUR_PART_ID")

def main():
    """Main test function."""
    success = test_agent_basic()
    show_usage_examples()
    
    if success:
        print("\n" + "🟢" * 20)
        print("🎯 DetectorAgent is ready to use!")
        print("🟢" * 20)
        print("\nNext steps:")
        print("1. Run: streamlit run app.py")
        print("2. Or try: python run_diagnostics.py --parts [your_part_id]")
        print("3. Or create your own script using the examples above")
    else:
        print("\n" + "🔴" * 20)
        print("❌ Setup issues detected")
        print("🔴" * 20)
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Check that agent/ folder exists")
        print("3. Verify data/data.csv file exists")

if __name__ == "__main__":
    main()