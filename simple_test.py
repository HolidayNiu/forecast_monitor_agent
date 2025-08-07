# Test to verify flexible data length implementation
import sys
sys.path.append('modules')

# Test loader module changes
try:
    from modules.loader import get_item_data
    print("SUCCESS: loader module imported")
    
    # Test function signature
    import inspect
    sig = inspect.signature(get_item_data)
    params = list(sig.parameters.keys())
    print("Function parameters: {}".format(params))
    
    if 'min_historical_months' in params and 'min_forecast_months' in params:
        print("SUCCESS: New flexible parameters found in loader")
    else:
        print("ERROR: Missing flexible parameters in loader")
        
except Exception as e:
    print("ERROR with loader: {}".format(e))

# Mock the pandas functionality for testing
class MockSeries:
    def __init__(self, data):
        self.data = data
        self.values = data
    
    def __len__(self):
        return len(self.data)
    
    def head(self, n=5):
        return MockSeries(self.data[:n])
    
    def tail(self, n=5):
        return MockSeries(self.data[-n:])
    
    def tolist(self):
        return self.data
    
    def mean(self):
        return sum(self.data) / len(self.data)
    
    def std(self):
        mean_val = self.mean()
        variance = sum((x - mean_val) ** 2 for x in self.data) / len(self.data)
        return variance ** 0.5
    
    def min(self):
        return min(self.data)
    
    def max(self):
        return max(self.data)

# Test basic diagnostic logic without pandas
def test_basic_logic():
    print("Testing basic forecast monitoring logic...")
    
    # Mock historical data (36 months) - showing seasonal pattern and trend
    historical = [100, 120, 110, 90, 80, 85, 105, 125, 115, 95, 85, 90,  # Year 1
                 110, 130, 120, 100, 90, 95, 115, 135, 125, 105, 95, 100,  # Year 2
                 120, 140, 130, 110, 100, 105, 125, 145, 135, 115, 105, 110]  # Year 3
    
    # Mock forecast data (18 months) - flat line, missing seasonality
    forecast = [112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112]
    
    hist_series = MockSeries(historical)
    forecast_series = MockSeries(forecast)
    
    print("Historical data length: {}".format(len(hist_series)))
    print("Forecast data length: {}".format(len(forecast_series)))
    print("Historical mean: {:.2f}".format(hist_series.mean()))
    print("Forecast mean: {:.2f}".format(forecast_series.mean()))
    print("Historical std: {:.2f}".format(hist_series.std()))
    print("Forecast std: {:.2f}".format(forecast_series.std()))
    
    # Test volatility mismatch detection
    hist_cv = hist_series.std() / hist_series.mean()
    forecast_cv = forecast_series.std() / forecast_series.mean()
    volatility_ratio = forecast_cv / hist_cv
    
    print("Historical CV: {:.3f}".format(hist_cv))
    print("Forecast CV: {:.3f}".format(forecast_cv))
    print("Volatility ratio: {:.3f}".format(volatility_ratio))
    
    if volatility_ratio < 0.5:
        print("DETECTED: Volatility mismatch - forecast too flat")
    else:
        print("No volatility mismatch detected")
    
    # Test magnitude mismatch
    recent_actuals = hist_series.tail(6)
    early_forecast = forecast_series.head(6)
    
    recent_mean = recent_actuals.mean()
    forecast_mean = early_forecast.mean()
    pct_diff = abs(forecast_mean - recent_mean) / recent_mean
    
    print("Recent actuals mean: {:.2f}".format(recent_mean))
    print("Early forecast mean: {:.2f}".format(forecast_mean))
    print("Percentage difference: {:.1f}%".format(pct_diff*100))
    
    if pct_diff > 0.5:
        print("DETECTED: Magnitude mismatch")
    else:
        print("No magnitude mismatch detected")
    
    print("\nBasic logic test completed!")

if __name__ == "__main__":
    test_basic_logic()