"""
DetectorAgent for wrapping diagnostics logic in an agent-based structure.
"""
import pandas as pd
try:
    from typing import Dict, Any, Optional, Tuple
except ImportError:
    # Python 2.7 compatibility
    Dict = dict
    Any = object
    Optional = object
    Tuple = tuple

from .base import BaseAgent
from modules.diagnostics import run_all_diagnostics
from modules.loader import get_recent_actuals, get_early_forecast


class DetectorAgent(BaseAgent):
    """
    Agent that wraps the forecast diagnostics logic.
    
    This agent follows the observe -> reason -> act lifecycle:
    - observe(): Store the input data (historical and forecast data)
    - reason(): Apply diagnostic functions to detect forecast issues
    - act(): Return the diagnostic results with risk scores
    """
    
    def __init__(self, thresholds=None):
        """
        Initialize the DetectorAgent.
        
        Args:
            thresholds: Optional dictionary of thresholds for diagnostics.
                       Currently not used by the underlying diagnostics functions,
                       but preserved for future extensibility.
        """
        super().__init__()
        self.thresholds = thresholds or {}
        
        # Data storage for the agent lifecycle
        self._historical_data = None
        self._forecast_data = None
        self._recent_actuals = None
        self._early_forecast = None
    
    def observe(self, data):
        """
        Observe and store the input forecast data.
        
        Args:
            data: Tuple of (historical_data, forecast_data) as pandas Series
        """
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("Data must be a tuple of (historical_data, forecast_data)")
        
        historical_data, forecast_data = data
        
        if not isinstance(historical_data, pd.Series) or not isinstance(forecast_data, pd.Series):
            raise ValueError("Both historical_data and forecast_data must be pandas Series")
        
        # Store the main data
        self._historical_data = historical_data
        self._forecast_data = forecast_data
        
        # Prepare derived data for diagnostics
        self._recent_actuals = get_recent_actuals(historical_data)
        self._early_forecast = get_early_forecast(forecast_data)
        
        # Store in base class for consistency
        self._data = data
    
    def reason(self):
        """
        Apply diagnostic reasoning to the observed data.
        
        This method runs all diagnostic tests on the stored data
        and prepares the results for the act() method.
        """
        if self._historical_data is None or self._forecast_data is None:
            raise RuntimeError("No data observed. Call observe() first.")
        
        # Run all diagnostics using the existing function
        self._reasoning_result = run_all_diagnostics(
            historical_data=self._historical_data,
            forecast_data=self._forecast_data,
            recent_actuals=self._recent_actuals,
            early_forecast=self._early_forecast
        )
    
    def act(self):
        """
        Return the diagnostic results.
        
        Returns:
            Dictionary containing diagnostic results with the same structure
            as the original run_all_diagnostics() function:
            - trend_mismatch: trend analysis results
            - missing_seasonality: seasonality analysis results  
            - volatility_mismatch: volatility analysis results
            - magnitude_mismatch: magnitude analysis results
            - summary: overall risk score and statistics
        """
        if self._reasoning_result is None:
            raise RuntimeError("No reasoning performed. Call reason() first.")
        
        return self._reasoning_result
    
    def reset(self):
        """
        Reset the agent's memory for processing the next part.
        
        Clears all stored data and results, allowing the agent to be
        reused in loops for processing multiple parts.
        """
        super().reset()
        self._historical_data = None
        self._forecast_data = None
        self._recent_actuals = None
        self._early_forecast = None
    
    def process_single(self, historical_data, forecast_data):
        """
        Convenience method to process a single part through the full lifecycle.
        
        Args:
            historical_data: Historical data as pandas Series
            forecast_data: Forecast data as pandas Series
            
        Returns:
            Dictionary containing diagnostic results
        """
        self.reset()
        self.observe((historical_data, forecast_data))
        self.reason()
        return self.act()
    
    def get_data_summary(self):
        """
        Get a summary of the currently observed data.
        
        Returns:
            Dictionary with data statistics, or None if no data observed
        """
        if self._historical_data is None or self._forecast_data is None:
            return None
        
        return {
            'historical_length': len(self._historical_data),
            'forecast_length': len(self._forecast_data),
            'historical_mean': float(self._historical_data.mean()),
            'forecast_mean': float(self._forecast_data.mean()),
            'historical_std': float(self._historical_data.std()),
            'forecast_std': float(self._forecast_data.std())
        }