"""
Basic Retrain Agent - Ultra Simple
When issues detected: Test HoltWinters, ETS, AutoETS -> Select best -> Done
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Nixtla imports
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, Holt, HoltWinters
    from utilsforecast.losses import mse, mae
    NIXTLA_AVAILABLE = True
except ImportError:
    NIXTLA_AVAILABLE = False
    logging.warning("Nixtla not available.")

logger = logging.getLogger(__name__)


class BasicRetrainAgent:
    """Ultra simple: Always test HoltWinters, ETS, AutoETS -> Pick best accuracy"""
    
    def __init__(self):
        # Fixed 3 models only
        if NIXTLA_AVAILABLE:
            self.models = {
                'HoltWinters': HoltWinters,
                'Holt': Holt,  # ETS equivalent
                'AutoETS': AutoETS
            }
        else:
            self.models = {}
    
    def prepare_data(self, historical_data, item_id):
        """Prepare data for StatsForecast."""
        # Create monthly dates
        if hasattr(historical_data, 'index') and pd.api.types.is_datetime64_any_dtype(historical_data.index):
            dates = historical_data.index
        else:
            start_date = pd.Timestamp('2020-01-01')
            dates = pd.date_range(start=start_date, periods=len(historical_data), freq='MS')
        
        return pd.DataFrame({
            'unique_id': item_id,
            'ds': dates,
            'y': historical_data.values
        })
    
    def test_single_model(self, model_name, train_data, test_data, forecast_horizon):
        """Test one model and return results."""
        if not NIXTLA_AVAILABLE:
            return {'model': model_name, 'success': False, 'error': 'Nixtla not available'}
        
        try:
            # Create model instance
            model_class = self.models[model_name]
            if model_name == 'HoltWinters':
                model = model_class(season_length=12)
            elif model_name == 'Holt':
                model = model_class()
            else:  # AutoETS
                model = model_class(season_length=12)
            
            # Create StatsForecast and fit
            sf = StatsForecast(models=[model], freq='MS')
            forecast_df = sf.forecast(df=train_data, h=forecast_horizon)
            forecast_values = forecast_df[model_name].values
            
            # Evaluate on test data
            min_len = min(len(test_data), len(forecast_values))
            if min_len > 0:
                actual = test_data.values[:min_len]
                predicted = forecast_values[:min_len]
                
                mse_val = mse(actual, predicted)
                mae_val = mae(actual, predicted)
                
                # Simple accuracy: 1 - normalized RMSE
                rmse = np.sqrt(mse_val)
                data_range = np.max(actual) - np.min(actual)
                accuracy = max(0, 1 - (rmse / max(data_range, 1e-8)))
                
                return {
                    'model': model_name,
                    'success': True,
                    'forecast': forecast_values,
                    'mse': float(mse_val),
                    'mae': float(mae_val),
                    'accuracy': float(accuracy)
                }
            else:
                return {'model': model_name, 'success': False, 'error': 'No test data'}
                
        except Exception as e:
            logger.error("Model " + model_name + " failed: " + str(e))
            return {'model': model_name, 'success': False, 'error': str(e)}
    
    def retrain_and_select(self, historical_data, original_forecast, item_id):
        """
        Simple process: Test 3 models -> Pick best accuracy -> Return result
        """
        logger.info("Starting basic retrain for " + str(item_id))
        
        if not NIXTLA_AVAILABLE:
            return {
                'success': False,
                'error': 'Nixtla not available',
                'tested_models': []
            }
        
        # Prepare data - use 20% for testing
        forecast_horizon = len(original_forecast)
        test_size = max(1, len(historical_data) // 5)
        
        train_data_series = historical_data[:-test_size]
        test_data_series = historical_data[-test_size:]
        
        train_data = self.prepare_data(train_data_series, item_id)
        
        # Test all 3 models
        results = []
        model_names = list(self.models.keys())
        
        for model_name in model_names:
            logger.info("Testing " + model_name)
            result = self.test_single_model(model_name, train_data, test_data_series, forecast_horizon)
            results.append(result)
        
        # Find best model
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {
                'success': False,
                'error': 'All models failed',
                'tested_models': model_names,
                'results': results
            }
        
        # Select highest accuracy
        best_result = max(successful_results, key=lambda x: x['accuracy'])
        
        # Generate final forecast with full data
        full_train_data = self.prepare_data(historical_data, item_id)
        final_result = self.test_single_model(best_result['model'], full_train_data, 
                                            pd.Series([0] * forecast_horizon), forecast_horizon)
        
        return {
            'success': True,
            'tested_models': model_names,
            'best_model': best_result['model'],
            'best_accuracy': best_result['accuracy'],
            'best_forecast': final_result.get('forecast', []) if final_result.get('success') else [],
            'all_results': results
        }