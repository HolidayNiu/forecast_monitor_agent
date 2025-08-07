"""
Basic Retrain Agent - Uses Nixtla's cross_validation for model selection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Callable
import logging
from datetime import datetime
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add project root to path  
sys.path.append(str(Path(__file__).parent.parent))


class BasicRetrainAgent:
    """
    Retraining agent that uses Nixtla's cross_validation for HoltWinters, ETS, AutoETS
    """
    
    def __init__(self):
        """Initialize basic retrain agent"""
        self.available_models = ['HoltWinters', 'ETS', 'AutoETS']
        
    def retrain_item(self, df: pd.DataFrame, item_loc_id: str, current_model: str = None) -> Dict:
        """
        Retrain item using Nixtla's cross_validation and select best model.
        
        Args:
            df: Full dataset with FORECAST_MONTH column
            item_loc_id: Item identifier
            current_model: Current model name
            
        Returns:
            Dictionary with retraining results
        """
        try:
            # Import required Nixtla libraries
            from statsforecast import StatsForecast
            from statsforecast.models import (
                HoltWinters,
                ETS,
                AutoETS
            )
            
            # Get item data and prepare historical data only
            item_df = df[df['item_loc_id'] == item_loc_id].sort_values('FORECAST_MONTH').copy()
            
            # Convert FORECAST_MONTH to datetime
            item_df['ds'] = pd.to_datetime(item_df['FORECAST_MONTH'])
            
            # Get only historical data (where clean_qty is not NaN)
            historical_df = item_df[pd.notna(item_df['clean_qty'])].copy()
            
            if len(historical_df) < 24:  # Need enough data for cross validation
                raise ValueError(f"Need at least 24 data points for cross validation, got {len(historical_df)}")
            
            # Prepare data for StatsForecast using actual dates
            sf_df = pd.DataFrame({
                'unique_id': 'item',
                'ds': historical_df['ds'],
                'y': historical_df['clean_qty']
            })
            
            # Define the 3 fixed models
            models = [
                HoltWinters(season_length=12, alias='HoltWinters'),
                ETS(season_length=12, alias='ETS'),
                AutoETS(season_length=12, alias='AutoETS')
            ]
            
            logger.info("Training models: HoltWinters, ETS, AutoETS")
            sf = StatsForecast(models=models, freq='MS', n_jobs=1)
            
            # Use Nixtla's cross_validation
            # Use last 6 months as test set, with 2 folds
            cv_results = sf.cross_validation(
                df=sf_df,
                h=6,  # forecast horizon for each fold
                step_size=3,  # step between folds
                n_windows=2  # number of cross validation windows
            )
            
            logger.info("Cross validation completed")
            
            # Evaluate CV results following the proper structure
            def mae(y_true, y_pred):
                return np.mean(np.abs(y_true - y_pred))
            
            def mape(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            def rmse(y_true, y_pred):
                return np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            metrics = [mae, mape, rmse]
            eval_results = self.evaluate(cv_results, metrics)
            
            logger.info("CV evaluation completed")
            
            # Process evaluation results for each model
            model_results = []
            for model_name in self.available_models:
                if model_name in eval_results.columns:
                    model_metrics = eval_results[model_name]
                    
                    mae_score = model_metrics[model_metrics['metric'] == 'mae'].iloc[0] if len(model_metrics[model_metrics['metric'] == 'mae']) > 0 else np.nan
                    mape_score = model_metrics[model_metrics['metric'] == 'mape'].iloc[0] if len(model_metrics[model_metrics['metric'] == 'mape']) > 0 else np.nan
                    rmse_score = model_metrics[model_metrics['metric'] == 'rmse'].iloc[0] if len(model_metrics[model_metrics['metric'] == 'rmse']) > 0 else np.nan
                    
                    if not np.isnan(mape_score):
                        model_results.append({
                            'model': model_name,
                            'mae': mae_score,
                            'mape': mape_score,
                            'rmse': rmse_score,
                            'accuracy_score': 1 / (1 + mape_score/100),
                            'cv_points': len(cv_results)
                        })
                        
                        logger.info(f"{model_name} - MAE: {mae_score:.2f}, MAPE: {mape_score:.2f}%, RMSE: {rmse_score:.2f}")
                    else:
                        logger.warning(f"Invalid metrics for {model_name}")
                else:
                    logger.warning(f"Model {model_name} not found in evaluation results")
            
            if not model_results:
                raise Exception("No models completed successfully in cross validation")
            
            # Select best model (lowest MAPE)
            best_model_result = min(model_results, key=lambda x: x['mape'])
            best_model_name = best_model_result['model']
            
            logger.info(f"Best model selected: {best_model_name} with MAPE: {best_model_result['mape']:.2f}%")
            
            # Fit best model on full historical data and generate forecast
            best_model_class = self._get_model_class(best_model_name)
            sf_final = StatsForecast(models=[best_model_class], freq='MS', n_jobs=1)
            sf_final.fit(sf_df)
            
            # Generate 18-month forecast
            forecast = sf_final.predict(h=18)
            full_forecast = forecast[best_model_name].tolist()
            
            return {
                'success': True,
                'item_loc_id': item_loc_id,
                'best_model': best_model_name,
                'previous_model': current_model,
                'model_results': model_results,
                'best_model_metrics': best_model_result,
                'full_forecast': full_forecast,
                'historical_data': historical_df['clean_qty'].tolist(),
                'historical_dates': historical_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'cross_validation_summary': {
                    'total_cv_points': len(cv_results),
                    'forecast_horizon': 6,
                    'n_windows': 2,
                    'historical_months': len(historical_df)
                },
                'timestamp': datetime.now()
            }
                
        except ImportError as e:
            logger.error(f"Nixtla statsforecast not available: {e}")
            raise Exception("statsforecast library required. Install with: pip install statsforecast")
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            raise Exception(f"Retraining failed: {str(e)}")
    
    def evaluate(self, df: pd.DataFrame, metrics: List[Callable]) -> pd.DataFrame:
        """Evaluate CV results following the proper Nixtla structure"""
        eval_ = {}
        models = df.loc[:, ~df.columns.str.contains('unique_id|y|ds|cutoff|lo|hi')].columns
        for model in models:
            eval_[model] = {}
            for metric in metrics:
                # Handle NaN values in predictions
                y_true = df['y'].values
                y_pred = df[model].values
                
                # Remove NaN pairs
                valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                if np.sum(valid_mask) > 0:
                    eval_[model][metric.__name__] = metric(y_true[valid_mask], y_pred[valid_mask])
                else:
                    eval_[model][metric.__name__] = np.nan
                    
        eval_df = pd.DataFrame(eval_).rename_axis('metric').reset_index()
        eval_df.insert(0, 'cutoff', df['cutoff'].iloc[0] if 'cutoff' in df.columns else None)
        eval_df.insert(0, 'unique_id', df['unique_id'].iloc[0])
        return eval_df
    
    def _get_model_class(self, model_name: str):
        """Get model class for the given model name"""
        from statsforecast.models import (
            HoltWinters,
            ETS,
            AutoETS
        )
        
        model_map = {
            'HoltWinters': HoltWinters(season_length=12, alias='HoltWinters'),
            'ETS': ETS(season_length=12, alias='ETS'),
            'AutoETS': AutoETS(season_length=12, alias='AutoETS')
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_map[model_name]
    
    def get_model_comparison(self, results: Dict) -> pd.DataFrame:
        """Get comparison table of model performance"""
        if not results['success']:
            return pd.DataFrame()
        
        model_data = []
        for model_result in results['model_results']:
            model_data.append({
                'Model': model_result['model'],
                'MAE': round(model_result['mae'], 2),
                'MAPE (%)': round(model_result['mape'], 2),
                'RMSE': round(model_result['rmse'], 2),
                'Accuracy Score': round(model_result['accuracy_score'], 3),
                'CV Points': model_result['cv_points'],
                'Selected': '✓' if model_result['model'] == results['best_model'] else ''
            })
        
        return pd.DataFrame(model_data)