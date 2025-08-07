"""
Data loader module for forecast monitoring agent.
"""
import pandas as pd
from typing import Dict, Tuple, List
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """Load the forecast data from CSV."""
    return pd.read_csv(file_path)


def get_item_data(df: pd.DataFrame, item_loc_id: str) -> Tuple[pd.Series, pd.Series]:
    """
    Extract historical and forecast data for a specific item.
    
    Returns:
        historical_data: clean_qty where not null (historical data)
        forecast_data: best_model_forecast where clean_qty is null (forecast data)
    """
    item_df = df[df['item_loc_id'] == item_loc_id].sort_values('FORECAST_MONTH')
    
    if len(item_df) != 54:  # Should be 54 months total
        raise ValueError(f"Expected 54 months of data for {item_loc_id}, got {len(item_df)}")
    
    # Historical data: where clean_qty is not null
    historical_mask = pd.notna(item_df['clean_qty'])
    historical_data = item_df[historical_mask]['clean_qty'].reset_index(drop=True)
    
    # Forecast data: where clean_qty is null, use best_model_forecast
    forecast_mask = pd.isna(item_df['clean_qty'])
    forecast_data = item_df[forecast_mask]['best_model_forecast'].reset_index(drop=True)
    
    return historical_data, forecast_data


def get_all_item_ids(df: pd.DataFrame) -> List[str]:
    """Get list of all unique item_loc_ids."""
    return df['item_loc_id'].unique().tolist()


def get_recent_actuals(historical_data: pd.Series, months: int = 6) -> pd.Series:
    """Get the last N months of historical data."""
    return historical_data.tail(months)


def get_early_forecast(forecast_data: pd.Series, months: int = 6) -> pd.Series:
    """Get the first N months of forecast data."""
    return forecast_data.head(months)