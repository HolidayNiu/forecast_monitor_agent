"""
Data loader module for forecast monitoring agent.
"""
import pandas as pd
from typing import Dict, Tuple, List
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """Load the forecast data from CSV."""
    return pd.read_csv(file_path)


def get_item_data(df: pd.DataFrame, item_loc_id: str, min_historical_months: int = 6, min_forecast_months: int = 6) -> Tuple[pd.Series, pd.Series]:
    """
    Extract historical and forecast data for a specific item with flexible data lengths.
    
    Args:
        df: DataFrame containing the forecast data
        item_loc_id: Item identifier to extract data for
        min_historical_months: Minimum number of historical months required (default: 6)
        min_forecast_months: Minimum number of forecast months required (default: 6)
    
    Returns:
        historical_data: clean_qty where not null (historical data)
        forecast_data: best_model_forecast where clean_qty is null (forecast data)
    
    Raises:
        ValueError: If minimum data requirements are not met
    """
    item_df = df[df['item_loc_id'] == item_loc_id].sort_values('FORECAST_MONTH')
    
    if len(item_df) == 0:
        raise ValueError(f"No data found for {item_loc_id}")
    
    # Historical data: where clean_qty is not null
    historical_mask = pd.notna(item_df['clean_qty'])
    historical_data = item_df[historical_mask]['clean_qty'].reset_index(drop=True)
    
    # Forecast data: where clean_qty is null, use best_model_forecast
    forecast_mask = pd.isna(item_df['clean_qty'])
    forecast_data = item_df[forecast_mask]['best_model_forecast'].reset_index(drop=True)
    
    # Validate minimum data requirements
    if len(historical_data) < min_historical_months:
        raise ValueError(f"Insufficient historical data for {item_loc_id}: "
                        f"got {len(historical_data)} months, need at least {min_historical_months}")
    
    if len(forecast_data) < min_forecast_months:
        raise ValueError(f"Insufficient forecast data for {item_loc_id}: "
                        f"got {len(forecast_data)} months, need at least {min_forecast_months}")
    
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