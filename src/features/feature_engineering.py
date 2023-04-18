import os, talib
from dotenv import load_dotenv
from ast import literal_eval
import pandas as pd
import numpy as np

load_dotenv()

CLEAN_PATH = os.getenv("CLEAN_PATH")
PROCESSED_PATH = os.getenv("PROCESSED_PATH")
INDICATORS = literal_eval(os.getenv("INDICATORS"))
PRICE_COL = os.getenv("PRICE_COL")

def load_stock_data(file_path):
    """
    Load stock data from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the stock data.

    Returns:
        pd.DataFrame: A DataFrame containing the stock data.
    """
    return pd.read_csv(file_path, index_col="Date", parse_dates=True)


def create_lagged_price_features(data, n_lags=5):
    """
    Create time-lagged price features for the given stock data.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock data.
        n_lags (int): The number of lagged features to create.

    Returns:
        pd.DataFrame: A DataFrame with the lagged features.
    """
    lagged_data = data.copy()

    for i in range(1, n_lags + 1):
        lagged_data[f"Close_lag_{i}"] = lagged_data[PRICE_COL].shift(i)

    # Remove the rows with missing values introduced by lagging
    lagged_data = lagged_data.dropna()

    return lagged_data

def create_lagged_volume_features(data, n_lags=5):
    """
    Create time-lagged volume features for the given stock data.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock data.
        n_lags (int): The number of lagged features to create.

    Returns:
        pd.DataFrame: A DataFrame with the lagged features.
    """
    lagged_data = data.copy()

    for i in range(1, n_lags + 1):
        lagged_data[f"Volume_lag_{i}"] = lagged_data["Volume"].shift(i)

    # Remove the rows with missing values introduced by lagging
    lagged_data = lagged_data.dropna()

    return lagged_data

def create_technical_indicators(data, indicators):
    """
    Create technical indicators for the given stock data.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock data.
        indicators (dict): A dictionary with column names as keys and dictionaries containing the technical indicator names and their parameters as values.

    Returns:
        pd.DataFrame: A DataFrame with the technical indicators.
    """
    technical_data = data.copy()
    for column_name, indicator_info in indicators.items():
        indicator = indicator_info['indicator']
        params = indicator_info['params']
        indicator_function = getattr(talib, indicator)
        indicator_values = indicator_function(technical_data[PRICE_COL], **params)

        # Check if the indicator returns multiple outputs
        if isinstance(indicator_values, tuple):
            for i, value in enumerate(indicator_values):
                technical_data[f"{column_name}_{i+1}"] = value
        else:
            technical_data[column_name] = indicator_values

    # Remove the rows with missing values introduced by the technical indicators
    technical_data = technical_data.dropna()

    return technical_data

def add_return_features(data, log_returns=True):
    """
    Add return features to the given stock data.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock data.
        log_returns (bool): Whether to calculate logarithmic returns. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with the return features added.
    """
    data_with_returns = data.copy()
    if log_returns:
        data_with_returns['Log_Return'] = np.log(data_with_returns[PRICE_COL] / data_with_returns[PRICE_COL].shift(1))
    else:
        data_with_returns['Return'] = data_with_returns[PRICE_COL].pct_change()

    # Remove the first row with NaN value introduced by calculating the returns
    data_with_returns = data_with_returns.dropna()

    return data_with_returns


if __name__ == "__main__" :

    clean_files = os.listdir(CLEAN_PATH)


    for file in clean_files:

        file_path = os.path.join(CLEAN_PATH,file)
        clean_data = load_stock_data(file_path)
        lagged_price_data = create_lagged_price_features(clean_data,n_lags=10)
        lagged_volume_data = create_lagged_volume_features(lagged_price_data,n_lags=10) 
        feature_data = create_technical_indicators(lagged_volume_data,INDICATORS)
        return_data = add_return_features(feature_data,log_returns=False)

        return_data.to_csv(os.path.join(PROCESSED_PATH,file))

