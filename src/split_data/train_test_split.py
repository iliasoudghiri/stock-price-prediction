import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROCESSED_PATH = os.getenv("PROCESSED_PATH")
TRAIN_PATH = os.getenv("TRAIN_PATH")
VALIDATION_PATH = os.getenv("VALIDATION_PATH")
TEST_PATH = os.getenv("TEST_PATH")

PRICE_COL = os.getenv("PRICE_COL")
TARGET_COL = os.getenv("TARGET_COL")

def load_stock_data(file_path):
    """
    Load stock data from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the stock data.

    Returns:
        pd.DataFrame: A DataFrame containing the stock data.
    """
    return pd.read_csv(file_path, index_col="Date", parse_dates=True)

def add_target_variable(data, target_col=TARGET_COL):
    """
    Add the target variable (next day closing price) to the given stock data.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock data.
        target_col (str): The name of the target column. Default is "Next_Day_Close".

    Returns:
        pd.DataFrame: A DataFrame with the target variable added.
    """
    data_with_target = data.copy()
    # Make the target variable the next day return
    data_with_target[target_col] = (data_with_target[PRICE_COL].shift(-1)-data_with_target[PRICE_COL])/data_with_target[PRICE_COL]
    
    # Make the target variable the next day closing price
    # data_with_target[target_col] = data_with_target[PRICE_COL].shift(-1)
    
    # Remove the last row with NaN value introduced by shifting the closing price
    data_with_target = data_with_target[:-1]

    return data_with_target

def time_based_split(data, train_ratio=0.7, validation_ratio=0.2):
    """
    Perform a time-based split of the given stock data into train, validation, and test sets.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock data.
        train_ratio (float): The proportion of the data to include in the training set. Default is 0.7.
        validation_ratio (float): The proportion of the data to include in the validation set. Default is 0.2.

    Returns:
        tuple: A tuple containing the train, validation, and test DataFrames.
    """
    train_size = int(len(data) * train_ratio)
    validation_size = int(len(data) * validation_ratio)

    train_data = data.iloc[:train_size]
    validation_data = data.iloc[train_size:train_size + validation_size]
    test_data = data.iloc[train_size + validation_size:]

    return train_data, validation_data, test_data

if __name__ == "__main__":

    processed_files = os.listdir(PROCESSED_PATH)

    for file in processed_files:

        file_path = os.path.join(PROCESSED_PATH,file)
        # Load the feature-engineered data
        feature_engineered_data = load_stock_data(file_path)

        # Add the target variable
        data_with_target = add_target_variable(feature_engineered_data)

        # Perform the time-based train, validation, and test split
        train_data, validation_data, test_data = time_based_split(data_with_target)

        # Save the train, validation, and test sets as separate CSV files
        train_data.to_csv(os.path.join(TRAIN_PATH,file))
        validation_data.to_csv(os.path.join(VALIDATION_PATH,file))
        test_data.to_csv(os.path.join(TEST_PATH,file))

