import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_PATH = os.getenv("RAW_PATH")
CLEAN_PATH = os.getenv("CLEAN_PATH")
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

    
def clean_stock_data(data):
    """
    Clean the stock data by handling missing values.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock data.

    Returns:
        pd.DataFrame: A cleaned DataFrame.
    """
    cleaned_data = data.copy()
    cleaned_data = cleaned_data.fillna(method="ffill")

    return cleaned_data

def normalize_close_price(data):
    """
    Normalize the stock data by selecting relevant columns and normalizing the data.

    Args:
        data (pd.DataFrame): A DataFrame containing the stock data.

    Returns:
        pd.DataFrame: A preprocessed DataFrame.
    """
    preprocessed_data = data.copy()

    # Select relevant columns
    relevant_column = preprocessed_data[[PRICE_COL]]

    # Normalize the data
    preprocessed_data["Noramlized_Close"] = (relevant_column - relevant_column.min()) / (relevant_column.max() - relevant_column.min())

    return preprocessed_data

def save_data(data,file_path):

    data.to_csv(file_path)


if __name__ == "__main__":

    raw_files = os.listdir(RAW_PATH)

    for raw_file in raw_files:

        stock_data = load_stock_data(os.path.join(RAW_PATH,raw_file))
        # Clean the stock data
        cleaned_data = clean_stock_data(stock_data)

        # Preprocess the stock data
        preprocessed_data = normalize_close_price(cleaned_data)

        # Save the preprocessed data to a CSV file
        preprocessed_data.to_csv(os.path.join(CLEAN_PATH,raw_file))
        