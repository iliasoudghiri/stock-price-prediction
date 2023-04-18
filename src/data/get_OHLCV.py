import os
import yfinance as yf
from dotenv import load_dotenv
from ast import literal_eval

load_dotenv()

TICKERS = literal_eval(os.getenv("TICKERS"))
START_DATE = os.getenv("START_DATE")
END_DATE = os.getenv("END_DATE")


def download_stock_data(ticker, start_date, end_date, data_folder="data/raw"):
    """
    Download stock data for the given ticker and date range, and save it as a CSV file.

    Args:
        ticker (str): The stock symbol to download data for.
        start_date (str): The start date for the data (YYYY-MM-DD format).
        end_date (str): The end date for the data (YYYY-MM-DD format).
        data_folder (str): The folder where the data should be saved.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    file_path = os.path.join(data_folder, f"{ticker}_{start_date}_{end_date}.csv")
    stock_data.to_csv(file_path)

if __name__ == "__main__":

    for ticker in TICKERS:
        print(f"Dowloading data for {ticker}")
        download_stock_data(ticker, START_DATE, END_DATE)