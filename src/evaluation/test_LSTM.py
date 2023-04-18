import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from dotenv import load_dotenv

load_dotenv()

# Load the saved LSTM model
MODEL_PATH = os.getenv("MODEL_PATH")
TEST_PATH = os.getenv("TEST_PATH")
PREDICTIONS_PATH = os.getenv("PREDICTIONS_PATH")
TARGET_COL = os.getenv("TARGET_COL")
lookback = 60

def load_stock_data(file_path):
    """
    Load stock data from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the stock data.

    Returns:
        pd.DataFrame: A DataFrame containing the stock data.
    """
    return pd.read_csv(file_path, index_col="Date", parse_dates=True)

def create_dataset(X, lookback=60):
    X_new = []
    for i in range(len(X) - lookback):
        X_new.append(X[i:(i + lookback), :])
    return np.array(X_new)

if __name__ == "__main__" : 

    test_data = load_stock_data(os.path.join(TEST_PATH,"AAPL_2010-01-01_2023-01-01.csv"))
    lstm_model = load_model(os.path.join(MODEL_PATH,"best_lstm_model.h5"))

    # Transform the test data to the appropriate format for the LSTM model
    X_test = test_data.drop(TARGET_COL, axis=1).values 
    y_test = test_data[TARGET_COL].values  
    
    X_test = create_dataset(X_test, lookback)
    y_test = y_test[lookback:]

    # Make predictions on the test data using the trained LSTM model
    y_pred = lstm_model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_absolute_percentage_error(y_test, y_pred)
    print(f"Mean absolute percentage error on test data: {mse}")


    # Save the predictions and performance metrics
    results_df = pd.DataFrame({"actual": y_test, "predicted": y_pred.flatten()})
    results_df.to_csv(os.path.join(PREDICTIONS_PATH,"predictions_best_lstm_model.csv"), index=False)
    
    with open("data/metrics/best_lstm_model_performance.txt", "w") as f:
        f.write(f"Mean absolute percentage error on test data: {mse}\n")