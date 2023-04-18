import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from keras.losses import MeanAbsolutePercentageError
from dotenv import load_dotenv
from ast import literal_eval

load_dotenv()

LOOKBACK = literal_eval(os.getenv("LOOKBACK"))
TRAIN_PATH = os.getenv("TRAIN_PATH")
VALIDATION_PATH = os.getenv("VALIDATION_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")

TARGET_COL = os.getenv("TARGET_COL")
LSTM_UNITS = os.getenv("LSTM_UNITS")
OPTIMIZER  = os.getenv("OPTIMIZER")

def load_stock_data(file_path):
    """
    Load stock data from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the stock data.

    Returns:
        pd.DataFrame: A DataFrame containing the stock data.
    """
    return pd.read_csv(file_path, index_col="Date", parse_dates=True)

# Preprocess the data for the LSTM model
def preprocess_lstm_data(data, target_col, lookback=60):
    data = data.copy()
    input_features = data.drop(columns=[target_col]).values
    target = data[target_col].values
    
    scaler = MinMaxScaler()
    input_features = scaler.fit_transform(input_features)
    
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(input_features[i - lookback:i])
        y.append(target[i])
    
    return np.array(X), np.array(y)

# create the LSTM model
def create_lstm_model(units=50, optimizer="adam"):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss=MeanAbsolutePercentageError())
    return model

def custom_scorer(model, X, y_true):
    y_pred = model.predict(X)
    mse = mean_absolute_percentage_error(y_true, y_pred)
    return -mse

if __name__ == "__main__":

    # Load the train, validation, and test data
    train_data = load_stock_data(os.path.join(TRAIN_PATH,"AAPL_2010-01-01_2023-01-01.csv"))
    validation_data = load_stock_data(os.path.join(VALIDATION_PATH,"AAPL_2010-01-01_2023-01-01.csv"))

    X_train, y_train = preprocess_lstm_data(train_data, TARGET_COL, lookback=LOOKBACK)
    X_validation, y_validation = preprocess_lstm_data(validation_data, TARGET_COL, lookback=LOOKBACK)

    # Build the LSTM model architecture
    lstm_model = KerasRegressor(build_fn=create_lstm_model, epochs=100, batch_size=32, verbose=0)

    # Define the hyperparameter search space for the grid search
    param_grid = {
        "units": [50, 100],
        "optimizer": ["adam", "rmsprop"]
    }

    # Create a TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=3)

    # Perform grid search using GridSearchCV
    grid_search = GridSearchCV(estimator=lstm_model, param_grid=param_grid, scoring=custom_scorer, cv=tscv, verbose = 5)
    grid_search.fit(X_train, y_train)

    # Extract the best hyperparameters
    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    # Retrain the LSTM model with the best hyperparameters and early stopping
    best_lstm_model = create_lstm_model(units=best_params["units"], optimizer=best_params["optimizer"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    best_lstm_model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_validation, y_validation), callbacks=[early_stopping])
    
    best_lstm_model.save("models/best_lstm_model.h5")
    
    # Train the LSTM model using the training data
    # early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    # history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_validation, y_validation), callbacks=[early_stopping])

    # Evaluate the LSTM model on the validation data
    # validation_predictions = model.predict(X_validation)
