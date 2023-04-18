import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

PLOTS_PATH = os.getenv("PLOTS_PATH")

# Load test data
TEST_PATH = os.getenv("TEST_PATH")
df_test = pd.read_csv(os.path.join(TEST_PATH,"AAPL_2010-01-01_2023-01-01.csv"), index_col="Date", parse_dates=True)

# Load predictions and actual values
PREDICTIONS_PATH = os.getenv("PREDICTIONS_PATH")
df = pd.read_csv(os.path.join(PREDICTIONS_PATH,"predictions_best_lstm_model.csv"))

# Compute next day's predicted price using the previous day's actual price and the predicted return from the model
df_test["predicted"] = df_test["Close"].shift(-1) * (1 + df_test["Next_Day_Return"])

# Create a line chart comparing actual and predicted values
plt.figure(figsize=(16, 8))
# plt.plot(df_test["Close"], label="Actual")
# bar plot for actual values to make it easier to see the difference between actual and predicted values

plt.bar(df["actual"], label="Actual")
plt.bar(df["predicted"], label="Predicted", alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Actual vs Predicted Stock Prices (LSTM))")
plt.legend()
plt.grid()

# Save the plot as an image
output_image_path = os.path.join(PLOTS_PATH,"actual_vs_predicted_prices.png")
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
plt.savefig(output_image_path, dpi=300)

# Show the plot
plt.show()
