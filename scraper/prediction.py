import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# Load cleaned data
df = pd.read_csv("final_data.csv")

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Drop duplicates if any
df = df.drop_duplicates(subset=["date", "keyword"]).reset_index(drop=True)

# Create a folder for forecast plots
os.makedirs("forecast_plots", exist_ok=True)

# Forecast each keyword individually
for kw in df['keyword'].unique():
    keyword_df = df[df['keyword'] == kw]

    # Prepare data for Prophet
    prophet_df = keyword_df[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})

    # Check for minimum data points
    if len(prophet_df) < 10:
        print(f"[⚠] Skipped {kw}: not enough data.")
        continue

    try:
        # Fit model
        model = Prophet()
        model.fit(prophet_df)

        # Make future predictions (next 30 days)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Plot and save
        fig = model.plot(forecast)
        plt.title(f"Forecast for: {kw}")
        plt.xlabel("Date")
        plt.ylabel("Search Interest")
        plt.tight_layout()
        plt.savefig(f"forecast_plots/{kw}_forecast.png")
        plt.close()

        print(f"[✔] Saved forecast for: {kw}")
    except Exception as e:
        print(f"[✘] Failed: {kw} -> {e}")
