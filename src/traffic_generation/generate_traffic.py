"""Script to generate traffic to the prediction API endpoint. It was copied and adapted from the given evaluation script."""

import requests
import pandas as pd
import datetime
import io
import zipfile
import sys
import warnings

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# --- Configuration ---
DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
API_PREDICT_URL = "http://bike-api:8080/predict"

NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']
ALL_MODEL_FEATS = NUM_FEATS + CAT_FEATS

DTEDAY_COL_NAME = 'dteday'


# --- Fonctions d'ingestion et de préparation des données (alignées sur l'examen Evidently) ---
def _fetch_data() -> pd.DataFrame:
    """Fetches the bike sharing dataset and returns a DataFrame."""
    print("Fetching data from UCI archive...")
    try:
        content = requests.get(DATASET_URL, verify=False, timeout=60).content
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            df = pd.read_csv(z.open("hour.csv"), header=0, sep=',', parse_dates=[DTEDAY_COL_NAME])
        print("Data fetched successfully.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}. Check URL or network connection.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing fetched data: {e}")
        sys.exit(1)

def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Processes raw data, setting a DatetimeIndex as in the exam script."""
    print("Processing raw data...")
    raw_data['hr'] = raw_data['hr'].astype(int)
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(row[DTEDAY_COL_NAME].date(), datetime.time(row.hr)),
        axis=1
    )
    raw_data = raw_data.sort_index()
    print("Data processed successfully.")
    return raw_data

def generate_traffic(count: int, full_data: pd.DataFrame):
    """
    Generates simulated traffic to the /predict endpoint.
    """
    print(f"\n--- Generating {count} prediction requests to {API_PREDICT_URL} ---")

    predict_sample_df = full_data.loc['2011-01-01 00:00:00':'2011-01-31 23:00:00'].copy()
    if predict_sample_df.empty:
        print("Warning: No data for prediction traffic. Check date ranges.")
        return

    if predict_sample_df.shape[0] < count:
        print("Warning: Not enough data for prediction traffic. Using available data.")
        predict_samples = predict_sample_df[ALL_MODEL_FEATS + [DTEDAY_COL_NAME]].to_dict(orient='records')
    else:
        predict_samples = predict_sample_df[ALL_MODEL_FEATS + [DTEDAY_COL_NAME]].sample(n=count, random_state=42).to_dict(orient='records')

    for i, sample_features in enumerate(predict_samples):
        if i % 10 == 0:
            print(f"  - Sending prediction request {i+1}/{count}...")
        try:
            sample_features_copy = sample_features.copy()
            if isinstance(sample_features_copy.get(DTEDAY_COL_NAME), datetime.date):
                sample_features_copy[DTEDAY_COL_NAME] = sample_features_copy[DTEDAY_COL_NAME].strftime('%Y-%m-%d')
                
            response = requests.post(API_PREDICT_URL, json=sample_features_copy, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"    - Error sending prediction request {i+1}: {e}")
        except Exception as e:
            print(f"    - Unexpected error for prediction request {i+1}: {e}")
    print(f"{count} prediction requests sent.")


# --- Main execution logic ---
if __name__ == "__main__":
    _full_data_cache = _process_data(_fetch_data())
    generate_traffic(500, _full_data_cache)