"""Script to trigger the high RMSE alert. It was copied and adapted from the given evaluation script."""

import json
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
API_EVALUATE_URL = "http://bike-api:8080/evaluate"

EVALUATION_SAMPLE_SIZE = 1000 # Nombre d'échantillons à utiliser pour chaque évaluation

WEEKLY_PERIODS = {
    'week1_february': ('2011-01-29 00:00:00', '2011-02-07 23:00:00'),
    'week2_february': ('2011-02-08 00:00:00', '2011-02-14 23:00:00'),
    'week3_february': ('2011-02-15 00:00:00', '2011-02-21 23:00:00')
}
DEFAULT_EVAL_PERIOD = WEEKLY_PERIODS['week3_february']
DEFAULT_PERIOD_NAME = 'week3_february'

NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']
ALL_MODEL_FEATS = NUM_FEATS + CAT_FEATS
TARGET = 'cnt'

DTEDAY_COL_NAME = 'dteday'
COLUMNS_FOR_EVALUATION_PAYLOAD = ALL_MODEL_FEATS + [TARGET, DTEDAY_COL_NAME]


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


def run_evaluation(full_data: pd.DataFrame, period_name: str, start_date_str: str, end_date_str: str):
    """
    Prepares evaluation data for a given period and sends it to the API.
    """
    print(f"\n--- Running evaluation for period: {period_name} ({start_date_str} to {end_date_str}) ---")

    try:
        current_period_data = full_data.loc[start_date_str:end_date_str].copy()
        
        if current_period_data.empty:
            print(f"No data found for period {period_name}. Skipping evaluation.")
            return

        eval_payload_df = current_period_data[COLUMNS_FOR_EVALUATION_PAYLOAD].copy()
        
        if eval_payload_df.shape[0] > EVALUATION_SAMPLE_SIZE:
            eval_payload_df = eval_payload_df.sample(n=EVALUATION_SAMPLE_SIZE, random_state=42)

        eval_payload_df[DTEDAY_COL_NAME] = eval_payload_df[DTEDAY_COL_NAME].astype(str)

        evaluation_data_payload = eval_payload_df.to_dict(orient='records')
        
        print(f"Sending {len(evaluation_data_payload)} samples to API endpoint {API_EVALUATE_URL}...")
        response = requests.post(
            API_EVALUATE_URL,
            json={'data': evaluation_data_payload, 'evaluation_period_name': period_name},
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        
        print("Evaluation successful!")
        print(f"  - Message: {result.get('message')}")
        
        rmse_value = result.get('rmse')
        mape_value = result.get('mape')

        print(f"  - RMSE: {rmse_value:.4f}" if rmse_value is not None else "  - RMSE: N/A")
        print(f"  - MAPE: {mape_value:.4f}" if mape_value is not None else "  - MAPE: N/A")
        
        print(f"  - Data Drift Detected: {'Yes' if result.get('drift_detected') == 1 else 'No'}")
        print(f"  - Evaluated Items: {result.get('evaluated_items')}")

    except requests.exceptions.RequestException as e:
        print(f"Error sending evaluation request: {e}. Is the API running at {API_EVALUATE_URL}?")
    except json.JSONDecodeError:
        print(f"Error decoding response JSON from API. Response text: {response.text}")
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")


# --- Main execution logic ---
if __name__ == "__main__":
    _full_data_cache = _process_data(_fetch_data())

    start_date, end_date = DEFAULT_EVAL_PERIOD
    run_evaluation(_full_data_cache, DEFAULT_PERIOD_NAME, start_date, end_date)