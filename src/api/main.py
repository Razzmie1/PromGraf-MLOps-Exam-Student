from evidently import ColumnMapping
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import logging
import datetime
import time
from typing import Optional, Any

from evidently.report import Report
from evidently.metrics import RegressionPerformanceMetrics, DatasetDriftMetric

from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge
from sklearn.ensemble import RandomForestRegressor


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0"
)

# --- Prometheus Metrics Definitions ---
registry = CollectorRegistry()

api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

model_rmse_score = Gauge(
    'model_rmse_score',
    'RMSE score of the regression model',
    registry=registry
)

model_mae_score = Gauge(
    'model_mae_score',
    'MAE score of the regression model',
    registry=registry
)

model_r2_score = Gauge(
    'model_r2_score',
    'R2 score of the regression model',
    registry=registry
)

# Important for monitoring as it possibly explains when model performance decreases
evidently_data_drift_detected_status = Gauge(
    'data_drift_status',
    'Status of possible Data Drift',
    registry=registry
)


# --- Global Variables for Model and Data ---
TARGET = 'cnt'
PREDICTION = 'prediction'
NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']


# --- Data Ingestion and Preparation Functions ---
def _fetch_data() -> pd.DataFrame:
    content = requests.get("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", verify=False).content

    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday']) 
    return raw_data

def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)), axis=1)
    return raw_data

def _train_and_predict_reference_model():
    bike_data = _process_data(_fetch_data())
    reference_jan11 = bike_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']

    regressor = RandomForestRegressor(random_state = 0, n_estimators = 50)
    regressor.fit(reference_jan11[NUM_FEATS + CAT_FEATS], reference_jan11[TARGET])

    ref_prediction = regressor.predict(reference_jan11[NUM_FEATS + CAT_FEATS])
    reference_jan11[PREDICTION] = ref_prediction
    return regressor, reference_jan11

# TODO: refactor
REGRESSOR, REFERENCE_JAN11 = _train_and_predict_reference_model()


# --- Pydantic Models for API Input/Output ---
class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: datetime.date = Field(..., example="2011-01-01", description="Date of the record in YYYY-MM-DD format.")

class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)

class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="List of data points, each containing features and the true target ('cnt').")
    evaluation_period_name: str = Field("unknown_period", description="Name of the period being evaluated (e.g., 'week1_february').")
    model_config = {'arbitrary_types_allowed': True}

class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: BikeSharingInput):
    start_time = time.time()
    status_code = "200"

    try:
        input_dict = input.model_dump()
        input_data = pd.DataFrame(input_dict, index=[0])[NUM_FEATS + CAT_FEATS].copy()
        pred_cnt = REGRESSOR.predict(input_data).item()
        logger.info(f"Predicted bike count: {pred_cnt} for input: {input_dict}")
        return PredictionOutput(predicted_count=pred_cnt)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        status_code = "500"
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        api_request_duration_seconds.labels(endpoint="/predict", method="POST", status_code=status_code).observe(duration)
        api_requests_total.labels(endpoint="/predict", method="POST", status_code=status_code).inc()

@app.post("/evaluate", response_model=EvaluationReportOutput)
async def evaulate(eval_data: EvaluationData):
    start_time = time.time()
    status_code = "200"

    try:
        if not eval_data.data:
            status_code = "400"
            raise HTTPException(status_code=400, detail="No items provided for evaluation.")
        
        current_data = pd.DataFrame(eval_data.data)[NUM_FEATS + CAT_FEATS + [TARGET]].copy()
        predictions = REGRESSOR.predict(current_data[NUM_FEATS + CAT_FEATS])
        current_data[PREDICTION] = predictions

        # Column mapping for Evidently
        column_mapping = ColumnMapping()
        column_mapping.target = TARGET
        column_mapping.prediction = PREDICTION
        column_mapping.numerical_features = NUM_FEATS
        column_mapping.categorical_features = CAT_FEATS

        # Generate Evidently Report
        report = Report(metrics=[RegressionPerformanceMetrics(), DatasetDriftMetric()])
        report.run(
            reference_data=REFERENCE_JAN11,
            current_data=current_data,
            column_mapping=column_mapping
        )

        # Extract metrics and data drift status from the report
        report_dict = report.as_dict()
        dataset_drift = report_dict["metrics"][1]["result"]["dataset_drift"]
        regression_metrics = report_dict['metrics'][0]['result']['current']
        rmse = regression_metrics['rmse']
        mae = regression_metrics['mean_abs_error']
        r2score = regression_metrics['r2_score']

        # Update Model Metrics Gauges
        model_rmse_score.set(rmse)
        model_mae_score.set(mae)
        model_r2_score.set(r2score)
        evidently_data_drift_detected_status.set(int(dataset_drift))
        message = f"Evaluation completed for period: {eval_data.evaluation_period_name}"

        logger.info(f"{message} | RMSE: {rmse}, MAE: {mae}, R2: {r2score}, Data Drift Detected: {dataset_drift}")
        return EvaluationReportOutput(
            message=message,
            rmse=rmse,
            mape=None,
            mae=mae,
            r2score=r2score,
            drift_detected=int(dataset_drift),
            evaluated_items=len(eval_data.data)
        )
    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        status_code = "500"
        raise HTTPException(status_code=500, detail=f"Evaluation failed due to an internal error: {e}")
    finally:
        # Update API Metrics
        end_time = time.time()
        duration = end_time - start_time
        api_request_duration_seconds.labels(endpoint="/evaluate", method="POST", status_code=status_code).observe(duration)
        api_requests_total.labels(endpoint="/evaluate", method="POST", status_code=status_code).inc()

@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(registry), media_type="text/plain")