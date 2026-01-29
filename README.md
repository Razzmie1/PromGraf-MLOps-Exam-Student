# Exam of the Prometheus & Grafana course.

### Additional Informations :

I chose the data drift detected status from evidently, which is important for monitoring as it possibly explains why the model performance decreases.

In the Grafana UI I chose the alert on a high RMSE with a threshold of 30.

For the traffic generation and fire alert script, I defined a service similar to the evaluation service and adjusted its script accordingly.
The alert can be triggered by making an evaluation on the 3rd week data where the RMSE exceeds the threshold of 30.
Therefore, I reused the evaluation script and only adjusted the time range.

In the Makefile are the following commands defined:

- `all`: Starts the API and the monitoring setup
- `stop`: Stops all running services
- `evaluation`: Starts the evaluation service, which executes [run_evaluation.py](./src/evaluation/run_evaluation.py)
- `traffic-generation`: Starts the traffic-generation service, which executes [generate_traffic.py](./src/traffic_generation/generate_traffic.py)
- `fire-alert`: Starts the trigger-rmse service, which executes [trigger.py](./src/trigger_rmse/trigger.py)

### Repository Structure :

```
├── deployment
│   ├── prometheus/
│   │   └── prometheus.yml
├── src/
│   ├── api/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── evaluation/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── run_evaluation.py
├── docker-compose.yml
└── Makefile
```

#### Exam Context

You are tasked with implementing a comprehensive monitoring solution for a bike sharing prediction model (`cnt`) based on the "Bike Sharing UCI" dataset. The goal is to ensure that the model's performance and data drifts are constantly monitored, visualized, and alertable, with a particular focus on automating the creation of Grafana dashboards.

You will start from the following Git repository:

https://github.com/DataScientest/PromGraf-MLOps-Exam-Student

This one contains an empty base structure for the API and configurations.

**Key variables of the dataset:**
*   **Target variable (`target`):** `cnt`
*   **Numerical variables:** `temp`, `atemp`, `hum`, `windspeed`, `mnth`, `hr`, `weekday`
*   **Categorical variables:** `season`, `holiday`, `workingday`, `weathersit`

#### General Instructions

1.  **Code and Configuration Quality:** The code must be clean, commented, and the configurations clear and well-structured.
2.  **Reproducibility:** The project must be able to run and function on another machine by executing a simple command `make`.
3.  **Automation:** Favor automation whenever possible, especially for setting up Grafana dashboards.
4.  **Versioning:** Ensure that all necessary files (code, configurations, JSON dashboards) are included in your submission.

#### Specific Tasks

To succeed in this exam, you will need to implement the following points:

**I. Preparation of the Environment and the API:**

*   **API Construction:**
    *   You will need to **build the FastAPI** (`src/api/main.py`) for a regression model predicting the number of bikes (`cnt`).
    *   **Integrate the data loading and preparation functions** (`_fetch_data`, `_process_data`) as well as **training the `RandomForestRegressor`** (`_train_and_predict_reference_model`) on the data from January 2011. The model should be trained only once (for example, at the startup of the API container or via a dedicated `make train` target) and loaded for inference.
    *   Your API should expose an endpoint `/predict` that accepts the features from the `Bike Sharing` dataset (the `BikeSharingInput` class provided) and returns a prediction.
    *   Ensure that the `Dockerfile` and `requirements.txt` of your API are correct (including all necessary dependencies).
*   Configure the `docker-compose.yml` to launch:
    *   Your API (which you will name `bike-api`, on port 8080).
    *   Prometheus (on port 9090).
    *   Grafana (on port 3000).
    *   `node-exporter` (on port 9100) for monitoring the host infrastructure.

**II. API Instrumentation and Metrics Collection in Prometheus:**

*   In the file `api/main.py`:
    *   Define and increment the following metrics (using `prometheus_client` and your `CollectorRegistry`):
        *   `api_requests_total` (Counter, with labels `endpoint`, `method`, `status_code`).
        *   `api_request_duration_seconds` (Histogram, with labels `endpoint`, `method`, `status_code`).
        *   `model_rmse_score` (Gauge, for the Root Mean Squared Error of the regression model, updated via the `/evaluate` endpoint).
        *   `model_mae_score` (Gauge, for the Mean Absolute Error of the regression model, updated via the `/evaluate` endpoint).
        *   `model_r2_score` (Gauge, for the R-squared score of the regression model, updated via the `/evaluate` endpoint).
        *   **A metric of your choice:** Implement **an additional metric deemed relevant** for monitoring this regression model and drifts (for example, `model_mape_score`, `evidently_data_drift_detected_status` (Gauge), or the drift score for a specific feature). Briefly justify (in code comments or a quick `README`) why this metric is relevant.
    *   Implement the `/evaluate` endpoint:
        *   It will accept "current" data for a period (e.g., one week in February).
        *   It will use the trained model to make predictions on this data.
        *   It will execute an **Evidently report** (`RegressionPreset` or `DataDriftPreset`) using the January data (reference) and the provided "current" data.
        *   **It will extract the `RMSE`, `MAE`, `R2Score`, and your chosen metric** from the Evidently report results (use the Evidently documentation or the structure of `Report` / `Snapshot` / `Metric` objects for this) and update the corresponding Prometheus `Gauge` or `Counter`.
        *   The class `EvaluationReportOutput` gives you an example of the expected output format.
    *   Expose all these metrics via the `/metrics` endpoint.
*   Configure `deployment/prometheus/prometheus.yml` to:
    *   Scrape your API `bike-api`.
    *   Scrape `node-exporter`.
*   Create a file `deployment/prometheus/rules/alert_rules.yml` and configure at least one Prometheus alert rule (for example, if the API is `down`).

**III. Automated Visualization with Grafana:**

*   Configure Grafana to run with Docker Compose.
*   **Implement "Dashboards as Code":**
    *   Create a folder `deployment/grafana/dashboards/` and place JSON files of dashboards previously created (by yourself via the interface, retrieved from the Grafana Hub, etc.) in it.
    *   Configure Grafana provisioning via a YAML file (for example, `deployment/grafana/provisioning/dashboards.yaml`) to automatically load these dashboards on startup.
*   **Create three distinct dashboards:**
    *   **Dashboard "API Performance":** Must include panels for request rate, latency (P95), and error rate of your API.
    *   **Dashboard "Model Performance & Drift":** Must include panels for model scores (`model_rmse_score`, `model_mae_score`, `model_r2_score`, etc.) and the custom metric you have chosen (e.g., data drift score).
    *   **Dashboard "Infrastructure Overview":** Must include panels for CPU usage, RAM, and disk space (via `node-exporter`).

> You are free to add panels for each dashboard if the metrics they track are relevant to the dashboard in question.

**IV. Alerting (Prometheus and Grafana):**

*   You must have an alert configured in Prometheus, as requested above.
*   **Configure at least one alert directly in the Grafana interface.** This alert should be based on an ML metric (e.g., if the model's RMSE exceeds a threshold, or if your chosen drift metric indicates a problem).

**V. Traffic Simulation and Evaluation:**

*   The Python script `run_evaluation.py` will be **provided** to you. This script will load a sample of data from the "Bike Sharing" dataset for specific periods (e.g., weeks of February) and send this data to the `/evaluate` endpoint of your API. You will need to ensure that your `/evaluate` endpoint accepts the data format sent by this script.
*   Provide a simple script (or a repeated `curl` command in a shell script) to generate traffic on the `/predict` endpoint to simulate real usage of the API.

**VI. Deliverables :**

*   Create a **`.tar` or `.zip` archive** of your entire project.
*   The archive must contain all necessary files (your `docker-compose.yml`, your `src` folder, your `deployment` folder, your `makefile`, etc.).
*   The project must contain a **`Makefile` at the root** with the following targets:
    *   `all`: To start all services (API, Prometheus, Grafana, Node Exporter).
    *   `stop`: To stop all services.
    *   `evaluation`: To run the `run_evaluation.py` script, which will update the metrics in Prometheus.
    *   `fire-alert`: To intentionally trigger one of the alerts you have configured (e.g., an endpoint `/trigger-drift` in your API that simulates drift by returning extreme values to Evidently, or by forcing the RMSE to be very high for a small batch of data). You will need to justify which alert is being tested.
