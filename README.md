# easy-predict

`easy-predict` is a lightweight Flask-based service for time series prediction. It exposes a simple endpoint to extrapolate the next value from a numeric series using a log1p linear regression model.

## What this repo contains

- `timeseries/app.py` — the Flask application implementing `/log1p` and `/timeseries/log1p`
- `tests/test_app.py` — unit tests for prediction behavior, request validation, and rate limiting
- `requirements.txt` / `pyproject.toml` — project dependencies and Python packaging metadata

## Try it live

Visit **easy-predict.com** to see the project in action and test the prediction endpoint.

## Local usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python timeseries/app.py
   ```
3. Send a POST request to:
   - `http://localhost:8000/timeseries/log1p`
   - `http://localhost:8000/log1p`

Required headers for POST requests:

- `X-Agent-Type: ai`
- `X-402-Cost: 0.001`
- `X-402: <token>` or `X-402-Facilitator: <token>` when `X402_TOKEN` is configured

The service also exposes OpenAPI discovery metadata at `/openapi.json`.
