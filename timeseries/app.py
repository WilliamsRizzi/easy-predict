from flask import Flask, request, jsonify
import numpy as np
import os
import time
from collections import defaultdict, deque
from functools import wraps
from threading import Lock

app = Flask(__name__)

RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
rate_limit_records = defaultdict(deque)
rate_limit_lock = Lock()


def rate_limit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        client_id = request.remote_addr or 'unknown'
        now = time.time()
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        with rate_limit_lock:
            records = rate_limit_records[client_id]
            while records and records[0] < cutoff:
                records.popleft()
            if len(records) >= RATE_LIMIT_MAX_REQUESTS:
                return jsonify(error="Too many requests, rate limit exceeded"), 429
            records.append(now)
        return f(*args, **kwargs)
    return wrapper


def reset_rate_limits():
    with rate_limit_lock:
        rate_limit_records.clear()


def predict_next_log1p(series):
    arr = np.array(series, dtype=float)
    if arr.size < 3 or arr.size > 10:
        raise ValueError("Series length must be between 3 and 10.")
    y = np.log1p(arr)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * len(y) + intercept
    pred = np.expm1(y_pred)
    return float(pred), float(slope), float(intercept)


def require_x402(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        header_token = request.headers.get('X-402')
        header_cost = request.headers.get('X-402-Cost')
        agent = request.headers.get('X-Agent-Type')
        expected_cost = '0.001'
        expected_agent = 'ai'
        env_token = os.environ.get('X402_TOKEN')
        if agent != expected_agent:
            return jsonify(error='Forbidden: only AI agents allowed'), 403
        if header_cost != expected_cost:
            return jsonify(error='Forbidden: incorrect cost'), 403
        if env_token:
            if header_token != env_token:
                return jsonify(error='Forbidden: invalid X-402 token'), 403
        else:
            if not header_token:
                return jsonify(error='Forbidden: X-402 token required'), 403
        return f(*args, **kwargs)
    return wrapper


@app.route('/log1p', methods=['GET'])
@app.route('/timeseries/log1p', methods=['GET'])
def log1p_info():
    return (
        "<html><body>"
        "<h1>Timeseries Log1p Predictor</h1>"
        "<p>Send a POST request with JSON body:</p>"
        "<pre>{\"series\": [1, 2, 3]}</pre>"
        "<p>Headers required for POST:</p>"
        "<ul>"
        "<li>X-Agent-Type: ai</li>"
        "<li>X-402-Cost: 0.001</li>"
        "<li>X-402: &lt;token&gt; if X402_TOKEN is set</li>"
        "</ul>"
        "</body></html>"
    )


@app.route('/log1p', methods=['POST'])
@app.route('/timeseries/log1p', methods=['POST'])
@rate_limit
@require_x402
def log1p_predict():
    if not request.is_json:
        return jsonify(error="JSON body required"), 400
    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error="Invalid JSON body"), 400
    # Accept either {"series": [...]} or a raw JSON array
    series = None
    if isinstance(data, dict):
        series = data.get('series')
    elif isinstance(data, list):
        series = data
    if series is None:
        return jsonify(error="Provide 'series' array in JSON body or raw JSON array"), 400
    try:
        pred, slope, intercept = predict_next_log1p(series)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(prediction=pred, method="log1p-linear-extrapolation", slope=slope, intercept=intercept)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
