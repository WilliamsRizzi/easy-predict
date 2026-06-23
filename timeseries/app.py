from flask import Flask, request, jsonify
import numpy as np
import os
from functools import wraps

app = Flask(__name__)


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


@app.route('/log1p', methods=['POST'])
@app.route('/timeseries/log1p', methods=['POST'])
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


@require_x402
def log1p_predict():
    data = request.get_json()
    if data is None:
        return jsonify(error="JSON body required"), 400
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
