from flask import Flask, request, jsonify
import numpy as np

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
