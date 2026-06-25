import os
import json
import base64
import numpy as np
from flask import Blueprint, request, jsonify, send_from_directory

blueprint = Blueprint('anomaly_detection', __name__)

PAY_TO_ADDRESS  = os.environ.get('PAY_TO_ADDRESS',  '0xc99b83818c8865340AC55C45554f377f41c68DBC')
X402_ASSET      = os.environ.get('X402_ASSET',      '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913')
X402_NETWORK    = os.environ.get('X402_NETWORK',    'eip155:8453')
X402_MAX_AMOUNT = os.environ.get('X402_MAX_AMOUNT', '10000')
PUBLIC_BASE_URL = os.environ.get('PUBLIC_BASE_URL', 'https://easy-predict.com').rstrip('/')

ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'public')


# ---------------------------------------------------------------------------
# x402 helpers
# ---------------------------------------------------------------------------

def _payment_requirements() -> dict:
    return {
        'scheme':            'exact',
        'network':           X402_NETWORK,
        'asset':             X402_ASSET,
        'amount':            X402_MAX_AMOUNT,
        'payTo':             PAY_TO_ADDRESS,
        'maxTimeoutSeconds': 60,
        'extra':             {'name': 'USD Coin', 'version': '2'},
    }


def _payment_required(resource_url: str):
    body = {
        'x402Version': 2,
        'error':       'Payment Required',
        'resource': {
            'url':         resource_url,
            'description': 'Detect anomalies in a numeric series using z-score method.',
            'mimeType':    'application/json',
        },
        'accepts': [_payment_requirements()],
    }
    encoded = base64.b64encode(json.dumps(body).encode()).decode()
    resp = jsonify(body)
    resp.status_code = 402
    resp.headers['PAYMENT-REQUIRED'] = encoded
    resp.headers['Access-Control-Expose-Headers'] = 'PAYMENT-REQUIRED'
    return resp


def _has_payment() -> bool:
    return bool(
        request.headers.get('PAYMENT-SIGNATURE') or
        request.headers.get('Payment-Signature') or
        request.headers.get('X-PAYMENT') or
        request.headers.get('X-Payment')
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_anomalies(series: list, threshold: float = 2.0) -> dict:
    """Z-score anomaly detection: flag points where |z| > threshold."""
    arr = np.array(series, dtype=float)
    n = len(arr)
    if n < 3 or n > 1000:
        raise ValueError('Series length must be between 3 and 1000')
    if threshold <= 0 or threshold > 10:
        raise ValueError('threshold must be between 0 (exclusive) and 10')
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std == 0:
        anomalies = []
    else:
        z_scores = (arr - mean) / std
        anomalies = [
            {'index': int(i), 'value': float(arr[i]), 'z_score': round(float(z_scores[i]), 6)}
            for i in range(n) if abs(z_scores[i]) > threshold
        ]
    return {'anomalies': anomalies, 'method': 'z-score', 'mean': mean, 'std': std, 'threshold': threshold}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@blueprint.route('/anomaly-detection', methods=['GET'])
def anomaly_detection_get():
    return send_from_directory(ROOT_DIR, 'index.html')


@blueprint.route('/anomaly-detection', methods=['POST'])
def anomaly_detection_post():
    """
    Detect anomalies in a numeric series using z-score method.

    Payment gate runs FIRST so unpaid probes always receive a 402.
    """
    resource_url = f'{PUBLIC_BASE_URL}/anomaly-detection'

    if not _has_payment():
        return _payment_required(resource_url)

    if not request.is_json:
        return jsonify(error='JSON body required'), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error='Invalid JSON body'), 400

    context = None
    threshold = 2.0
    if isinstance(data, list):
        series = data
    elif isinstance(data, dict):
        series = data.get('series')
        raw_context = data.get('context')
        if raw_context is not None:
            if not isinstance(raw_context, str):
                return jsonify(error="'context' must be a string"), 400
            if len(raw_context) > 200:
                return jsonify(error="'context' must be 200 characters or fewer"), 400
            context = raw_context
        raw_threshold = data.get('threshold')
        if raw_threshold is not None:
            if not isinstance(raw_threshold, (int, float)) or isinstance(raw_threshold, bool):
                return jsonify(error="'threshold' must be a number"), 400
            threshold = float(raw_threshold)
    else:
        series = None

    if series is None:
        return jsonify(error="Provide a 'series' key with a number array, or send a bare JSON array"), 400

    try:
        result = detect_anomalies(series, threshold)
    except (ValueError, Exception) as exc:
        return jsonify(error=str(exc)), 400

    if context is not None:
        result['context'] = context
    return jsonify(result)
