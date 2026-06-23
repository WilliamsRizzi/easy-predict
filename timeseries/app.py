from flask import Flask, request, jsonify
import numpy as np
import os
import time
from collections import defaultdict, deque
from functools import wraps
from threading import Lock

app = Flask(__name__)


# Enforce X402 paywall early so probes receive a 402 before request body validation
@app.before_request
def enforce_x402_paywall():
    # Only apply to prediction POST endpoints
    if request.method != 'POST':
        return None
    path = request.path.rstrip('/')
    if path not in ('/log1p', '/timeseries/log1p'):
        return None
    # perform the same checks as require_x402 but return early
    header_token = request.headers.get('X-402')
    header_facilitator = request.headers.get('X-402-Facilitator')
    header_cost = request.headers.get('X-402-Cost')
    agent = request.headers.get('X-Agent-Type')
    expected_cost = '0.001'
    expected_agent = 'ai'
    env_token = os.environ.get('X402_TOKEN')
    if agent != expected_agent:
        return jsonify(error='Forbidden: only AI agents allowed'), 403
    if header_cost != expected_cost:
        return jsonify(error='Payment Required: incorrect cost'), 402
    if env_token:
        if header_token != env_token and header_facilitator != env_token:
            return jsonify(error='Payment Required: invalid X-402 token'), 402
    else:
        if not header_token and not header_facilitator:
            return jsonify(error='Payment Required: X-402 token or facilitator required'), 402
    return None

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
        header_facilitator = request.headers.get('X-402-Facilitator')
        header_cost = request.headers.get('X-402-Cost')
        agent = request.headers.get('X-Agent-Type')
        expected_cost = '0.001'
        expected_agent = 'ai'
        env_token = os.environ.get('X402_TOKEN')
        if agent != expected_agent:
            return jsonify(error='Forbidden: only AI agents allowed'), 403
        if header_cost != expected_cost:
            return jsonify(error='Payment Required: incorrect cost'), 402
        if env_token:
            if header_token != env_token and header_facilitator != env_token:
                return jsonify(error='Payment Required: invalid X-402 token'), 402
        else:
            if not header_token and not header_facilitator:
                return jsonify(error='Payment Required: X-402 token or facilitator required'), 402
        return f(*args, **kwargs)
    return wrapper


def get_openapi_spec(base_url=None):
    if base_url is None:
        base_url = request.host_url.rstrip('/')
    contact_email = os.environ.get('CONTACT_EMAIL', 'support@easy-predict.com')
    return {
        'openapi': '3.1.0',
        'info': {
            'title': 'easy-predict Time Series Prediction API',
            'version': '1.0.0',
            'description': 'Predict the next value in a numeric series using log1p linear extrapolation.',
            'contact': {'email': contact_email},
            'x-guidance': 'Use POST /log1p or /timeseries/log1p with X-Agent-Type, X-402-Cost, and X-402 or X-402-Facilitator headers to request a prediction.'
        },
        'servers': [{'url': base_url}],
        'paths': {
            '/log1p': {
                'get': {
                    'summary': 'API information',
                    'responses': {'200': {'description': 'Information about the log1p endpoint'}}
                },
                'post': get_log1p_operation()
            },
            '/timeseries/log1p': {
                'get': {
                    'summary': 'API information',
                    'responses': {'200': {'description': 'Information about the log1p endpoint'}}
                },
                'post': get_log1p_operation()
            }
        }
    }


def get_log1p_operation():
    return {
        'parameters': [
            {'name': 'X-Agent-Type', 'in': 'header', 'required': True, 'schema': {'type': 'string'}, 'description': 'must be ai'},
            {'name': 'X-402-Cost', 'in': 'header', 'required': True, 'schema': {'type': 'string'}, 'description': 'must be 0.001'},
            {'name': 'X-402', 'in': 'header', 'required': False, 'schema': {'type': 'string'}, 'description': 'Coinbase X402 token'},
            {'name': 'X-402-Facilitator', 'in': 'header', 'required': False, 'schema': {'type': 'string'}, 'description': 'Facilitator token header alternative to X-402'}
        ],
        'summary': 'Predict the next value for a numeric series',
        'description': 'Accepts a JSON array or an object containing a series array and returns the next predicted value.',
        'requestBody': {
            'required': True,
            'content': {
                'application/json': {
                    'schema': {
                        'oneOf': [
                            {
                                'type': 'array',
                                'items': {'type': 'number'},
                                'minItems': 3,
                                'maxItems': 10
                            },
                            {
                                'type': 'object',
                                'properties': {
                                    'series': {
                                        'type': 'array',
                                        'items': {'type': 'number'},
                                        'minItems': 3,
                                        'maxItems': 10
                                    }
                                },
                                'required': ['series']
                            }
                        ]
                    }
                }
            }
        },
        'responses': {
            '200': {
                'description': 'Prediction result',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'prediction': {'type': 'number'},
                                'method': {'type': 'string'},
                                'slope': {'type': 'number'},
                                'intercept': {'type': 'number'}
                            },
                            'required': ['prediction', 'method', 'slope', 'intercept']
                        }
                    }
                }
            },
            '400': {'description': 'Invalid request'},
            '402': {'description': 'Payment Required'},
            '403': {'description': 'Forbidden'},
            '429': {'description': 'Too Many Requests'}
        },
        'x-payment-info': {
            'price': {
                'fixed': {'mode': 'fixed', 'currency': 'USD', 'amount': '0.001'},
                'dynamic': {'mode': 'dynamic', 'currency': 'USD', 'min': '0.001', 'max': '0.001'}
            },
            'protocols': [{'x402': {}}]
        }
    }


@app.route('/openapi.json', methods=['GET'])
def openapi_json():
    return jsonify(get_openapi_spec())


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
        "<li>X-402: &lt;token&gt; or X-402-Facilitator: &lt;token&gt; if X402_TOKEN is set</li>"
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
