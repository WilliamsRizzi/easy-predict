import os
import json
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, redirect
from anomaly_detection.app import blueprint as anomaly_detection_blueprint

app = Flask(__name__)
app.register_blueprint(anomaly_detection_blueprint)

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables.
# ---------------------------------------------------------------------------
PAY_TO_ADDRESS = os.environ.get('PAY_TO_ADDRESS', '0xc99b83818c8865340AC55C45554f377f41c68DBC')
X402_ASSET     = os.environ.get('X402_ASSET',     '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913')
X402_NETWORK   = os.environ.get('X402_NETWORK',   'eip155:8453')
# 0.01 USDC in atomic units (USDC has 6 decimals → 0.01 * 10^6 = 10000)
X402_MAX_AMOUNT = os.environ.get('X402_MAX_AMOUNT', '10000')
FACILITATOR_URL = os.environ.get('FACILITATOR_URL', 'https://x402.org/facilitator')
PUBLIC_BASE_URL = os.environ.get('PUBLIC_BASE_URL', 'https://easy-predict.com').rstrip('/')

# Absolute path to the public/ directory for static files.
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'public')


# ---------------------------------------------------------------------------
# x402 helpers
# ---------------------------------------------------------------------------

def _payment_requirements() -> dict:
    """x402 v2 PaymentRequirements — resource/description/mimeType live in PaymentRequired.resource."""
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
    """Return HTTP 402 with x402 v2 payment challenge in PAYMENT-REQUIRED header + body."""
    body = {
        'x402Version': 2,
        'error':       'Payment Required',
        'resource': {
            'url':         resource_url,
            'description': 'Predict the next value in a numeric series using automatic model selection (linear, log1p-linear, last-delta, mean).',
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
    """Accept v2 PAYMENT-SIGNATURE or legacy X-PAYMENT header."""
    return bool(
        request.headers.get('PAYMENT-SIGNATURE') or
        request.headers.get('Payment-Signature') or
        request.headers.get('X-PAYMENT') or
        request.headers.get('X-Payment')
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_log1p(series: list) -> tuple[float, float, float]:
    """Log1p linear extrapolation: fit in log1p space, predict the next point."""
    arr = np.array(series, dtype=float)
    n = len(arr)
    if n < 3 or n > 1000:
        raise ValueError('Series length must be between 3 and 1000')
    y = np.log1p(arr)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    prediction = float(np.expm1(slope * n + intercept))
    return prediction, float(slope), float(intercept)


def predict_next(series: list) -> dict:
    """Model-selection prediction.

    Holds out the last point, evaluates four candidate models against it,
    then retrains the winner on the full series to produce the final prediction.

    Models
    ------
    linear       — OLS on raw values
    log1p-linear — OLS in log1p space (skipped if any value ≤ -1)
    last-delta   — last value + mean first-difference (naive drift)
    mean         — global mean (best for stationary/mean-reverting series)
    """
    arr = np.array(series, dtype=float)
    n = len(arr)
    if n < 3 or n > 1000:
        raise ValueError('Series length must be between 3 and 1000')

    train = arr[:-1]
    holdout_val = float(arr[-1])
    m = len(train)
    candidates: dict = {}

    # linear
    try:
        sl, ic = np.polyfit(np.arange(m, dtype=float), train, 1)
        h_err = abs(float(sl * m + ic) - holdout_val)
        sl_f, ic_f = np.polyfit(np.arange(n, dtype=float), arr, 1)
        candidates['linear'] = {
            'holdout_error': h_err,
            'prediction': float(sl_f * n + ic_f),
            'slope': round(float(sl_f), 6),
            'intercept': round(float(ic_f), 6),
        }
    except Exception:
        pass

    # log1p-linear — only valid when all values > -1
    if np.all(arr > -1):
        try:
            sl, ic = np.polyfit(np.arange(m, dtype=float), np.log1p(train), 1)
            h_err = abs(float(np.expm1(sl * m + ic)) - holdout_val)
            sl_f, ic_f = np.polyfit(np.arange(n, dtype=float), np.log1p(arr), 1)
            candidates['log1p-linear'] = {
                'holdout_error': h_err,
                'prediction': float(np.expm1(sl_f * n + ic_f)),
                'slope': round(float(sl_f), 6),
                'intercept': round(float(ic_f), 6),
            }
        except Exception:
            pass

    # last-delta: last value + mean first-difference
    try:
        h_err = abs(float(train[-1]) + float(np.mean(np.diff(train))) - holdout_val)
        candidates['last-delta'] = {
            'holdout_error': h_err,
            'prediction': float(arr[-1]) + float(np.mean(np.diff(arr))),
        }
    except Exception:
        pass

    # mean: global mean (best for stationary series)
    try:
        candidates['mean'] = {
            'holdout_error': abs(float(np.mean(train)) - holdout_val),
            'prediction': float(np.mean(arr)),
        }
    except Exception:
        pass

    if not candidates:
        raise ValueError('No model could be fit to this series')

    winner_name = min(candidates, key=lambda k: candidates[k]['holdout_error'])
    winner = candidates[winner_name]
    result: dict = {
        'prediction': winner['prediction'],
        'method': winner_name,
        'holdout_errors': {k: round(v['holdout_error'], 6) for k, v in candidates.items()},
    }
    if 'slope' in winner:
        result['slope'] = winner['slope']
        result['intercept'] = winner['intercept']
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Splash page — served from the repo-root index.html."""
    return send_from_directory(ROOT_DIR, 'index.html')


@app.route('/timeseries', methods=['GET'])
def timeseries_get():
    if 'application/json' in request.headers.get('Accept', ''):
        return _payment_required(f'{PUBLIC_BASE_URL}/timeseries')
    return send_from_directory(ROOT_DIR, 'index.html')


@app.route('/timeseries', methods=['POST'])
def timeseries_post():
    """
    Predict the next value in a numeric time series.

    Payment gate runs FIRST so unpaid probes (e.g. from x402scan) always
    receive a 402 challenge with the full payment terms — never a 400 from
    body validation or a 403 from identity checks.
    """
    resource_url = f'{PUBLIC_BASE_URL}/timeseries'

    if not _has_payment():
        return _payment_required(resource_url)

    if not request.is_json:
        return jsonify(error='JSON body required'), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error='Invalid JSON body'), 400

    context = None
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
    else:
        series = None

    if series is None:
        return jsonify(error="Provide a 'series' key with a number array, or send a bare JSON array"), 400

    try:
        result = predict_next(series)
    except (ValueError, Exception) as exc:
        return jsonify(error=str(exc)), 400

    if context is not None:
        result['context'] = context
    return jsonify(result)


@app.route('/llms.txt')
def llms_txt():
    return send_from_directory(ROOT_DIR, 'llms.txt', mimetype='text/plain')


@app.route('/llm.txt')
def llm_txt_redirect():
    return redirect('/llms.txt', 301)


@app.route('/openapi.json')
def openapi_spec():
    """Serve the static openapi.json discovery document. Free, no payment."""
    return send_from_directory(ROOT_DIR, 'openapi.json')


@app.route('/.well-known/x402')
def well_known_x402():
    """Machine-readable x402 v2 resource list. Free, no payment."""
    return jsonify({
        'x402Version': 2,
        'openapi':     f'{PUBLIC_BASE_URL}/openapi.json',
        'resources': [
            {
                'resource': {
                    'url':         f'{PUBLIC_BASE_URL}/timeseries',
                    'description': 'Predict the next value in a numeric series using automatic model selection (linear, log1p-linear, last-delta, mean).',
                    'mimeType':    'application/json',
                },
                'method':  'POST',
                'accepts': [_payment_requirements()],
            },
            {
                'resource': {
                    'url':         f'{PUBLIC_BASE_URL}/anomaly-detection',
                    'description': 'Detect anomalies in a numeric series using z-score method.',
                    'mimeType':    'application/json',
                },
                'method':  'POST',
                'accepts': [_payment_requirements()],
            },
        ],
    })


@app.route('/favicon.ico')
@app.route('/favicon.svg')
def favicon():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="7" fill="#0a1628"/>'
        '<polyline points="5,24 9,18 14,14 19,12" fill="none" stroke="white" stroke-width="0.7" stroke-linecap="round" stroke-linejoin="round"/>'
        '<polyline points="19,12 25,8 27,15 21,19 19,12" fill="none" stroke="white" stroke-width="0.7" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="5"  cy="24" r="1.2" fill="white"/>'
        '<circle cx="9"  cy="18" r="1.2" fill="white"/>'
        '<circle cx="14" cy="14" r="1.2" fill="white"/>'
        '<circle cx="19" cy="12" r="1.2" fill="white"/>'
        '<circle cx="25" cy="8"  r="1.2" fill="white"/>'
        '<circle cx="27" cy="15" r="1.2" fill="white"/>'
        '<circle cx="21" cy="19" r="1.2" fill="white"/>'
        '</svg>'
    )
    return svg, 200, {
        'Content-Type':  'image/svg+xml',
        'Cache-Control': 'public, max-age=86400',
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
