from flask import Flask, request, jsonify
import numpy as np
import os
import time
import json
import base64
from collections import defaultdict, deque
from functools import wraps
from threading import Lock

app = Flask(__name__)


# ---------------------------------------------------------------------------
# x402 configuration
#
# These describe how the protected endpoints are paid for. They are all
# overridable via environment variables so the same code can run against
# testnet/mainnet or a different receiving wallet without code changes.
# Defaults target Base mainnet with USDC.
# ---------------------------------------------------------------------------
X402_VERSION = int(os.environ.get('X402_VERSION', '1'))
# Wallet that receives payments (the x402 `payTo` address).
PAY_TO_ADDRESS = os.environ.get('PAY_TO_ADDRESS', '0xc99b83818c8865340AC55C45554f377f41c68DBC')
# Settlement network. "base" == Base mainnet (CAIP-2 eip155:8453).
X402_NETWORK = os.environ.get('X402_NETWORK', 'base')
# USDC contract on Base mainnet.
X402_ASSET = os.environ.get('X402_ASSET', '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913')
X402_ASSET_NAME = os.environ.get('X402_ASSET_NAME', 'USD Coin')
# EIP-712 domain version for USDC on Base (used by the `exact` EVM scheme).
X402_ASSET_EIP712_VERSION = os.environ.get('X402_ASSET_EIP712_VERSION', '2')
# Human-readable price and its on-chain atomic amount (USDC has 6 decimals,
# so 0.001 USDC == 1000 atomic units).
X402_PRICE_USD = os.environ.get('X402_PRICE_USD', '0.001')
X402_MAX_AMOUNT_REQUIRED = os.environ.get('X402_MAX_AMOUNT_REQUIRED', '1000')
X402_MAX_TIMEOUT_SECONDS = int(os.environ.get('X402_MAX_TIMEOUT_SECONDS', '60'))
# Facilitator that verifies/settles payments. Overridable; for Base mainnet
# this is typically the Coinbase CDP facilitator.
X402_FACILITATOR_URL = os.environ.get('X402_FACILITATOR_URL', 'https://x402.org/facilitator')

# Paths that sit behind the paywall (POST only).
PAID_PATHS = ('/log1p', '/timeseries/log1p')


def x402_payment_requirements(resource_url):
    """Build a single x402 `PaymentRequirements` object for a resource."""
    return {
        'scheme': 'exact',
        'network': X402_NETWORK,
        'maxAmountRequired': X402_MAX_AMOUNT_REQUIRED,
        'resource': resource_url,
        'description': 'Predict the next value in a numeric series via log1p linear extrapolation.',
        'mimeType': 'application/json',
        'payTo': PAY_TO_ADDRESS,
        'maxTimeoutSeconds': X402_MAX_TIMEOUT_SECONDS,
        'asset': X402_ASSET,
        'extra': {
            'name': X402_ASSET_NAME,
            'version': X402_ASSET_EIP712_VERSION,
        },
    }


def x402_payment_required_response(error_message, resource_url=None):
    """Return a fully-formed HTTP 402 response with x402 payment metadata.

    The body follows the x402 wire format (``x402Version`` + ``accepts``) that
    facilitators and x402 client libraries parse. The same payload is echoed in
    the ``X-Payment-Required`` / ``PAYMENT-REQUIRED`` headers (x402 v2 naming)
    so clients can read it from either place.
    """
    if resource_url is None:
        resource_url = request.base_url
    requirements = x402_payment_requirements(resource_url)
    body = {
        'x402Version': X402_VERSION,
        'error': error_message,
        'accepts': [requirements],
        'facilitator': X402_FACILITATOR_URL,
    }
    resp = jsonify(body)
    resp.status_code = 402
    encoded = base64.b64encode(json.dumps(body).encode('utf-8')).decode('ascii')
    resp.headers['X-Payment-Required'] = encoded
    resp.headers['PAYMENT-REQUIRED'] = encoded
    resp.headers['Access-Control-Expose-Headers'] = 'X-Payment-Required, PAYMENT-REQUIRED'
    return resp


def x402_block_reason():
    """Decide whether the current request may pass the paywall.

    Returns ``None`` when the request is allowed, otherwise a ``(kind, message)``
    tuple where ``kind`` is ``'forbidden'`` (403, wrong caller identity) or
    ``'payment'`` (402, payment missing/invalid).
    """
    agent = request.headers.get('X-Agent-Type')
    header_cost = request.headers.get('X-402-Cost')
    header_token = request.headers.get('X-402')
    header_facilitator = request.headers.get('X-402-Facilitator')
    # A real x402 client settles by sending the signed payment payload here.
    header_payment = request.headers.get('X-PAYMENT') or request.headers.get('X-Payment')
    env_token = os.environ.get('X402_TOKEN')

    # Identity gate: this service is for AI agents only.
    if agent != 'ai':
        return ('forbidden', 'Forbidden: only AI agents allowed')
    # Price gate.
    if header_cost != X402_PRICE_USD:
        return ('payment', 'Payment Required: missing or incorrect X-402-Cost; expected %s' % X402_PRICE_USD)
    # Presence of a real x402 payment payload satisfies the paywall.
    if header_payment:
        return None
    # Otherwise fall back to the configured shared payment token.
    if env_token:
        if header_token != env_token and header_facilitator != env_token:
            return ('payment', 'Payment Required: missing or invalid x402 payment')
    else:
        if not header_token and not header_facilitator:
            return ('payment', 'Payment Required: X-402 token, facilitator, or X-PAYMENT required')
    return None


def _respond_to_block(reason):
    kind, message = reason
    if kind == 'forbidden':
        return jsonify(error=message), 403
    return x402_payment_required_response(message)


# Enforce X402 paywall early so probes receive a 402 (with payment metadata)
# before request body validation.
@app.before_request
def enforce_x402_paywall():
    if request.method != 'POST':
        return None
    if request.path.rstrip('/') not in PAID_PATHS:
        return None
    reason = x402_block_reason()
    if reason is None:
        return None
    return _respond_to_block(reason)


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
        reason = x402_block_reason()
        if reason is None:
            return f(*args, **kwargs)
        return _respond_to_block(reason)
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
            'x-guidance': 'Use POST /log1p or /timeseries/log1p with X-Agent-Type, X-402-Cost, and an x402 payment (X-PAYMENT header) or X-402 / X-402-Facilitator token to request a prediction. Calling without payment returns HTTP 402 with payment requirements in the response body `accepts` array.'
        },
        'servers': [{'url': base_url}],
        # Global x402 defaults so indexers can resolve payment context once.
        'x-x402': {
            'version': X402_VERSION,
            'network': X402_NETWORK,
            'asset': X402_ASSET,
            'payTo': PAY_TO_ADDRESS,
            'facilitator': X402_FACILITATOR_URL,
        },
        'paths': {
            '/log1p': {
                'get': {
                    'summary': 'API information',
                    'description': 'Free, human-readable usage information for the log1p endpoint.',
                    'security': [],
                    'responses': {'200': {'description': 'Information about the log1p endpoint'}}
                },
                'post': get_log1p_operation(base_url + '/log1p')
            },
            '/timeseries/log1p': {
                'get': {
                    'summary': 'API information',
                    'description': 'Free, human-readable usage information for the log1p endpoint.',
                    'security': [],
                    'responses': {'200': {'description': 'Information about the log1p endpoint'}}
                },
                'post': get_log1p_operation(base_url + '/timeseries/log1p')
            },
            '/openapi.json': {
                'get': {
                    'summary': 'OpenAPI discovery document',
                    'description': 'This OpenAPI document. Free to fetch.',
                    'security': [],
                    'responses': {'200': {'description': 'OpenAPI 3.1 document describing this API'}}
                }
            },
            '/.well-known/x402': {
                'get': {
                    'summary': 'x402 discovery document',
                    'description': 'Machine-readable list of paid x402 resources and a pointer to the OpenAPI document. Free to fetch.',
                    'security': [],
                    'responses': {'200': {'description': 'x402 resource discovery payload'}}
                }
            }
        },
        'components': {
            'securitySchemes': {
                'x402': {
                    'type': 'apiKey',
                    'in': 'header',
                    'name': 'X-PAYMENT',
                    'description': 'x402 micropayment. Send a base64-encoded signed payment payload in the X-PAYMENT header. Without it the endpoint returns HTTP 402 with the required payment terms in the response body `accepts` array.'
                }
            }
        }
    }


def get_log1p_operation(resource_url):
    return {
        # Paid: this operation is gated by the x402 security scheme.
        'security': [{'x402': []}],
        'parameters': [
            {'name': 'X-Agent-Type', 'in': 'header', 'required': True, 'schema': {'type': 'string'}, 'description': 'must be ai'},
            {'name': 'X-402-Cost', 'in': 'header', 'required': True, 'schema': {'type': 'string'}, 'description': 'must be 0.001'},
            {'name': 'X-PAYMENT', 'in': 'header', 'required': False, 'schema': {'type': 'string'}, 'description': 'x402 signed payment payload (base64)'},
            {'name': 'X-402', 'in': 'header', 'required': False, 'schema': {'type': 'string'}, 'description': 'Coinbase X402 token'},
            {'name': 'X-402-Facilitator', 'in': 'header', 'required': False, 'schema': {'type': 'string'}, 'description': 'Facilitator token header alternative to X-402'}
        ],
        'summary': 'Predict the next value for a numeric series',
        'description': 'Accepts a JSON array or an object containing a series array and returns the next predicted value. Paid via x402 micropayment.',
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
            '402': {
                'description': 'Payment Required. Body contains x402 payment requirements.',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'x402Version': {'type': 'integer'},
                                'error': {'type': 'string'},
                                'accepts': {'type': 'array', 'items': {'type': 'object'}},
                                'facilitator': {'type': 'string'}
                            },
                            'required': ['x402Version', 'accepts']
                        }
                    }
                }
            },
            '403': {'description': 'Forbidden'},
            '429': {'description': 'Too Many Requests'}
        },
        'x-payment-info': {
            'price': {
                'fixed': {'mode': 'fixed', 'currency': 'USD', 'amount': X402_PRICE_USD},
                'dynamic': {'mode': 'dynamic', 'currency': 'USD', 'min': X402_PRICE_USD, 'max': X402_PRICE_USD}
            },
            'protocols': [{'x402': {}}],
            'x402': {
                'version': X402_VERSION,
                'accepts': [x402_payment_requirements(resource_url)],
                'facilitator': X402_FACILITATOR_URL
            }
        }
    }


def get_x402_discovery(base_url=None):
    """Machine-readable x402 resource list for /.well-known/x402."""
    if base_url is None:
        base_url = request.host_url.rstrip('/')
    resources = []
    for path in PAID_PATHS:
        resource_url = base_url + path
        resources.append({
            'resource': resource_url,
            'method': 'POST',
            'accepts': [x402_payment_requirements(resource_url)],
        })
    return {
        'x402Version': X402_VERSION,
        'openapi': base_url + '/openapi.json',
        'facilitator': X402_FACILITATOR_URL,
        'resources': resources,
    }


@app.route('/openapi.json', methods=['GET'])
def openapi_json():
    return jsonify(get_openapi_spec())


@app.route('/.well-known/x402', methods=['GET'])
def well_known_x402():
    return jsonify(get_x402_discovery())


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
        "<li>X-PAYMENT: &lt;x402 payload&gt; (or X-402 / X-402-Facilitator token)</li>"
        "</ul>"
        "<p>See <a href=\"/openapi.json\">/openapi.json</a> and "
        "<a href=\"/.well-known/x402\">/.well-known/x402</a> for discovery.</p>"
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
