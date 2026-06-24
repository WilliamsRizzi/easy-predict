import os
import pytest
from timeseries.app import predict_log1p


# ---------------------------------------------------------------------------
# Unit tests — prediction function
# ---------------------------------------------------------------------------

def test_predict_valid():
    pred, slope, intercept = predict_log1p([1, 2, 3, 4, 5])
    assert isinstance(pred, float)
    assert pred > 0

def test_predict_too_short():
    with pytest.raises(ValueError):
        predict_log1p([1, 2])

def test_predict_too_long():
    with pytest.raises(ValueError):
        predict_log1p(list(range(11)))


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

def test_index(client):
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'easy-predict' in resp.data

def test_timeseries_get(client):
    resp = client.get('/timeseries')
    assert resp.status_code == 200
    assert b'easy-predict' in resp.data

def test_openapi_json(client):
    resp = client.get('/openapi.json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['openapi'].startswith('3')
    assert '/timeseries' in data['paths']
    paid = data['paths']['/timeseries']['post']
    assert paid['security'] == [{'x402': []}]
    assert paid['x-payment-info']['x402']['accepts'][0]['payTo'].startswith('0x')

def test_well_known_x402(client):
    resp = client.get('/.well-known/x402')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['x402Version'] == 1
    assert data['openapi'].endswith('/openapi.json')
    assert data['resources'][0]['method'] == 'POST'
    assert data['resources'][0]['accepts'][0]['scheme'] == 'exact'

def test_favicon(client):
    resp = client.get('/favicon.ico')
    assert resp.status_code == 200
    assert b'svg' in resp.data


# ---------------------------------------------------------------------------
# x402 paywall — unpaid probes must get 402, never 400/403
# ---------------------------------------------------------------------------

def test_unpaid_probe_returns_402(client):
    """Simulate x402scan: bare POST with no payment, no body, no headers."""
    original = os.environ.pop('X402_TOKEN', None)
    try:
        resp = client.post('/timeseries')
    finally:
        if original is not None:
            os.environ['X402_TOKEN'] = original
    assert resp.status_code == 402
    body = resp.get_json()
    assert body['x402Version'] == 1
    assert body['accepts'][0]['scheme'] == 'exact'
    assert body['accepts'][0]['network'] == 'base'
    assert body['accepts'][0]['payTo'].startswith('0x')
    assert resp.headers.get('X-Payment-Required')
    assert resp.headers.get('PAYMENT-REQUIRED')

def test_payment_gate_before_body_validation(client):
    """No payment + no body → 402, not 400. Gate runs before parsing."""
    resp = client.post('/timeseries', content_type='application/json', data='')
    assert resp.status_code == 402


# ---------------------------------------------------------------------------
# Paid requests
# ---------------------------------------------------------------------------

def test_paid_object_body(client):
    resp = client.post('/timeseries',
                       json={'series': [1, 2, 3, 4, 5]},
                       headers={'X-PAYMENT': 'signed-payload'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'prediction' in data
    assert data['method'] == 'log1p-linear-extrapolation'
    assert 'slope' in data
    assert 'intercept' in data

def test_paid_array_body(client):
    resp = client.post('/timeseries',
                       json=[1.0, 2.3, 4.1, 6.8, 9.2],
                       headers={'X-PAYMENT': 'signed-payload'})
    assert resp.status_code == 200
    assert 'prediction' in resp.get_json()

def test_paid_x_payment_case_insensitive(client):
    """Flask normalises headers; both X-PAYMENT and X-Payment should work."""
    resp = client.post('/timeseries',
                       json=[1, 2, 3],
                       headers={'X-Payment': 'signed-payload'})
    assert resp.status_code == 200

def test_paid_bad_series_length(client):
    resp = client.post('/timeseries',
                       json=[1, 2],
                       headers={'X-PAYMENT': 'signed-payload'})
    assert resp.status_code == 400

def test_paid_no_series_key(client):
    resp = client.post('/timeseries',
                       json={'foo': [1, 2, 3]},
                       headers={'X-PAYMENT': 'signed-payload'})
    assert resp.status_code == 400

def test_paid_not_json(client):
    resp = client.post('/timeseries',
                       data='not json',
                       content_type='text/plain',
                       headers={'X-PAYMENT': 'signed-payload'})
    assert resp.status_code == 400
