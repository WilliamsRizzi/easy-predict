import os
import pytest
from timeseries.app import predict_log1p, predict_next
from anomaly_detection.app import detect_anomalies


# ---------------------------------------------------------------------------
# Unit tests — predict_log1p (kept for backward compat)
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
        predict_log1p(list(range(1001)))


# ---------------------------------------------------------------------------
# Unit tests — predict_next (model selection)
# ---------------------------------------------------------------------------

def test_predict_next_linear_series():
    # Perfect arithmetic sequence — linear should win with near-zero holdout error
    result = predict_next([2, 4, 6, 8, 10])
    assert result['method'] == 'linear'
    assert abs(result['prediction'] - 12.0) < 0.01
    assert 'holdout_errors' in result
    assert len(result['holdout_errors']) >= 3
    ci = result['confidence_interval']
    assert ci['lower'] - 1e-9 <= result['prediction'] <= ci['upper'] + 1e-9
    assert ci['level'] == 0.95

def test_predict_next_stationary_series():
    # Constant series — all models tie at 0 error; prediction must be ~5
    result = predict_next([5, 5, 5, 5, 5, 5])
    assert abs(result['prediction'] - 5.0) < 0.01
    assert 'holdout_errors' in result

def test_predict_next_has_slope_when_linear():
    result = predict_next([1, 2, 3, 4, 5])
    if result['method'] in ('linear', 'log1p-linear'):
        assert 'slope' in result
        assert 'intercept' in result

def test_predict_next_negative_values_skip_log1p():
    # Negative values: log1p-linear must be skipped
    result = predict_next([-5, -3, -1, 1, 3])
    assert result['method'] != 'log1p-linear'

def test_predict_next_too_short():
    with pytest.raises(ValueError):
        predict_next([1, 2])

def test_predict_next_too_long():
    with pytest.raises(ValueError):
        predict_next(list(range(1001)))


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
    assert data['x402Version'] == 2
    assert data['openapi'].endswith('/openapi.json')
    resource_entry = data['resources'][0]
    assert resource_entry['method'] == 'POST'
    assert resource_entry['resource']['url'].endswith('/timeseries')
    assert resource_entry['accepts'][0]['scheme'] == 'exact'
    assert resource_entry['accepts'][0]['amount'] == '10000'

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
    assert body['x402Version'] == 2
    assert body['resource']['url'].endswith('/timeseries')
    assert body['accepts'][0]['scheme'] == 'exact'
    assert body['accepts'][0]['network'] == 'eip155:8453'
    assert body['accepts'][0]['amount'] == '10000'
    assert body['accepts'][0]['payTo'].startswith('0x')
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
    assert 'method' in data
    assert data['method'] in ('linear', 'log1p-linear', 'last-delta', 'mean')
    assert 'holdout_errors' in data
    assert 'confidence_interval' in data
    ci = data['confidence_interval']
    assert ci['lower'] - 1e-9 <= data['prediction'] <= ci['upper'] + 1e-9
    assert ci['level'] == 0.95

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


# ---------------------------------------------------------------------------
# Anomaly detection — unit tests
# ---------------------------------------------------------------------------

def test_detect_anomalies_valid():
    # n=6 needed: with sample std (ddof=1), max z-score for n=5 is ~1.79 < default threshold 2.0
    result = detect_anomalies([1, 1, 1, 1, 1, 100])
    assert isinstance(result['anomalies'], list)
    assert result['method'] == 'z-score'
    assert len(result['anomalies']) > 0
    assert result['anomalies'][0]['index'] == 5
    nr = result['normal_range']
    assert nr['lower'] <= nr['upper']
    assert nr['lower'] == round(result['mean'] - result['threshold'] * result['std'], 6)
    assert nr['upper'] == round(result['mean'] + result['threshold'] * result['std'], 6)

def test_detect_anomalies_no_anomalies():
    result = detect_anomalies([1, 2, 3, 4, 5])
    assert result['anomalies'] == []

def test_detect_anomalies_custom_threshold():
    result = detect_anomalies([1, 2, 3, 4, 5, 100], threshold=1.0)
    assert len(result['anomalies']) > 0

def test_detect_anomalies_too_short():
    with pytest.raises(ValueError):
        detect_anomalies([1, 2])

def test_detect_anomalies_bad_threshold():
    with pytest.raises(ValueError):
        detect_anomalies([1, 2, 3], threshold=0)


# ---------------------------------------------------------------------------
# Anomaly detection — route tests
# ---------------------------------------------------------------------------

def test_anomaly_unpaid_returns_402(client):
    resp = client.post('/anomaly-detection')
    assert resp.status_code == 402
    body = resp.get_json()
    assert body['x402Version'] == 2
    assert body['resource']['url'].endswith('/anomaly-detection')
    assert resp.headers.get('PAYMENT-REQUIRED')

def test_anomaly_payment_gate_before_body(client):
    resp = client.post('/anomaly-detection', content_type='application/json', data='')
    assert resp.status_code == 402

def test_anomaly_paid_array_body(client):
    resp = client.post('/anomaly-detection',
                       json=[1, 1, 1, 1, 100],
                       headers={'X-PAYMENT': 'signed-payload'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'anomalies' in data
    assert data['method'] == 'z-score'
    assert 'mean' in data
    assert 'std' in data
    assert 'threshold' in data
    assert 'normal_range' in data
    nr = data['normal_range']
    assert nr['lower'] <= nr['upper']

def test_anomaly_paid_object_body(client):
    resp = client.post('/anomaly-detection',
                       json={'series': [1, 1, 1, 1, 100], 'threshold': 1.5},
                       headers={'X-PAYMENT': 'signed-payload'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['threshold'] == 1.5

def test_anomaly_paid_with_context(client):
    resp = client.post('/anomaly-detection',
                       json={'series': [1, 2, 3, 4, 5], 'context': 'cpu usage %'},
                       headers={'X-PAYMENT': 'signed-payload'})
    assert resp.status_code == 200
    assert resp.get_json()['context'] == 'cpu usage %'

def test_anomaly_paid_bad_series_length(client):
    resp = client.post('/anomaly-detection',
                       json=[1, 2],
                       headers={'X-PAYMENT': 'signed-payload'})
    assert resp.status_code == 400

def test_anomaly_paid_bad_threshold(client):
    resp = client.post('/anomaly-detection',
                       json={'series': [1, 2, 3, 4, 5], 'threshold': 0},
                       headers={'X-PAYMENT': 'signed-payload'})
    assert resp.status_code == 400

def test_anomaly_get(client):
    resp = client.get('/anomaly-detection')
    assert resp.status_code == 200
