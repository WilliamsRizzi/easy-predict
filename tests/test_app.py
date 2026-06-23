import importlib
import pytest
from timeseries.app import app, predict_next_log1p
app_module = importlib.import_module('timeseries.app')

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_predict_next_log1p_valid():
    series = [1, 2, 3]
    pred, slope, intercept = predict_next_log1p(series)
    assert isinstance(pred, float)
    assert pred >= 0

def test_predict_next_log1p_length_errors():
    with pytest.raises(ValueError):
        predict_next_log1p([1, 2])
    with pytest.raises(ValueError):
        predict_next_log1p(list(range(20)))

def test_log1p_endpoint_array(client):
    headers = {'X-402': 'testtoken', 'X-402-Cost': '0.001', 'X-Agent-Type': 'ai'}
    resp = client.post('/timeseries/log1p', json=[1, 2, 3, 4], headers=headers)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'prediction' in data

def test_log1p_endpoint_object(client):
    headers = {'X-402': 'testtoken', 'X-402-Cost': '0.001', 'X-Agent-Type': 'ai'}
    resp = client.post('/timeseries/log1p', json={'series': [1, 1, 2]}, headers=headers)
    assert resp.status_code == 200

def test_log1p_endpoint_facilitator_header(client):
    headers = {'X-402-Facilitator': 'testtoken', 'X-402-Cost': '0.001', 'X-Agent-Type': 'ai'}
    resp = client.post('/timeseries/log1p', json=[1, 2, 3], headers=headers)
    assert resp.status_code == 200


def test_log1p_endpoint_payment_required_missing_headers(client):
    resp = client.post('/timeseries/log1p', json=[1, 2, 3])
    assert resp.status_code == 403


def test_log1p_endpoint_payment_required_wrong_cost(client):
    headers = {'X-402': 'testtoken', 'X-402-Cost': '0.01', 'X-Agent-Type': 'ai'}
    resp = client.post('/timeseries/log1p', json=[1, 2, 3], headers=headers)
    assert resp.status_code == 402


def test_log1p_endpoint_forbidden_wrong_agent(client):
    headers = {'X-402': 'testtoken', 'X-402-Cost': '0.001', 'X-Agent-Type': 'bot'}
    resp = client.post('/timeseries/log1p', json=[1,2,3], headers=headers)
    assert resp.status_code == 403


def test_openapi_discovery(client):
    resp = client.get('/openapi.json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['openapi'].startswith('3')
    assert data['paths']['/timeseries/log1p']['post']['x-payment-info']['protocols'][0] == {'x402': {}}
    assert 'x-guidance' in data['info']


def test_log1p_endpoint_bad_length(client):
    headers = {'X-402': 'testtoken', 'X-402-Cost': '0.001', 'X-Agent-Type': 'ai'}
    resp = client.post('/timeseries/log1p', json=[1, 2], headers=headers)
    assert resp.status_code == 400

def test_log1p_endpoint_no_json(client):
    headers = {'X-402': 'testtoken', 'X-402-Cost': '0.001', 'X-Agent-Type': 'ai'}
    resp = client.post('/timeseries/log1p', data='notjson', content_type='text/plain', headers=headers)
    assert resp.status_code == 400

def test_log1p_missing_headers(client):
    resp = client.post('/timeseries/log1p', json=[1,2,3])
    assert resp.status_code == 403


def test_log1p_wrong_cost_or_agent(client):
    headers = {'X-402': 'testtoken', 'X-402-Cost': '0.01', 'X-Agent-Type': 'bot'}
    resp = client.post('/timeseries/log1p', json=[1,2,3], headers=headers)
    assert resp.status_code == 403


def test_log1p_rate_limit(client):
    original_limit = app_module.RATE_LIMIT_MAX_REQUESTS
    app_module.RATE_LIMIT_MAX_REQUESTS = 2
    app_module.rate_limit_records.clear()
    headers = {'X-402': 'testtoken', 'X-402-Cost': '0.001', 'X-Agent-Type': 'ai'}
    try:
        for _ in range(2):
            resp = client.post('/timeseries/log1p', json=[1, 2, 3], headers=headers, environ_base={'REMOTE_ADDR': '127.0.0.1'})
            assert resp.status_code == 200
        resp = client.post('/timeseries/log1p', json=[1, 2, 3], headers=headers, environ_base={'REMOTE_ADDR': '127.0.0.1'})
        assert resp.status_code == 429
        data = resp.get_json()
        assert data['error'] == 'Too many requests, rate limit exceeded'
    finally:
        app_module.RATE_LIMIT_MAX_REQUESTS = original_limit
        app_module.rate_limit_records.clear()


def test_log1p_endpoint_get(client):
    resp = client.get('/timeseries/log1p')
    assert resp.status_code == 200
    assert b'Timeseries Log1p Predictor' in resp.data
