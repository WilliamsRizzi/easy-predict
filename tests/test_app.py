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
    # No payment of any kind -> 402 payment challenge (not 403). The payment
    # gate must precede the identity gate so discovery crawlers get a 402.
    resp = client.post('/timeseries/log1p', json=[1, 2, 3])
    assert resp.status_code == 402


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
    assert resp.status_code == 402


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


def test_402_returns_x402_payment_requirements(client):
    # Correct identity and cost but no payment -> 402 with x402 metadata
    headers = {'X-402-Cost': '0.001', 'X-Agent-Type': 'ai'}
    # Ensure the shared-token path is exercised by clearing the env token
    import os
    original = os.environ.pop('X402_TOKEN', None)
    try:
        resp = client.post('/timeseries/log1p', json=[1, 2, 3], headers=headers)
    finally:
        if original is not None:
            os.environ['X402_TOKEN'] = original
    assert resp.status_code == 402
    body = resp.get_json()
    assert body['x402Version'] == 1
    assert isinstance(body['accepts'], list) and body['accepts']
    req = body['accepts'][0]
    assert req['scheme'] == 'exact'
    assert req['network'] == 'base'
    assert req['payTo'].startswith('0x')
    assert req['maxAmountRequired'] == '1000'
    assert req['asset'].startswith('0x')
    # x402 v2 header echo present
    assert resp.headers.get('X-Payment-Required')
    assert resp.headers.get('PAYMENT-REQUIRED')


@pytest.mark.parametrize('path', ['/log1p', '/timeseries/log1p'])
def test_unpaid_probe_returns_402_challenge(client, path):
    # Exactly what an x402 discovery crawler (x402scan) does: a bare POST with
    # no identity, cost, or payment headers and no body. It must get a 402
    # payment challenge -- never a 403 or a body-validation error -- so the
    # endpoint is registered as a paid x402 resource.
    import os
    original = os.environ.pop('X402_TOKEN', None)
    try:
        resp = client.post(path)
    finally:
        if original is not None:
            os.environ['X402_TOKEN'] = original
    assert resp.status_code == 402
    body = resp.get_json()
    assert body['accepts'][0]['scheme'] == 'exact'
    assert resp.headers.get('X-Payment-Required')


def test_x402_payment_header_satisfies_paywall(client):
    headers = {'X-PAYMENT': 'deadbeef', 'X-402-Cost': '0.001', 'X-Agent-Type': 'ai'}
    resp = client.post('/timeseries/log1p', json=[1, 2, 3], headers=headers)
    assert resp.status_code == 200


def test_openapi_paid_and_free_security(client):
    resp = client.get('/openapi.json')
    data = resp.get_json()
    # Paid POST declares the x402 security scheme
    assert data['paths']['/timeseries/log1p']['post']['security'] == [{'x402': []}]
    # Free GET declares empty security
    assert data['paths']['/timeseries/log1p']['get']['security'] == []
    assert data['paths']['/openapi.json']['get']['security'] == []
    # Security scheme is defined
    assert data['components']['securitySchemes']['x402']['in'] == 'header'
    # Enriched x402 payment requirements present on the paid operation
    accepts = data['paths']['/timeseries/log1p']['post']['x-payment-info']['x402']['accepts']
    assert accepts[0]['payTo'].startswith('0x')


def test_well_known_x402_discovery(client):
    resp = client.get('/.well-known/x402')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['x402Version'] == 1
    assert data['openapi'].endswith('/openapi.json')
    paths = {r['resource'].rsplit('/', 1)[-1] or r['resource'] for r in data['resources']}
    assert any(r['method'] == 'POST' for r in data['resources'])
    assert data['resources'][0]['accepts'][0]['scheme'] == 'exact'
