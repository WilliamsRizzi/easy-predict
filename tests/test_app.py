import pytest
from timeseries import predict_next_log1p, app

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
