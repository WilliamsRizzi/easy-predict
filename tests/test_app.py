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
    resp = client.post('/timeseries/log1p', json=[1, 2, 3, 4])
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'prediction' in data

def test_log1p_endpoint_object(client):
    resp = client.post('/timeseries/log1p', json={'series': [1, 1, 2]})
    assert resp.status_code == 200

def test_log1p_endpoint_bad_length(client):
    resp = client.post('/timeseries/log1p', json=[1, 2])
    assert resp.status_code == 400

def test_log1p_endpoint_no_json(client):
    resp = client.post('/timeseries/log1p', data='notjson', content_type='text/plain')
    assert resp.status_code == 400
