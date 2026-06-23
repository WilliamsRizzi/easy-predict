# easy-predict

`easy-predict` is a lightweight Flask-based service for time series prediction. It exposes a simple endpoint to extrapolate the next value from a numeric series using a log1p linear regression model.

## What this repo contains

- `timeseries/app.py` — the Flask application implementing `/log1p` and `/timeseries/log1p`
- `tests/test_app.py` — unit tests for prediction behavior, request validation, and rate limiting
- `requirements.txt` / `pyproject.toml` — project dependencies and Python packaging metadata

## Try it live

Visit **easy-predict.com** to see the project in action and test the prediction endpoint.

## Local usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python timeseries/app.py
   ```
3. Send a POST request to:
   - `http://localhost:8000/timeseries/log1p`
   - `http://localhost:8000/log1p`

Required headers for POST requests:

- `X-Agent-Type: ai`
- `X-402-Cost: 0.001`
- `X-PAYMENT: <x402 payload>` — a signed x402 payment, **or** `X-402: <token>` / `X-402-Facilitator: <token>` when `X402_TOKEN` is configured

## x402 discovery

The paid prediction endpoints are monetized with [x402](https://www.x402.org/).
The service publishes two discovery documents so indexers (e.g. x402scan.com)
can find and price the endpoints automatically:

- `GET /openapi.json` — OpenAPI 3.1 document. Paid operations declare the
  `x402` security scheme and an `x-payment-info` block; free operations declare
  `"security": []`.
- `GET /.well-known/x402` — machine-readable list of paid resources with their
  x402 `PaymentRequirements`, plus a pointer to `/openapi.json`.

Calling a paid endpoint without payment returns **HTTP 402** whose JSON body
follows the x402 wire format:

```json
{
  "x402Version": 1,
  "error": "Payment Required: missing or invalid x402 payment",
  "accepts": [
    {
      "scheme": "exact",
      "network": "base",
      "maxAmountRequired": "1000",
      "resource": "https://easy-predict.com/timeseries/log1p",
      "payTo": "0xc99b83818c8865340AC55C45554f377f41c68DBC",
      "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
      "maxTimeoutSeconds": 60,
      "extra": { "name": "USD Coin", "version": "2" }
    }
  ],
  "facilitator": "https://x402.org/facilitator"
}
```

The same payload is echoed (base64) in the `X-Payment-Required` /
`PAYMENT-REQUIRED` response headers.

### Configuration

Payment terms default to Base mainnet USDC at $0.001 and are overridable via
environment variables: `PAY_TO_ADDRESS`, `X402_NETWORK`, `X402_ASSET`,
`X402_ASSET_NAME`, `X402_PRICE_USD`, `X402_MAX_AMOUNT_REQUIRED`,
`X402_MAX_TIMEOUT_SECONDS`, and `X402_FACILITATOR_URL`.
