# easy-predict

Agent-first prediction and anomaly detection API. Send a list of numbers, get the next predicted value or a list of anomalous points. Paid per call via [x402](https://x402.org) v2 micropayments — $0.01 USDC on Base. No API keys, no accounts.

**Live at [easy-predict.com](https://easy-predict.com)**

---

## Endpoints

| Method | Path | Cost | Description |
|--------|------|------|-------------|
| POST | `/timeseries` | $0.01 USDC | Predict the next value in a numeric series |
| POST | `/anomaly-detection` | $0.01 USDC | Detect anomalies via z-score |
| GET | `/.well-known/x402` | free | x402 v2 resource discovery |
| GET | `/openapi.json` | free | OpenAPI 3.1 spec |
| GET | `/llms.txt` | free | Human/agent-readable API docs |

### POST /timeseries

```bash
curl https://easy-predict.com/timeseries \
  -H "Content-Type: application/json" \
  -H "PAYMENT-SIGNATURE: <signed-x402-payload>" \
  -d '{"series": [1.0, 2.3, 4.1, 6.8, 9.2], "context": "monthly revenue USD"}'
```

Response:

```json
{
  "prediction": 12.1,
  "method": "linear",
  "holdout_errors": {"linear": 0.05, "log1p-linear": 0.31, "last-delta": 0.05, "mean": 3.5},
  "slope": 1.94,
  "intercept": 0.12
}
```

Model selection: holds out the last point, evaluates `linear`, `log1p-linear`, `last-delta`, and `mean` against it, retrains the winner on the full series.

### POST /anomaly-detection

```bash
curl https://easy-predict.com/anomaly-detection \
  -H "Content-Type: application/json" \
  -H "PAYMENT-SIGNATURE: <signed-x402-payload>" \
  -d '{"series": [1.0, 2.3, 4.1, 6.8, 99.0], "threshold": 2.0}'
```

Response:

```json
{
  "anomalies": [{"index": 4, "value": 99.0, "z_score": 2.14}],
  "method": "z-score",
  "mean": 15.12,
  "std": 39.18,
  "threshold": 2.0
}
```

Both endpoints accept an optional `"context"` string (max 200 chars) echoed back in the response.

---

## x402 payment flow

Omit the payment header to get a 402 with the full payment terms:

```json
{
  "x402Version": 2,
  "error": "Payment Required",
  "accepts": [{
    "scheme": "exact",
    "network": "eip155:8453",
    "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "amount": "10000",
    "payTo": "0xc99b83818c8865340AC55C45554f377f41c68DBC",
    "maxTimeoutSeconds": 60
  }]
}
```

1. Parse `accepts[0]` from the 402 response
2. Sign a payment payload with your wallet
3. Base64-encode the signed JSON and retry with `PAYMENT-SIGNATURE: <encoded>`
4. The Cloudflare Worker verifies via `https://x402.org/facilitator` and settles on-chain

Payment: 10000 atomic units = $0.01 USDC (6 decimals) on Base mainnet.

---

## MCP server (Claude Desktop, Cursor, Windsurf)

The MCP server runs locally on your machine. Your wallet private key never leaves your device — it signs the x402 payment locally and only the signed header is sent to easy-predict.com.

Add to your MCP client config:

```json
{
  "mcpServers": {
    "easy-predict": {
      "command": "uvx",
      "args": ["easy-predict-mcp"],
      "env": {
        "WALLET_PRIVATE_KEY": "0xYOUR_WALLET_PRIVATE_KEY"
      }
    }
  }
}
```

Then ask Claude: *"Predict the next value for this series: 1.2, 2.4, 4.1, 6.8"* — it calls the tool, pays $0.01 USDC from your wallet automatically, and returns the forecast.

Listed on [Smithery](https://smithery.ai) — search `easy-predict` to install with one click.

Listed on [Smithery](https://smithery.ai) — search `easy-predict` to install with one click.

---

## Agent integration (Python)

A working Claude agent that handles the full 402 → sign → retry loop autonomously:

```bash
pip install anthropic requests eth-account
ANTHROPIC_API_KEY=sk-... WALLET_PRIVATE_KEY=0x... python examples/demo_agent.py
```

[`examples/demo_agent.py`](examples/demo_agent.py) — uses Anthropic tool use. The same pattern works with LangChain, LlamaIndex, CrewAI, or AutoGen.

Demo mode (no wallet, against local server):

```bash
cd timeseries && python app.py &
ANTHROPIC_API_KEY=sk-... python examples/demo_agent.py
```

---

## Architecture

- **Cloudflare Worker** (`src/index.ts`) — edge runtime, rate limiting, full x402 verify+settle via the facilitator
- **Flask backend** (`timeseries/app.py`) — Python prediction logic, local dev server
- **Static assets** (`public/`) — splash page, `openapi.json`, `llms.txt`

The Worker handles all production traffic. The Flask app is used for local development and runs on port 8000.

---

## Local development

```bash
pip install -r requirements.txt
cd timeseries && python app.py        # Flask on http://localhost:8000
```

The local Flask server uses presence-only payment checking — any non-empty `PAYMENT-SIGNATURE` header is accepted. Useful for testing without a real wallet.

For the Cloudflare Worker:

```bash
npm install
npx wrangler dev                      # Worker on http://localhost:8787
```

---

## Repo structure

```
timeseries/app.py                        Flask backend (prediction + anomaly detection logic)
anomaly_detection/app.py                 Anomaly detection blueprint
src/index.ts                             Cloudflare Worker (edge runtime)
mcp_server/easy_predict_mcp/server.py   MCP server — runs locally, signs payments on device
mcp_server/pyproject.toml               PyPI package definition (easy-predict-mcp)
smithery.yaml                            Smithery registry config
public/
  index.html                             Splash page
  openapi.json                           OpenAPI 3.1 spec
  llms.txt                               Human/agent-readable docs
examples/
  demo_agent.py                          Claude agent integration demo with x402 payments
tests/                                   Test suite
```

---

## Configuration

Flask backend env vars (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `PAY_TO_ADDRESS` | `0xc99b...` | Recipient wallet |
| `X402_ASSET` | USDC on Base | ERC-20 asset address |
| `X402_NETWORK` | `eip155:8453` | Chain ID |
| `X402_MAX_AMOUNT` | `10000` | Atomic units ($0.01) |
| `FACILITATOR_URL` | `https://x402.org/facilitator` | x402 facilitator |
| `PUBLIC_BASE_URL` | `https://easy-predict.com` | Base URL for resource URLs in 402 responses |
