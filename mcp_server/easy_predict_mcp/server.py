import os
import json
import time
import base64
import secrets
import requests
from mcp.server.fastmcp import FastMCP

API_BASE    = os.environ.get("EASY_PREDICT_URL", "https://easy-predict.com").rstrip("/")
PRIVATE_KEY = os.environ.get("WALLET_PRIVATE_KEY")

mcp = FastMCP(
    "easy-predict",
    instructions=(
        "Use predict_timeseries to forecast the next value in a numeric series, "
        "or detect_anomalies to find outliers. "
        "Each call costs $0.01 USDC on Base, paid automatically from your local wallet via x402. "
        "Your private key never leaves your machine."
    ),
)


def _call_with_payment(url: str, body: dict) -> dict:
    """
    POST to an x402-gated endpoint.
    Signs the payment locally — the private key never leaves this process.
    Only the signed payment header is sent to the remote server.
    """
    if not PRIVATE_KEY:
        raise ValueError(
            "WALLET_PRIVATE_KEY is not set. "
            "Add it to your MCP client config under 'env': {\"WALLET_PRIVATE_KEY\": \"0x...\"}"
        )

    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError:
        raise RuntimeError(
            "eth-account is not installed. "
            "Reinstall the package: pip install easy-predict-mcp"
        )

    resp = requests.post(url, json=body)
    if resp.status_code == 200:
        return resp.json()
    if resp.status_code != 402:
        resp.raise_for_status()

    challenge = resp.json()
    reqs      = challenge["accepts"][0]
    account   = Account.from_key(PRIVATE_KEY)

    payload = {
        "x402Version": 2,
        "scheme":      reqs["scheme"],
        "network":     reqs["network"],
        "asset":       reqs["asset"],
        "amount":      reqs["amount"],
        "payTo":       reqs["payTo"],
        "resource":    url,
        "from":        account.address,
        "nonce":       secrets.token_hex(16),
        "validUntil":  int(time.time()) + reqs.get("maxTimeoutSeconds", 60),
    }

    # Sign locally — only the resulting signature is sent over the network
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    sig       = account.sign_message(encode_defunct(text=canonical))
    signed    = {"payload": payload, "signature": sig.signature.hex(), "from": account.address}
    header    = base64.b64encode(json.dumps(signed).encode()).decode()

    resp = requests.post(url, json=body, headers={"PAYMENT-SIGNATURE": header})
    if not resp.ok:
        raise ValueError(f"API error after payment: {resp.status_code} — {resp.text[:200]}")
    return resp.json()


@mcp.tool()
def predict_timeseries(series: list[float], context: str = "") -> str:
    """
    Predict the next value in a numeric time series.

    Calls easy-predict.com and pays $0.01 USDC on Base automatically via x402.
    Your private key signs the payment locally and is never sent to any server.
    Uses automatic model selection: linear, log1p-linear, last-delta, or mean.

    Args:
        series:  List of 3–1000 numbers, ordered oldest to newest.
        context: What the series represents, e.g. "monthly revenue USD". Max 200 chars.
    """
    body: dict = {"series": series}
    if context:
        body["context"] = context[:200]
    return json.dumps(_call_with_payment(f"{API_BASE}/timeseries", body))


@mcp.tool()
def detect_anomalies(series: list[float], threshold: float = 2.0, context: str = "") -> str:
    """
    Detect anomalies in a numeric time series using z-score analysis.

    Calls easy-predict.com and pays $0.01 USDC on Base automatically via x402.
    Your private key signs the payment locally and is never sent to any server.

    Args:
        series:    List of 3–1000 numbers to analyze.
        threshold: Z-score cutoff (default 2.0, range 0–10). Lower = more sensitive.
        context:   What the series represents. Max 200 chars.
    """
    body: dict = {"series": series, "threshold": threshold}
    if context:
        body["context"] = context[:200]
    return json.dumps(_call_with_payment(f"{API_BASE}/anomaly-detection", body))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
