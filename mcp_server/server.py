#!/usr/bin/env python3
"""
easy-predict MCP server

Exposes two tools to any MCP-compatible client (Claude Desktop, Cursor, Windsurf, …):
  predict_timeseries  — forecast the next value in a numeric series
  detect_anomalies    — find anomalous points via z-score

Each tool call costs $0.01 USDC on Base, paid automatically via x402 v2.

Environment variables
---------------------
  WALLET_PRIVATE_KEY   Required. Hex private key of a Base wallet holding USDC.
  EASY_PREDICT_URL     Override API base URL (default: https://easy-predict.com).

Usage
-----
  pip install mcp[cli] requests eth-account
  WALLET_PRIVATE_KEY=0x... python mcp_server/server.py
"""

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
        "Each call costs $0.01 USDC on Base, paid automatically — "
        "no API key or account needed beyond a funded wallet."
    ),
)


def _call_with_payment(url: str, body: dict) -> dict:
    """POST to an x402-gated endpoint. Handles the 402 → sign → retry flow."""
    if not PRIVATE_KEY:
        raise ValueError(
            "WALLET_PRIVATE_KEY is not set. "
            "Configure a Base wallet private key holding USDC to use easy-predict tools."
        )

    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError:
        raise RuntimeError(
            "eth-account is required: pip install eth-account"
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

    Sends the series to easy-predict.com and automatically pays $0.01 USDC on
    Base via x402. Uses automatic model selection (linear, log1p-linear,
    last-delta, mean) — picks the model with the lowest holdout error.

    Args:
        series:  List of 3–1000 numbers, ordered oldest to newest.
        context: What the series represents, e.g. "monthly revenue USD". Max 200 chars.

    Returns:
        JSON with prediction, method, holdout_errors, and optional slope/intercept.
    """
    body: dict = {"series": series}
    if context:
        body["context"] = context[:200]
    return json.dumps(_call_with_payment(f"{API_BASE}/timeseries", body))


@mcp.tool()
def detect_anomalies(series: list[float], threshold: float = 2.0, context: str = "") -> str:
    """
    Detect anomalies in a numeric time series using z-score analysis.

    Sends the series to easy-predict.com and automatically pays $0.01 USDC on
    Base via x402. Each anomaly in the response includes its index, value, and z-score.

    Args:
        series:    List of 3–1000 numbers to analyze.
        threshold: Z-score cutoff (default 2.0, range 0–10).
                   Lower = more sensitive (flags more points).
        context:   What the series represents. Max 200 chars.

    Returns:
        JSON with anomalies list, method, mean, std, and threshold.
    """
    body: dict = {"series": series, "threshold": threshold}
    if context:
        body["context"] = context[:200]
    return json.dumps(_call_with_payment(f"{API_BASE}/anomaly-detection", body))


if __name__ == "__main__":
    mcp.run()
