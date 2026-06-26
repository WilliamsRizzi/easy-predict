#!/usr/bin/env python3
"""
Demo: AI agent autonomously calling easy-predict.com with x402 micropayments.

Shows how a Claude agent can:
  1. Receive a forecasting / anomaly-detection task in natural language
  2. Call easy-predict.com, hitting a 402 Payment Required gate
  3. Parse the x402 v2 payment challenge from the response
  4. Sign and submit a $0.01 USDC micropayment on Base
  5. Retry with the payment header and return the result

Two modes
---------
  DEMO_MODE (default)
    Runs against a local Flask server. Payments are mocked — the header is
    structurally valid but unsigned. Start the local server first:

        cd timeseries && python app.py   # listens on http://localhost:8000

  LIVE MODE
    Calls the real easy-predict.com API and pays real USDC on Base.
    Requires eth-account (pip install eth-account) and a funded wallet.

Environment variables
---------------------
  ANTHROPIC_API_KEY   Required — your Anthropic API key.
  WALLET_PRIVATE_KEY  Hex private key of a Base wallet holding USDC on Base.
                      When set, LIVE MODE is enabled automatically.
  EASY_PREDICT_URL    Override the API base URL.
                      Default DEMO_MODE: http://localhost:8000
                      Default LIVE MODE: https://easy-predict.com

Dependencies
------------
  pip install anthropic requests          # always required
  pip install eth-account                 # required only for LIVE MODE
"""

import os
import json
import time
import base64
import secrets
import requests
import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────

_PRIVATE_KEY = os.environ.get("WALLET_PRIVATE_KEY")
LIVE_MODE = bool(_PRIVATE_KEY)

_default_url = "https://easy-predict.com" if LIVE_MODE else "http://localhost:8000"
API_BASE = os.environ.get("EASY_PREDICT_URL", _default_url).rstrip("/")

FACILITATOR_URL = "https://x402.org/facilitator"


# ── x402 payment helpers ────────────────────────────────────────────────────────

def _build_payment_payload(challenge: dict, resource_url: str) -> dict:
    """Build a signed x402 v2 payment payload using eth_account."""
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError:
        raise RuntimeError(
            "Live mode requires eth-account: pip install eth-account\n"
            "Or omit WALLET_PRIVATE_KEY to run in demo mode."
        )

    reqs = challenge["accepts"][0]
    account = Account.from_key(_PRIVATE_KEY)

    payload = {
        "x402Version": 2,
        "scheme":      reqs["scheme"],
        "network":     reqs["network"],
        "asset":       reqs["asset"],
        "amount":      reqs["amount"],
        "payTo":       reqs["payTo"],
        "resource":    resource_url,
        "from":        account.address,
        "nonce":       secrets.token_hex(16),
        "validUntil":  int(time.time()) + reqs.get("maxTimeoutSeconds", 60),
    }

    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    sig = account.sign_message(encode_defunct(text=canonical))

    return {
        "payload":   payload,
        "signature": sig.signature.hex(),
        "from":      account.address,
    }


def _mock_payment_header(challenge: dict, resource_url: str) -> str:
    """
    Return a structurally valid but unsigned payment header.

    The local Flask server uses presence-only checking (_has_payment()), so any
    non-empty PAYMENT-SIGNATURE value unlocks the endpoint. Do not use against
    the production Cloudflare Worker — it verifies via the x402 facilitator.
    """
    reqs = challenge["accepts"][0]
    mock = {
        "payload": {
            "x402Version": 2,
            "scheme":      reqs["scheme"],
            "network":     reqs["network"],
            "asset":       reqs["asset"],
            "amount":      reqs["amount"],
            "payTo":       reqs["payTo"],
            "resource":    resource_url,
            "from":        "0xDEMO000000000000000000000000000000000000",
            "nonce":       secrets.token_hex(16),
            "validUntil":  int(time.time()) + 60,
        },
        "signature": "0x" + "00" * 65,
        "from":      "0xDEMO000000000000000000000000000000000000",
    }
    return base64.b64encode(json.dumps(mock).encode()).decode()


def _call_with_payment(url: str, body: dict) -> dict:
    """
    POST to an x402-gated endpoint. Handles the 402 → sign → retry flow.

    Returns the parsed JSON on success; raises on persistent failure.
    """
    resp = requests.post(url, json=body)

    if resp.status_code == 200:
        return resp.json()

    if resp.status_code != 402:
        resp.raise_for_status()

    # Parse the payment challenge
    challenge = resp.json()
    amount_units = int(challenge["accepts"][0]["amount"])
    cost_usd = amount_units / 1_000_000
    print(f"    [x402] 402 received — cost ${cost_usd:.4f} USDC on Base")

    if LIVE_MODE:
        signed = _build_payment_payload(challenge, url)
        payment_header = base64.b64encode(json.dumps(signed).encode()).decode()
        print(f"    [x402] Signed by {signed['from'][:10]}…")
    else:
        payment_header = _mock_payment_header(challenge, url)
        print("    [x402] DEMO MODE — mock payment header (local server only)")

    resp = requests.post(
        url,
        json=body,
        headers={"PAYMENT-SIGNATURE": payment_header},
    )

    if not resp.ok:
        raise ValueError(
            f"API error after payment: {resp.status_code} — {resp.text[:300]}"
        )

    return resp.json()


# ── API wrappers (called by Claude's tools) ─────────────────────────────────────

def predict_timeseries(series: list, context: str = "") -> dict:
    body: dict = {"series": series}
    if context:
        body["context"] = context[:200]
    return _call_with_payment(f"{API_BASE}/timeseries", body)


def detect_anomalies(series: list, threshold: float = 2.0, context: str = "") -> dict:
    body: dict = {"series": series, "threshold": threshold}
    if context:
        body["context"] = context[:200]
    return _call_with_payment(f"{API_BASE}/anomaly-detection", body)


# ── Claude tool definitions ─────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "predict_timeseries",
        "description": (
            "Predict the next value in a numeric time series. "
            "Calls easy-predict.com and automatically pays $0.01 USDC on Base "
            "via x402 micropayment when required. "
            "Use when the user provides historical numbers and wants a forecast."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Historical values, 3–1000 items, oldest first.",
                },
                "context": {
                    "type": "string",
                    "description": (
                        "What the series represents, e.g. 'monthly revenue USD'. "
                        "Max 200 chars. Optional."
                    ),
                },
            },
            "required": ["series"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Detect anomalies in a numeric series using z-score analysis. "
            "Calls easy-predict.com and automatically pays $0.01 USDC on Base "
            "via x402 micropayment when required. "
            "Use when the user wants to find outliers or suspicious data points."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Values to analyze, 3–1000 items.",
                },
                "threshold": {
                    "type": "number",
                    "description": "Z-score cutoff (default 2.0). Higher = fewer anomalies flagged.",
                },
                "context": {
                    "type": "string",
                    "description": "What the series represents. Max 200 chars. Optional.",
                },
            },
            "required": ["series"],
        },
    },
]

_TOOL_FNS = {
    "predict_timeseries": predict_timeseries,
    "detect_anomalies":   detect_anomalies,
}

_client = anthropic.Anthropic()


# ── Agent loop ──────────────────────────────────────────────────────────────────

def run_agent(task: str) -> str:
    """Run the prediction agent on a natural-language task; return final answer."""
    print(f"\n{'─' * 60}")
    print(f"Task: {task}")
    print('─' * 60)

    messages = [{"role": "user", "content": task}]

    while True:
        response = _client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Stream any text blocks to stdout immediately
        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"\nAgent: {block.text}")

        if response.stop_reason == "end_turn":
            return next(
                (b.text for b in response.content if b.type == "text"), ""
            )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                fn = _TOOL_FNS[block.name]
                args_preview = json.dumps(block.input)[:120]
                print(f"\n  → {block.name}({args_preview})")

                try:
                    result = fn(**block.input)
                    result_str = json.dumps(result)
                    print(f"  ← {result_str[:200]}")
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_str,
                    })
                except Exception as exc:
                    print(f"  ! error: {exc}")
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     f"Error: {exc}",
                        "is_error":    True,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",      "content": tool_results})
            continue

        break  # unexpected stop_reason

    return ""


# ── Demo entry point ────────────────────────────────────────────────────────────

DEMO_TASKS = [
    (
        "My monthly active users for the past 8 months: "
        "1200, 1350, 1480, 1620, 1780, 1950, 2100, 2280. "
        "Forecast next month."
    ),
    (
        "Check these server response times (ms) for anomalies: "
        "45, 48, 52, 47, 46, 51, 250, 49, 47, 48, 46, 310, 50"
    ),
    (
        "I have quarterly revenue ($M): 2.1, 2.4, 2.8, 3.2, 3.7, 4.1. "
        "Detect any anomalies and then forecast Q7."
    ),
]

if __name__ == "__main__":
    mode = "LIVE (real x402 payments on Base)" if LIVE_MODE else "DEMO (mock payments — local server)"
    print(f"easy-predict agent demo")
    print(f"Mode: {mode}")
    print(f"API:  {API_BASE}")

    if not LIVE_MODE:
        print(
            "\nRunning in DEMO MODE against local Flask server.\n"
            "Start it first:  cd timeseries && python app.py\n"
            "Set WALLET_PRIVATE_KEY to switch to live mode with real payments.\n"
        )

    for task in DEMO_TASKS:
        run_agent(task)

    print("\nDemo complete.")
