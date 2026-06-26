import * as secp from '@noble/secp256k1';
import { keccak_256 } from '@noble/hashes/sha3.js';

// ── JSON-RPC types ─────────────────────────────────────────────────────────────

interface RpcRequest {
  jsonrpc: '2.0';
  id: string | number | null;
  method: string;
  params?: Record<string, unknown>;
}

function ok(id: string | number | null, result: unknown): Response {
  return json({ jsonrpc: '2.0', id, result });
}

function err(id: string | number | null, code: number, message: string): Response {
  return json({ jsonrpc: '2.0', id, error: { code, message } });
}

const CORS = {
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Authorization, Content-Type, X-Wallet-Key',
};

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    headers: { 'Content-Type': 'application/json', ...CORS },
  });
}

// ── Tool definitions ───────────────────────────────────────────────────────────

const TOOLS = [
  {
    name: 'predict_timeseries',
    description:
      'Predict the next value in a numeric time series. ' +
      'Automatically pays $0.01 USDC on Base via x402 micropayment.',
    inputSchema: {
      type: 'object',
      properties: {
        series: {
          type: 'array',
          items: { type: 'number' },
          description: '3–1000 historical values, ordered oldest to newest.',
        },
        context: {
          type: 'string',
          description: 'What the series represents, e.g. "monthly revenue USD". Max 200 chars.',
        },
      },
      required: ['series'],
    },
  },
  {
    name: 'detect_anomalies',
    description:
      'Detect anomalies in a numeric time series using z-score analysis. ' +
      'Automatically pays $0.01 USDC on Base via x402 micropayment.',
    inputSchema: {
      type: 'object',
      properties: {
        series: {
          type: 'array',
          items: { type: 'number' },
          description: '3–1000 values to analyze.',
        },
        threshold: {
          type: 'number',
          description: 'Z-score cutoff (default 2.0, range 0–10). Lower = more sensitive.',
        },
        context: {
          type: 'string',
          description: 'What the series represents. Max 200 chars.',
        },
      },
      required: ['series'],
    },
  },
];

// ── Ethereum signing ───────────────────────────────────────────────────────────

function hexToBytes(hex: string): Uint8Array {
  const h = hex.startsWith('0x') ? hex.slice(2) : hex;
  const arr = new Uint8Array(h.length / 2);
  for (let i = 0; i < arr.length; i++) arr[i] = parseInt(h.slice(i * 2, i * 2 + 2), 16);
  return arr;
}

function bytesToHex(b: Uint8Array): string {
  return Array.from(b).map(x => x.toString(16).padStart(2, '0')).join('');
}

function getAddress(privateKeyHex: string): string {
  const pub  = secp.getPublicKey(hexToBytes(privateKeyHex), false); // uncompressed 65 bytes
  const hash = keccak_256(pub.slice(1));                             // drop 0x04 prefix
  return '0x' + bytesToHex(hash.slice(12));                          // last 20 bytes
}

async function personalSign(message: string, privateKeyHex: string): Promise<string> {
  const msgBytes = new TextEncoder().encode(message);
  const prefix   = new TextEncoder().encode(`\x19Ethereum Signed Message:\n${msgBytes.length}`);
  const combined = new Uint8Array([...prefix, ...msgBytes]);
  const hash     = keccak_256(combined);
  const [sig, recovery] = await secp.sign(hash, hexToBytes(privateKeyHex), { recovered: true, der: false });
  const v = (recovery + 27).toString(16).padStart(2, '0');
  return '0x' + bytesToHex(sig) + v;
}

// Sort object keys recursively for canonical JSON (matches Python's json.dumps sort_keys=True)
function sortedJson(val: unknown): string {
  if (val === null || typeof val !== 'object' || Array.isArray(val)) return JSON.stringify(val);
  const obj = val as Record<string, unknown>;
  const sorted = Object.keys(obj).sort().reduce((acc, k) => {
    acc[k] = JSON.parse(sortedJson(obj[k]));
    return acc;
  }, {} as Record<string, unknown>);
  return JSON.stringify(sorted);
}

// ── x402 payment ───────────────────────────────────────────────────────────────

async function callWithPayment(
  url: string,
  body: Record<string, unknown>,
  privateKeyHex: string,
): Promise<unknown> {
  const init = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };

  let resp = await fetch(url, init);
  if (resp.ok) return resp.json();
  if (resp.status !== 402) throw new Error(`API error: ${resp.status}`);

  const challenge = await resp.json() as { accepts: Array<Record<string, unknown>> };
  const reqs      = challenge.accepts[0];
  const address   = getAddress(privateKeyHex);
  const nonce     = crypto.randomUUID().replace(/-/g, '');

  const payload = {
    x402Version: 2,
    scheme:      reqs['scheme'],
    network:     reqs['network'],
    asset:       reqs['asset'],
    amount:      reqs['amount'],
    payTo:       reqs['payTo'],
    resource:    url,
    from:        address,
    nonce,
    validUntil:  Math.floor(Date.now() / 1000) + ((reqs['maxTimeoutSeconds'] as number) ?? 60),
  };

  const canonical = sortedJson(payload).replace(/,\s*/g, ',').replace(/:\s*/g, ':');
  const signature = await personalSign(canonical, privateKeyHex);
  const signed    = { payload, signature, from: address };
  const header    = btoa(JSON.stringify(signed));

  resp = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', 'PAYMENT-SIGNATURE': header },
    body:    JSON.stringify(body),
  });

  if (!resp.ok) throw new Error(`API error after payment: ${resp.status} ${await resp.text()}`);
  return resp.json();
}

// ── MCP request handler ────────────────────────────────────────────────────────

export async function handleMcpRequest(request: Request, fallbackWalletKey: string | undefined, baseUrl: string): Promise<Response> {
  // Prefer wallet key from the request header so each user pays from their own wallet.
  // Falls back to the operator's env-var key if no header is present.
  const authHeader = request.headers.get('Authorization') ?? request.headers.get('X-Wallet-Key') ?? '';
  const walletKey  = authHeader.replace(/^Bearer\s+/i, '').trim() || fallbackWalletKey;
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: CORS });
  }

  if (request.method === 'GET') {
    return new Response(JSON.stringify({ name: 'easy-predict', version: '1.0.0', tools: TOOLS.map(t => t.name) }), {
      headers: { 'Content-Type': 'application/json', ...CORS },
    });
  }

  if (request.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405, headers: CORS });
  }

  let rpc: RpcRequest;
  try {
    rpc = await request.json() as RpcRequest;
  } catch {
    return err(null, -32700, 'Parse error');
  }

  const { id, method, params } = rpc;

  if (method === 'initialize') {
    return ok(id, {
      protocolVersion: '2024-11-05',
      capabilities: { tools: {} },
      serverInfo: { name: 'easy-predict', version: '1.0.0' },
      instructions:
        'Use predict_timeseries to forecast numeric series or detect_anomalies to find outliers. ' +
        'Each call costs $0.01 USDC on Base via x402, paid automatically.',
    });
  }

  if (method === 'notifications/initialized') {
    // Notifications have no id; if a client sends one with an id, ack it.
    return id != null ? ok(id, {}) : new Response(null, { status: 204, headers: CORS });
  }

  if (method === 'tools/list') {
    return ok(id, { tools: TOOLS });
  }

  if (method === 'tools/call') {
    if (!walletKey) {
      return ok(id, {
        content: [{ type: 'text', text: 'Error: No wallet key provided. Add your Base wallet private key as an Authorization header: Authorization: Bearer 0xYOUR_PRIVATE_KEY' }],
        isError: true,
      });
    }

    const name      = (params as Record<string, unknown>)?.name as string;
    const args      = (params as Record<string, unknown>)?.arguments as Record<string, unknown> ?? {};
    const endpoint  = name === 'predict_timeseries' ? 'timeseries' : name === 'detect_anomalies' ? 'anomaly-detection' : null;

    if (!endpoint) return err(id, -32601, `Unknown tool: ${name}`);

    try {
      const body: Record<string, unknown> = { series: args['series'] };
      if (args['context'])   body['context']   = String(args['context']).slice(0, 200);
      if (args['threshold']) body['threshold']  = args['threshold'];

      const result = await callWithPayment(`${baseUrl}/${endpoint}`, body, walletKey);
      return ok(id, {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      });
    } catch (e) {
      return ok(id, {
        content: [{ type: 'text', text: `Error: ${(e as Error).message}` }],
        isError: true,
      });
    }
  }

  return err(id, -32601, `Method not found: ${method}`);
}
