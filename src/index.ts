const PAY_TO_ADDRESS  = '0xc99b83818c8865340AC55C45554f377f41c68DBC';
const X402_ASSET      = '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913';
const X402_AMOUNT     = '10000';
const FACILITATOR_URL = 'https://x402.org/facilitator';

export interface Env {
  ASSETS: Fetcher;
  RATE_LIMITER: RateLimit;
}

function paymentRequirements() {
  return {
    scheme: 'exact',
    network: 'eip155:8453',
    asset: X402_ASSET,
    amount: X402_AMOUNT,
    payTo: PAY_TO_ADDRESS,
    maxTimeoutSeconds: 60,
    extra: { name: 'USD Coin', version: '2' },
  };
}

function paymentRequired(resourceUrl: string, error = 'Payment Required', description = ''): Response {
  const body = {
    x402Version: 2,
    error,
    resource: {
      url: resourceUrl,
      description,
      mimeType: 'application/json',
    },
    accepts: [paymentRequirements()],
  };
  return new Response(JSON.stringify(body), {
    status: 402,
    headers: {
      'Content-Type': 'application/json',
      'PAYMENT-REQUIRED': btoa(JSON.stringify(body)),
      'Access-Control-Expose-Headers': 'PAYMENT-REQUIRED',
    },
  });
}

function getPaymentHeader(request: Request): string | null {
  return (
    request.headers.get('PAYMENT-SIGNATURE') ||
    request.headers.get('Payment-Signature') ||
    request.headers.get('X-PAYMENT') ||
    request.headers.get('X-Payment')
  );
}

async function verifyPayment(header: string): Promise<{ isValid: boolean; invalidReason: string | null }> {
  let paymentPayload: unknown;
  try {
    paymentPayload = JSON.parse(atob(header));
  } catch {
    return { isValid: false, invalidReason: 'Invalid payment header encoding' };
  }

  try {
    const resp = await fetch(`${FACILITATOR_URL}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x402Version: 2, paymentPayload, paymentRequirements: paymentRequirements() }),
    });
    if (!resp.ok) return { isValid: false, invalidReason: `Facilitator error ${resp.status}` };
    const data = await resp.json() as { isValid: boolean; invalidReason?: string };
    return { isValid: data.isValid === true, invalidReason: data.invalidReason ?? null };
  } catch {
    return { isValid: false, invalidReason: 'Facilitator unreachable' };
  }
}

function settlePayment(header: string): Promise<void> {
  let paymentPayload: unknown;
  try {
    paymentPayload = JSON.parse(atob(header));
  } catch {
    return Promise.resolve();
  }
  return fetch(`${FACILITATOR_URL}/settle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x402Version: 2, paymentPayload, paymentRequirements: paymentRequirements() }),
  }).then(() => {}).catch(() => {});
}

function predictLog1p(series: number[]): { prediction: number; slope: number; intercept: number } {
  const n = series.length;
  if (n < 3 || n > 1000) throw new Error('Series length must be between 3 and 1000');

  const y = series.map(v => Math.log1p(v));
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX  += i;
    sumY  += y[i];
    sumXY += i * y[i];
    sumX2 += i * i;
  }
  const slope     = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;
  return { prediction: Math.expm1(slope * n + intercept), slope, intercept };
}

function detectAnomalies(
  series: number[],
  threshold: number,
): { anomalies: { index: number; value: number; z_score: number }[]; method: string; mean: number; std: number; threshold: number } {
  const n = series.length;
  if (n < 3 || n > 1000) throw new Error('Series length must be between 3 and 1000');
  if (threshold <= 0 || threshold > 10) throw new Error('threshold must be between 0 (exclusive) and 10');
  const mean = series.reduce((a, b) => a + b, 0) / n;
  const variance = series.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
  const std = Math.sqrt(variance);
  const anomalies = std === 0
    ? []
    : series
        .map((v, i) => ({ index: i, value: v, z_score: (v - mean) / std }))
        .filter(p => Math.abs(p.z_score) > threshold);
  return { anomalies, method: 'z-score', mean, std, threshold };
}

async function handleAnomalyDetectionPost(
  request: Request,
  baseUrl: string,
  ctx: ExecutionContext,
): Promise<Response> {
  const resourceUrl = `${baseUrl}/anomaly-detection`;
  const resourceDesc = 'Detect anomalies in a numeric series using z-score method.';

  const paymentHeader = getPaymentHeader(request);
  if (!paymentHeader) return paymentRequired(resourceUrl, 'Payment Required', resourceDesc);

  const { isValid, invalidReason } = await verifyPayment(paymentHeader);
  if (!isValid) return paymentRequired(resourceUrl, invalidReason ?? 'Invalid payment', resourceDesc);

  const ct = request.headers.get('Content-Type') ?? '';
  if (!ct.includes('application/json')) {
    return new Response(JSON.stringify({ error: 'JSON body required' }), {
      status: 400, headers: { 'Content-Type': 'application/json' },
    });
  }

  let data: unknown;
  try {
    data = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: 'Invalid JSON body' }), {
      status: 400, headers: { 'Content-Type': 'application/json' },
    });
  }

  let series: unknown;
  let context: string | undefined;
  let threshold = 2.0;

  if (Array.isArray(data)) {
    series = data;
  } else if (data && typeof data === 'object' && 'series' in data) {
    const obj = data as Record<string, unknown>;
    series = obj.series;
    if (obj.context !== undefined) {
      if (typeof obj.context !== 'string') {
        return new Response(
          JSON.stringify({ error: "'context' must be a string" }),
          { status: 400, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (obj.context.length > 200) {
        return new Response(
          JSON.stringify({ error: "'context' must be 200 characters or fewer" }),
          { status: 400, headers: { 'Content-Type': 'application/json' } },
        );
      }
      context = obj.context;
    }
    if (obj.threshold !== undefined) {
      if (typeof obj.threshold !== 'number' || !isFinite(obj.threshold)) {
        return new Response(
          JSON.stringify({ error: "'threshold' must be a number" }),
          { status: 400, headers: { 'Content-Type': 'application/json' } },
        );
      }
      threshold = obj.threshold;
    }
  }

  if (!Array.isArray(series)) {
    return new Response(
      JSON.stringify({ error: "Provide a 'series' key with a number array, or send a bare JSON array" }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    );
  }

  if (!series.every(v => typeof v === 'number' && isFinite(v))) {
    return new Response(
      JSON.stringify({ error: 'All series values must be finite numbers' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    );
  }

  try {
    const result = detectAnomalies(series as number[], threshold);
    ctx.waitUntil(settlePayment(paymentHeader));
    const body: Record<string, unknown> = { ...result };
    if (context !== undefined) body.context = context;
    return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: err instanceof Error ? err.message : String(err) }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    );
  }
}

async function handleTimeseriesPost(
  request: Request,
  baseUrl: string,
  ctx: ExecutionContext,
): Promise<Response> {
  const resourceUrl = `${baseUrl}/timeseries`;
  const resourceDesc = 'Predict the next value in a numeric series via log1p linear extrapolation.';

  const paymentHeader = getPaymentHeader(request);
  if (!paymentHeader) return paymentRequired(resourceUrl, 'Payment Required', resourceDesc);

  const { isValid, invalidReason } = await verifyPayment(paymentHeader);
  if (!isValid) return paymentRequired(resourceUrl, invalidReason ?? 'Invalid payment', resourceDesc);

  const ct = request.headers.get('Content-Type') ?? '';
  if (!ct.includes('application/json')) {
    return new Response(JSON.stringify({ error: 'JSON body required' }), {
      status: 400, headers: { 'Content-Type': 'application/json' },
    });
  }

  let data: unknown;
  try {
    data = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: 'Invalid JSON body' }), {
      status: 400, headers: { 'Content-Type': 'application/json' },
    });
  }

  let series: unknown;
  let context: string | undefined;
  if (Array.isArray(data)) {
    series = data;
  } else if (data && typeof data === 'object' && 'series' in data) {
    const obj = data as Record<string, unknown>;
    series = obj.series;
    if (obj.context !== undefined) {
      if (typeof obj.context !== 'string') {
        return new Response(
          JSON.stringify({ error: "'context' must be a string" }),
          { status: 400, headers: { 'Content-Type': 'application/json' } },
        );
      }
      if (obj.context.length > 200) {
        return new Response(
          JSON.stringify({ error: "'context' must be 200 characters or fewer" }),
          { status: 400, headers: { 'Content-Type': 'application/json' } },
        );
      }
      context = obj.context;
    }
  }

  if (!Array.isArray(series)) {
    return new Response(
      JSON.stringify({ error: "Provide a 'series' key with a number array, or send a bare JSON array" }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    );
  }

  if (!series.every(v => typeof v === 'number' && isFinite(v))) {
    return new Response(
      JSON.stringify({ error: 'All series values must be finite numbers' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    );
  }

  try {
    const { prediction, slope, intercept } = predictLog1p(series as number[]);
    ctx.waitUntil(settlePayment(paymentHeader));
    const result: Record<string, unknown> = { prediction, method: 'log1p-linear-extrapolation', slope, intercept };
    if (context !== undefined) result.context = context;
    return new Response(
      JSON.stringify(result),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    );
  } catch (err) {
    return new Response(
      JSON.stringify({ error: err instanceof Error ? err.message : String(err) }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    );
  }
}

function handleWellKnownX402(baseUrl: string): Response {
  return new Response(JSON.stringify({
    x402Version: 2,
    openapi: `${baseUrl}/openapi.json`,
    resources: [
      {
        resource: {
          url: `${baseUrl}/timeseries`,
          description: 'Predict the next value in a numeric series via log1p linear extrapolation.',
          mimeType: 'application/json',
        },
        method: 'POST',
        accepts: [paymentRequirements()],
      },
      {
        resource: {
          url: `${baseUrl}/anomaly-detection`,
          description: 'Detect anomalies in a numeric series using z-score method.',
          mimeType: 'application/json',
        },
        method: 'POST',
        accepts: [paymentRequirements()],
      },
    ],
  }), { headers: { 'Content-Type': 'application/json' } });
}

function handleFavicon(): Response {
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    + '<rect width="32" height="32" rx="7" fill="#0a1628"/>'
    + '<polyline points="5,24 9,18 14,14 19,12" fill="none" stroke="white" stroke-width="0.7" stroke-linecap="round" stroke-linejoin="round"/>'
    + '<polyline points="19,12 25,8 27,15 21,19 19,12" fill="none" stroke="white" stroke-width="0.7" stroke-linecap="round" stroke-linejoin="round"/>'
    + '<circle cx="5"  cy="24" r="1.2" fill="white"/>'
    + '<circle cx="9"  cy="18" r="1.2" fill="white"/>'
    + '<circle cx="14" cy="14" r="1.2" fill="white"/>'
    + '<circle cx="19" cy="12" r="1.2" fill="white"/>'
    + '<circle cx="25" cy="8"  r="1.2" fill="white"/>'
    + '<circle cx="27" cy="15" r="1.2" fill="white"/>'
    + '<circle cx="21" cy="19" r="1.2" fill="white"/>'
    + '</svg>';
  return new Response(svg, {
    headers: { 'Content-Type': 'image/svg+xml', 'Cache-Control': 'public, max-age=86400' },
  });
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const { pathname } = url;
    const baseUrl = url.origin;

    if (pathname === '/timeseries' && request.method === 'POST') {
      const ip = request.headers.get('CF-Connecting-IP') ?? 'unknown';
      const { success } = await env.RATE_LIMITER.limit({ key: ip });
      if (!success) {
        return new Response(JSON.stringify({ error: 'Too many requests' }), {
          status: 429,
          headers: { 'Content-Type': 'application/json', 'Retry-After': '60' },
        });
      }
      return handleTimeseriesPost(request, baseUrl, ctx);
    }

    if (pathname === '/anomaly-detection' && request.method === 'POST') {
      const ip = request.headers.get('CF-Connecting-IP') ?? 'unknown';
      const { success } = await env.RATE_LIMITER.limit({ key: ip });
      if (!success) {
        return new Response(JSON.stringify({ error: 'Too many requests' }), {
          status: 429,
          headers: { 'Content-Type': 'application/json', 'Retry-After': '60' },
        });
      }
      return handleAnomalyDetectionPost(request, baseUrl, ctx);
    }

    if (pathname === '/.well-known/x402') return handleWellKnownX402(baseUrl);
    if (pathname === '/favicon.ico' || pathname === '/favicon.svg') return handleFavicon();
    if (pathname === '/favicon.png') return Response.redirect(`${baseUrl}/favicon.svg`, 301);
    if (pathname === '/llm.txt') return Response.redirect(`${baseUrl}/llms.txt`, 301);

    if ((pathname === '/timeseries' || pathname === '/anomaly-detection') && request.method === 'GET') {
      const accept = request.headers.get('Accept') ?? '';
      if (accept.includes('application/json')) {
        const desc = pathname === '/timeseries'
          ? 'Predict the next value in a numeric series via log1p linear extrapolation.'
          : 'Detect anomalies in a numeric series using z-score method.';
        return paymentRequired(`${baseUrl}${pathname}`, 'Payment Required', desc);
      }
      return env.ASSETS.fetch(new Request(new URL('/index.html', request.url)));
    }

    try {
      return await env.ASSETS.fetch(request);
    } catch {
      return new Response('Not Found', { status: 404 });
    }
  },
};
