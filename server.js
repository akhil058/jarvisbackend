
'use strict';

const http = require('node:http');
const crypto = require('node:crypto');

function loadConfig(env = process.env) {
  return {
    port: Number(env.PORT || 8787),
    accessToken: env.ACCESS_TOKEN || '',
    apiKey: env.GEMINI_API_KEY || '',
    apiUrl:
      'https://generativelanguage.googleapis.com/v1beta/models',
    model: env.AI_MODEL || 'gemini-2.5-flash',
    maxTokens: Number(env.MAX_OUTPUT_TOKENS || 1024),
    timeoutMs: Number(env.REQUEST_TIMEOUT_MS || 30000),
  };
}

function buildSystemPrompt(userName = 'User') {
  const name =
    typeof userName === 'string' && userName.trim()
      ? userName.trim().slice(0, 60)
      : 'User';

  return `
You are Jarvis, a helpful personal AI voice assistant.
You are speaking with ${name}.

Instructions:
- Give clear, friendly, accurate answers.
- Keep spoken answers concise unless more detail is requested.
- Explain technical topics step by step for beginners.
- Never claim to have performed an action that you did not perform.
- If you do not know something, say so honestly.
- Follow the user's latest request.
`;
}

function sanitizeMessages(input) {
  if (!Array.isArray(input)) {
    return [];
  }

  return input
    .filter(
      (message) =>
        message &&
        (message.role === 'user' ||
          message.role === 'assistant') &&
        typeof message.content === 'string' &&
        message.content.trim().length > 0
    )
    .slice(-20)
    .map((message) => ({
      role: message.role,
      content: message.content.trim().slice(0, 12000),
    }));
}

function createServer(config) {
  const rateLimits = new Map();

  function sendJson(res, statusCode, data) {
    res.writeHead(statusCode, {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'no-store',
      'x-content-type-options': 'nosniff',
    });

    res.end(JSON.stringify(data));
  }

  function getRequestBody(req) {
    return new Promise((resolve, reject) => {
      let body = '';
      let size = 0;
      const maxSize = 100 * 1024;

      req.on('data', (chunk) => {
        size += chunk.length;

        if (size > maxSize) {
          reject(new Error('BODY_TOO_LARGE'));
          req.destroy();
          return;
        }

        body += chunk;
      });

      req.on('end', () => {
        try {
          resolve(JSON.parse(body || '{}'));
        } catch {
          reject(new Error('INVALID_JSON'));
        }
      });

      req.on('error', reject);
    });
  }

  function isRateLimited(req) {
    const forwarded = req.socket.remoteAddress || 'unknown';
    const now = Date.now();
    const windowMs = 60_000;
    const maxRequests = 30;

    let entry = rateLimits.get(forwarded);

    if (!entry || now - entry.start > windowMs) {
      entry = { start: now, count: 0 };
    }

    entry.count += 1;
    rateLimits.set(forwarded, entry);

    if (rateLimits.size > 1000) {
      for (const [key, value] of rateLimits) {
        if (now - value.start > windowMs) {
          rateLimits.delete(key);
        }
      }
    }

    return entry.count > maxRequests;
  }

  const server = http.createServer(async (req, res) => {
    const url = new URL(
      req.url || '/',
      `http://${req.headers.host || 'localhost'}`
    );

    if (req.method === 'GET' && url.pathname === '/health') {
      return sendJson(res, 200, { ok: true });
    }

    if (req.method !== 'POST' || url.pathname !== '/chat') {
      return sendJson(res, 404, { error: 'Not found.' });
    }

    if (isRateLimited(req)) {
      return sendJson(res, 429, {
        error: 'Too many requests. Please wait a moment.',
      });
    }

    const authorization = req.headers.authorization || '';
    const expectedToken = `Bearer ${config.accessToken}`;

    if (
      !config.accessToken ||
      authorization.length !== expectedToken.length ||
      !crypto.timingSafeEqual(
        Buffer.from(authorization),
        Buffer.from(expectedToken)
      )
    ) {
      return sendJson(res, 401, {
        error: 'Unauthorized. Check the backend access token.',
      });
    }

    let data;

    try {
      data = await getRequestBody(req);
    } catch (err) {
      if (err.message === 'BODY_TOO_LARGE') {
        return sendJson(res, 413, {
          error: 'Request is too large.',
        });
      }

      return sendJson(res, 400, {
        error: 'Invalid JSON request.',
      });
    }

    const messages = sanitizeMessages(data.messages);

    if (messages.length === 0) {
      return sendJson(res, 400, {
        error: 'Please provide a message.',
      });
    }

    if (!config.apiKey) {
      return sendJson(res, 500, {
        error: 'Gemini API key is not configured on the backend.',
      });
    }

    const controller = new AbortController();
    const timer = setTimeout(
      () => controller.abort(),
      config.timeoutMs
    );

    let upstream;

    try {
      const endpoint =
        `${config.apiUrl}/${encodeURIComponent(config.model)}:generateContent`;

      upstream = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-goog-api-key': config.apiKey,
        },
        body: JSON.stringify({
          systemInstruction: {
            parts: [
              {
                text: buildSystemPrompt(data.userName),
              },
            ],
          },
          contents: messages.map((message) => ({
            role:
              message.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: message.content }],
          })),
          generationConfig: {
            maxOutputTokens: config.maxTokens,
          },
        }),
        signal: controller.signal,
      });
    } catch (err) {
      if (err && err.name === 'AbortError') {
        return sendJson(res, 504, {
          error: 'Gemini took too long to respond.',
        });
      }

      return sendJson(res, 502, {
        error: 'Could not reach Gemini. Check your internet connection.',
      });
    } finally {
      clearTimeout(timer);
    }

    if (!upstream.ok) {
      if (upstream.status === 401 || upstream.status === 403) {
        return sendJson(res, 502, {
          error:
            'Gemini rejected the API key. Check GEMINI_API_KEY and API access.',
        });
      }

      if (upstream.status === 429) {
        return sendJson(res, 429, {
          error:
            'Gemini rate limit or quota reached. Check your API quota.',
        });
      }

      return sendJson(res, 502, {
        error: `Gemini request failed with status ${upstream.status}.`,
      });
    }

    let result;

    try {
      result = await upstream.json();
    } catch {
      return sendJson(res, 502, {
        error: 'Gemini returned an invalid response.',
      });
    }

    const text = Array.isArray(
      result.candidates?.[0]?.content?.parts
    )
      ? result.candidates[0].content.parts
          .filter((part) => typeof part.text === 'string')
          .map((part) => part.text)
          .join(' ')
          .trim()
      : '';

    if (!text) {
      return sendJson(res, 502, {
        error:
          'Gemini returned no text. Try asking your question again.',
      });
    }

    return sendJson(res, 200, { text });
  });

  return server;
}

module.exports = {
  createServer,
  loadConfig,
  sanitizeMessages,
  buildSystemPrompt,
};

if (require.main === module) {
  const config = loadConfig();
  const missing = [];

  if (!config.apiKey) {
    missing.push('GEMINI_API_KEY');
  }

  if (!config.accessToken || config.accessToken.length < 24) {
    missing.push('ACCESS_TOKEN (at least 24 characters)');
  }

  if (missing.length > 0) {
    console.error(
      `Missing or invalid configuration: ${missing.join(', ')}`
    );
    console.error('Check your backend/.env file.');
    process.exit(1);
  }

  const server = createServer(config);

  server.listen(config.port, '0.0.0.0', () => {
    console.log(
      `JarvisAI backend listening on port ${config.port} (model: ${config.model})`
    );
  });
}