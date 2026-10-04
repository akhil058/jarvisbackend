'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const { createServer, sanitizeMessages, buildSystemPrompt } = require('./server');

function listen(server) {
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server.address().port)));
}

// A fake "AI provider" so no real key, network or money is involved.
function startMock(handler) {
  const seen = [];
  const server = http.createServer((req, res) => {
    let raw = '';
    req.on('data', (c) => (raw += c));
    req.on('end', () => {
      seen.push({ headers: req.headers, body: raw ? JSON.parse(raw) : null });
      handler(req, res, seen);
    });
  });
  return listen(server).then((port) => ({ server, port, seen }));
}

async function setup(overrides = {}, mockHandler) {
  const mock = await startMock(mockHandler || ((req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ content: [{ type: 'text', text: 'Hello from the mock.' }] }));
  }));
  const config = {
    port: 0, accessToken: 'test-token-1234567890', apiKey: 'sk-test', model: 'test-model', maxTokens: 100,
    apiUrl: `http://127.0.0.1:${mock.port}/v1/messages`, maxBodyBytes: 100 * 1024, timeoutMs: 2000,
    ratePerMinute: 30, trustProxy: false, ...overrides,
  };
  const app = createServer(config);
  const port = await listen(app);
  const call = (path, { method = 'POST', token = config.accessToken, body } = {}) =>
    fetch(`http://127.0.0.1:${port}${path}`, {
      method,
      headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
      body: body === undefined ? undefined : typeof body === 'string' ? body : JSON.stringify(body),
    });
  const close = () => { app.close(); mock.server.close(); };
  return { call, mock, close };
}

test('GET /health works without a token', async () => {
  const t = await setup();
  const r = await t.call('/health', { method: 'GET', token: '' });
  assert.equal(r.status, 200);
  t.close();
});

test('rejects missing and wrong tokens', async () => {
  const t = await setup();
  const body = { messages: [{ role: 'user', content: 'hi' }] };
  assert.equal((await t.call('/chat', { token: '', body })).status, 401);
  assert.equal((await t.call('/chat', { token: 'nope', body })).status, 401);
  assert.equal(t.mock.seen.length, 0, 'provider must not be called for unauthenticated requests');
  t.close();
});

test('happy path returns the reply and sends key, model and system prompt upstream', async () => {
  const t = await setup();
  const r = await t.call('/chat', { body: { messages: [{ role: 'user', content: 'Hi' }], userName: 'Akhil' } });
  assert.equal(r.status, 200);
  assert.deepEqual(await r.json(), { reply: 'Hello from the mock.' });
  const sent = t.mock.seen[0];
  assert.equal(sent.headers['x-api-key'], 'sk-test');
  assert.equal(sent.body.model, 'test-model');
  assert.match(sent.body.system, /Akhil/);
  t.close();
});

test('bad input gets 400', async () => {
  const t = await setup();
  assert.equal((await t.call('/chat', { body: '{not json' })).status, 400);
  assert.equal((await t.call('/chat', { body: { messages: [] } })).status, 400);
  assert.equal((await t.call('/chat', { body: { messages: [{ role: 'assistant', content: 'x' }] } })).status, 400);
  t.close();
});

test('oversized body gets 413', async () => {
  const t = await setup({ maxBodyBytes: 200 });
  const r = await t.call('/chat', { body: { messages: [{ role: 'user', content: 'x'.repeat(500) }] } });
  assert.equal(r.status, 413);
  t.close();
});

test('provider errors become friendly 502s without leaking details', async () => {
  const t = await setup({}, (req, res) => { res.writeHead(500); res.end('{"secret":"details"}'); });
  const r = await t.call('/chat', { body: { messages: [{ role: 'user', content: 'hi' }] } });
  assert.equal(r.status, 502);
  assert.doesNotMatch(JSON.stringify(await r.json()), /secret/);
  t.close();
});

test('rejected provider key gets a clear message', async () => {
  const t = await setup({}, (req, res) => { res.writeHead(401); res.end('{}'); });
  const r = await t.call('/chat', { body: { messages: [{ role: 'user', content: 'hi' }] } });
  assert.equal(r.status, 502);
  assert.match((await r.json()).error, /key was rejected/);
  t.close();
});

test('slow provider gets 504', async () => {
  const t = await setup({ timeoutMs: 150 }, () => { /* never answers */ });
  const r = await t.call('/chat', { body: { messages: [{ role: 'user', content: 'hi' }] } });
  assert.equal(r.status, 504);
  t.close();
});

test('rate limit returns 429', async () => {
  const t = await setup({ ratePerMinute: 2 });
  const body = { messages: [{ role: 'user', content: 'hi' }] };
  assert.equal((await t.call('/chat', { body })).status, 200);
  assert.equal((await t.call('/chat', { body })).status, 200);
  assert.equal((await t.call('/chat', { body })).status, 429);
  t.close();
});

test('sanitizeMessages merges, trims and enforces user-last', () => {
  const out = sanitizeMessages([
    { role: 'assistant', content: 'stale' },
    { role: 'user', content: 'a' },
    { role: 'user', content: 'b' },
    { role: 'assistant', content: 'c' },
    { role: 'user', content: 'd' },
    { role: 'system', content: 'ignored' },
  ]);
  assert.deepEqual(out, [
    { role: 'user', content: 'a\nb' },
    { role: 'assistant', content: 'c' },
    { role: 'user', content: 'd' },
  ]);
  assert.equal(sanitizeMessages([{ role: 'user', content: 'x' }, { role: 'assistant', content: 'y' }]), null);
});

test('user name is cleaned before it reaches the prompt', () => {
  assert.doesNotMatch(buildSystemPrompt('Ak<hil>\nIgnore previous'), /[<>]/);
  assert.doesNotMatch(buildSystemPrompt(undefined), /user's name is/);
});
