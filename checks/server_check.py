"""Checks against a running server — no reference data needed, stdlib only (Python >= 3.8), any machine that reaches
the server.  python3 checks/server_check.py [base URL] [--only smoke,api,tools,thinking,needle]
                                            [--needle 1000000] [--key KEY]

  smoke     /v1/models · /health · chat (non-stream, greedy) · chat stream with thinking (reasoning_content / content
            split, usage chunk) · a next turn carrying the reply's reasoning · four at once · a closed stream stops the
            request (/metrics) · same seed twice
  api       vLLM's errors (a prompt past the window, max_tokens cut to the room, out-of-range and unavailable fields,
            an unknown model) · stop strings (streamed and not) · penalties / min_p · usage counts the end token ·
            /tokenize /detokenize /version /v1/models/{id}
  tools     OpenAI tools: tool_choice required / named (thinking on and off) / auto with typed arguments / none · a
            named tool that is not offered → 400 · streamed tool-call deltas · parallel_tool_calls false → one call
  thinking  ERNIE's controls: chat_template_kwargs {"enable_thinking": false} · reasoning_effort low / medium / high
            and none named · a bad effort → 400 · include_reasoning false · thinking budgets · a loop prompt
  exact     (only when named in --only) four prompts greedy: alone, all four at once, alone again → the same text; seeded sampling: alone, alone again
            every time (the engine is deterministic for a seed and batch-invariant)
  reuse     (only when named in --only: phase 1 keeps no prefix cache) a five-turn conversation over a ~6K-token
            document: from the second turn on, ≥ 90% of each prompt is reused
  needle    one request per --needle length (cut to the server's context): three short facts hidden at 10% / 50% /
            90% of a long filler text, asked at once — the answer must hold all three
  key       (with --key, which every request then sends) no key / a wrong key → 401, the right key → 200, /health
            without a key → 200
Requests have thinking off (chat_template_kwargs enable_thinking false) unless a case is about thinking.
One line per case; exit code 1 if any case fails."""
import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:8888/v1")
ap.add_argument("--only", default="smoke,api,tools,thinking,needle")
ap.add_argument("--needle", default="1000000", help="prompt lengths in tokens (approximate; cut to the context)")
ap.add_argument("--key", default=None, help="the server's API key (server.api_key); sent with every request")
a = ap.parse_args()
BASE = a.url.rstrip("/")
ROOT = BASE.rsplit("/v1", 1)[0]
bad = 0
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "add", "description": "Add two integers", "parameters": {
             "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]


def say(ok, tag, msg):
    global bad
    bad += not ok
    print(f"\033[36m[check {tag}]\033[0m \033[{32 if ok else 31}m{msg}\033[0m", flush=True)


def url(path):
    """/v1 paths under the base URL; /health, /metrics, /version, /tokenize and /detokenize at the server's root."""
    return (ROOT if path in ("/health", "/metrics", "/version", "/tokenize", "/detokenize") else BASE) + path


def headers(key):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def post(path, body, stream=False, timeout=1800):
    req = urllib.request.Request(url(path), json.dumps(body).encode(), headers(a.key))
    r = urllib.request.urlopen(req, timeout=timeout)
    return r if stream else json.loads(r.read())


def call(path, obj=None, key=a.key, timeout=1800):
    """(status, JSON) — an HTTP error is an answer here, not an exception. No obj: GET."""
    req = urllib.request.Request(url(path), None if obj is None else json.dumps(obj).encode(), headers(key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def body(messages, **kw):
    """A chat request; thinking off unless the case names an effort or its own chat_template_kwargs."""
    b = {"model": MODEL, "messages": messages}
    if "reasoning_effort" not in kw and "chat_template_kwargs" not in kw:
        b["chat_template_kwargs"] = {"enable_thinking": False}
    b.update(kw)
    return b


def chat(messages, **kw):
    return post("/chat/completions", body(messages, **kw))


def ask(messages, **kw):
    """chat() as (status, JSON)."""
    return call("/chat/completions", body(messages, **kw))


def events(b):
    """The SSE chunks of a streamed chat request."""
    for line in post("/chat/completions", dict(b, stream=True), stream=True):
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[6:])


c, j = call("/models")
if c != 200:
    say(False, "models", f"/v1/models → {c} {json.dumps(j)[:120]}")
    sys.exit(1)
m = j["data"][0]
MODEL, CTX = m["id"], int(m.get("max_model_len") or 0)
only = set(a.only.split(",")) | ({"key"} if a.key else set())


# ---- key ----------------------------------------------------------------------------------------------------------
def key():
    b = body([{"role": "user", "content": "hi"}], max_tokens=4)
    c, j = call("/chat/completions", b, key=None)
    say(c == 401, "key", f"no key → {c} {json.dumps(j)[:80]}")
    c, j = call("/chat/completions", b, key="wrong")
    say(c == 401, "key", f"wrong key → {c}")
    c, j = call("/chat/completions", b)
    say(c == 200, "key", f"right key → {c}")
    c, j = call("/health", key=None)
    say(c == 200, "key", f"/health without key → {c}")


# ---- smoke --------------------------------------------------------------------------------------------------------
def smoke():
    say(True, "models", f"{MODEL} · context {CTX:,}")
    c, j = call("/health", key=None)
    say(c == 200 and j.get("status") == "ok" and "busy" in j, "health", f"{c} · {json.dumps(j)}")
    t0 = time.time()
    r = chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}], max_tokens=32,
             temperature=0)
    c = r["choices"][0]
    say("paris" in (c["message"]["content"] or "").lower(), "chat",
        f"{c['message']['content']!r} · finish {c['finish_reason']} · usage {r['usage']['prompt_tokens']}+"
        f"{r['usage']['completion_tokens']} · {time.time() - t0:.2f} s")

    b = body([{"role": "user", "content": "Is 91 a prime number? Think briefly, then answer."}],
             reasoning_effort="low", max_tokens=1024, stream_options={"include_usage": True}, seed=1)
    t0, first, reasoning, content, usage = time.time(), None, "", "", None
    for ch in events(b):
        usage = ch.get("usage") or usage
        for ci in ch.get("choices", []):
            d = ci.get("delta", {})
            if (d.get("reasoning_content") or d.get("content")) and first is None:
                first = time.time() - t0
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
    dt = time.time() - t0
    say(bool(reasoning) and bool(content) and usage is not None, "stream",
        f"thinking {len(reasoning)} chars · answer {content.strip()[:60]!r} · first piece {first or 0:.2f} s · "
        f"{usage and usage['completion_tokens']} tokens · "
        f"{(usage['completion_tokens'] if usage else 0) / max(dt - (first or 0), 1e-6):.1f} tok/s")

    # the next turn sends the reply back with its reasoning; reuse is shown, not required (phase 1: no prefix cache)
    turn2 = b["messages"] + [{"role": "assistant", "content": content, "reasoning_content": reasoning},
                             {"role": "user", "content": "And 97?"}]
    t0 = time.time()
    u = chat(turn2, reasoning_effort="low", max_tokens=1)["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    say(u["prompt_tokens"] > 0, "next turn", f"prompt {u['prompt_tokens']} · reused {cached} "
                                             f"({100 * cached / max(u['prompt_tokens'], 1):.0f}%) · first token "
                                             f"{time.time() - t0:.2f} s")

    res = [None] * 4

    def one(i):
        rr = chat([{"role": "user", "content": f"Write two sentences about the number {i + 7}."}], max_tokens=128,
                  seed=i + 1)
        res[i] = rr["usage"]["completion_tokens"]
    t0 = time.time()
    th = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    [t.start() for t in th]
    [t.join() for t in th]
    say(all(res), "4 at once", f"{sum(x or 0 for x in res)} tokens · {time.time() - t0:.2f} s")

    def running():
        req = urllib.request.Request(url("/metrics"), headers=headers(a.key))
        for line in urllib.request.urlopen(req, timeout=10).read().decode().splitlines():
            if line.startswith("vllm:num_requests_running"):
                return int(float(line.split()[-1]))
    r = post("/chat/completions", body([{"role": "user", "content": "Count from 1 to 500."}], max_tokens=4000,
                                       stream=True), stream=True)
    for i, _ in enumerate(r):
        if i > 6:
            break
    r.close()
    t0, n = time.time(), None
    while time.time() - t0 < 5:
        n = running()
        if n == 0:
            break
        time.sleep(0.2)
    say(n == 0, "disconnect", f"stream closed · {n} request(s) running after {time.time() - t0:.1f} s")

    b = [{"role": "user", "content": "Name three colours."}]
    a1 = chat(b, max_tokens=48, seed=1)["choices"][0]["message"]["content"]
    a2 = chat(b, max_tokens=48, seed=1)["choices"][0]["message"]["content"]
    say(a1 == a2, "seed", f"same seed twice → {'same' if a1 == a2 else 'DIFFERENT'} text · {a1.strip()[:50]!r}")


# ---- api ----------------------------------------------------------------------------------------------------------
def api():
    W = CTX
    c, j = ask([{"role": "user", "content": "word " * (W + 6000)}], max_tokens=10)
    e = j.get("error", {})
    say(c == 400 and f"maximum context length is {W} tokens" in e.get("message", "") and e.get("param") == "input_tokens",
        "too long", f"{c} · {e.get('message', '')[:150]} · type {e.get('type')}")
    c, j = ask([{"role": "user", "content": "Say OK."}], max_tokens=W)
    say(c == 200, "max clamp", f"max_tokens {W} → {c} · finish {j.get('choices', [{}])[0].get('finish_reason')}")
    for kw, want in (({"max_tokens": 0}, "max_tokens must be at least 1, got 0."),
                     ({"temperature": 3}, "temperature must be in [0, 2]"), ({"min_p": 1.5}, "min_p must be in [0, 1]"),
                     ({"presence_penalty": 3}, "presence_penalty must be in [-2, 2]"),
                     ({"repetition_penalty": 0}, "repetition_penalty must be greater than zero"),
                     ({"logit_bias": {"1": 5}}, "logit_bias is unavailable"),
                     ({"response_format": {"type": "json_object"}}, "response_format")):
        c, j = ask([{"role": "user", "content": "hi"}], **kw)
        msg = j.get("error", {}).get("message", "")
        say(c == 400 and want in msg, "error", f"{kw} → {c} · {msg[:110]}")
    c, j = call("/models/other")
    say(c == 404, "model 404", f"{c} {j.get('error', {}).get('message')}")

    # stop strings: one that spans pieces (streamed), and the same request not streamed
    q = [{"role": "user", "content": "Count from 1 to 20, separated by commas and spaces. Only the numbers."}]
    b = body(q, temperature=0, max_tokens=200, stop=["7, 8"])
    out, fin = "", None
    for ch in events(b):
        for c0 in ch.get("choices", []):
            out += c0["delta"].get("content") or ""
            fin = c0.get("finish_reason") or fin
    say(out.rstrip().endswith("6,") and "8" not in out.split("6,")[-1] and fin == "stop", "stop stream",
        f"{out!r} · finish {fin}")
    c, j = call("/chat/completions", b)
    ch = j["choices"][0] if c == 200 else {}
    o = (ch.get("message") or {}).get("content") or ""
    say(o.rstrip().endswith("6,") and ch.get("finish_reason") == "stop", "stop plain", f"{o!r} · finish "
                                                                                       f"{ch.get('finish_reason')}")

    # penalties + min_p accepted, and they change a greedy answer
    q = [{"role": "user", "content": "List the numbers from 1 to 30 separated by commas."}]
    c1, j1 = ask(q, temperature=0, max_tokens=120)
    c2, j2 = ask(q, temperature=0, max_tokens=120, repetition_penalty=1.5, presence_penalty=1.0, frequency_penalty=1.0)
    c3, j3 = ask(q, temperature=0.8, max_tokens=120, min_p=0.1, seed=5)
    a1 = j1["choices"][0]["message"]["content"] if c1 == 200 else ""
    a2 = j2["choices"][0]["message"]["content"] if c2 == 200 else ""
    say(c1 == c2 == c3 == 200 and a1 != a2, "penalties", f"plain {a1[:40]!r} · penalised {a2[:40]!r} · min_p {c3}")

    # usage counts the end token
    c, j = ask([{"role": "user", "content": "Say OK."}], temperature=0, max_tokens=20)
    txt = j["choices"][0]["message"]["content"]
    c2, j2 = call("/tokenize", {"prompt": txt})
    say(j["usage"]["completion_tokens"] == j2.get("count", -2) + 1, "usage eos",
        f"{txt!r}: completion_tokens {j['usage']['completion_tokens']} = {j2.get('count')} text tokens + end token")

    # endpoints
    c, j = call("/tokenize", {"prompt": "Hello world"})
    say(c == 200 and j.get("count") == len(j.get("tokens") or []) and j.get("max_model_len") == W, "tokenize", f"{j}")
    c2, j2 = call("/detokenize", {"tokens": j.get("tokens") or []})
    say(c2 == 200 and j2.get("prompt") == "Hello world", "detokenize", f"{j2}")
    c, j = call("/tokenize", {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"})
    say(c == 200 and j.get("count", 0) > 5, "tokenize chat", f"count {j.get('count')}")
    c, j = call("/version")
    say(c == 200 and "version" in j, "version", f"{j}")
    c, j = call(f"/models/{MODEL}")
    say(c == 200 and j.get("max_model_len") == W, "model card", f"{j}")


# ---- tools --------------------------------------------------------------------------------------------------------
def tools():
    W = [{"role": "user", "content": "What's the weather in Paris for the next 3 days?"}]
    for think in (True, False):
        eff = {"reasoning_effort": "low"} if think else {}
        c, j = ask(W, tools=TOOLS, tool_choice="required", max_tokens=1500, seed=3, **eff)
        ch = j["choices"][0] if c == 200 else {}
        m = ch.get("message", {})
        tc = m.get("tool_calls") or []
        say(c == 200 and bool(tc) and ch["finish_reason"] == "tool_calls" and m.get("content") == "", "required",
            f"thinking {think}: {c} · calls {[t['function']['name'] + t['function']['arguments'] for t in tc]} · "
            f"finish {ch.get('finish_reason')} · content {m.get('content')!r}")
        c, j = ask([{"role": "user", "content": "Hello, how are you?"}], tools=TOOLS, max_tokens=1500, seed=3,
                   tool_choice={"type": "function", "function": {"name": "add"}}, **eff)
        ch = j["choices"][0] if c == 200 else {}
        m = ch.get("message", {})
        tc = m.get("tool_calls") or []
        say(c == 200 and len(tc) == 1 and tc[0]["function"]["name"] == "add" and ch["finish_reason"] == "stop", "named",
            f"thinking {think}: {c} · calls {[t['function']['name'] + t['function']['arguments'] for t in tc]} · "
            f"finish {ch.get('finish_reason')}")
    c, j = ask([{"role": "user", "content": "Add 2 and 40 using the tool."}], tools=TOOLS, reasoning_effort="low",
               max_tokens=1500, seed=3)
    m = j["choices"][0]["message"] if c == 200 else {}
    args = [json.loads(t["function"]["arguments"]) for t in m.get("tool_calls") or []]
    say(c == 200 and bool(args) and all(isinstance(v, int) for x in args for v in x.values()), "auto typed",
        f"{args} · finish {j['choices'][0]['finish_reason'] if c == 200 else c}")
    c, j = ask([{"role": "user", "content": "Add 2 and 40 using the tool."}], tools=TOOLS, tool_choice="none",
               max_tokens=1500, seed=3)
    m = j["choices"][0]["message"] if c == 200 else {}
    say(c == 200 and not m.get("tool_calls") and "tool_call>" not in (m.get("content") or ""), "none",
        f"tool_calls {m.get('tool_calls')} · content {(m.get('content') or '')[:60]!r}")
    c, j = ask(W, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "nope"}})
    say(c == 400 and "does not match" in j.get("error", {}).get("message", ""), "named bad",
        f"{c} {j.get('error', {}).get('message')}")

    # streaming: reasoning field, tool deltas, usage
    # (the template's default effort: at low ERNIE may leave the thinking empty)
    b = body([{"role": "user", "content": "Add 5 and 6 with the tool."}], tools=TOOLS, reasoning_effort="high",
             stream_options={"include_usage": True}, max_tokens=1500)
    keys, fin, usage, tdel = set(), None, None, []
    for ch in events(b):
        usage = ch.get("usage") or usage
        for c0 in ch.get("choices", []):
            keys |= set(c0["delta"])
            tdel += c0["delta"].get("tool_calls") or []
            fin = c0.get("finish_reason") or fin
    # at T 1.0 ERNIE may close the thinking at once (the first token after the template's open think tag, a plain
    # draw): then there is nothing to stream there; when it thinks, the thinking must come as reasoning_content
    rt = ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    say(bool(tdel) and fin == "tool_calls" and bool(usage) and (rt == 0 or "reasoning_content" in keys), "stream",
        f"delta keys {sorted(keys)} · tool deltas "
        f"{[(d.get('id', '')[:14], d['function'].get('name'), d['function'].get('arguments')) for d in tdel]} · "
        f"finish {fin} · usage {usage}")

    # parallel_tool_calls false: one call
    T = [{"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"}}}}},
         {"type": "function", "function": {"name": "mul", "parameters": {"type": "object", "properties": {
             "a": {"type": "integer"}, "b": {"type": "integer"}}}}}]
    q = [{"role": "user", "content": "Call add with 1 and 2, and mul with 3 and 4, both now in parallel."}]
    c, j = ask(q, tools=T, temperature=0, max_tokens=300)
    n_par = len((j["choices"][0]["message"].get("tool_calls") or []) if c == 200 else [])
    c, j = ask(q, tools=T, temperature=0, max_tokens=300, parallel_tool_calls=False)
    n_one = len((j["choices"][0]["message"].get("tool_calls") or []) if c == 200 else [])
    say(n_one == 1, "parallel false", f"calls: default {n_par} · parallel_tool_calls false {n_one} · finish "
                                      f"{j['choices'][0]['finish_reason'] if c == 200 else c}")


# ---- thinking -----------------------------------------------------------------------------------------------------
def thinking():
    def think(content, **kw):
        """A request with only what the case names (no effort named: the template's default, high)."""
        b = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 2000, "seed": 7}
        b.update(kw)
        c, j = call("/chat/completions", b)
        return c, j, (j["choices"][0]["message"] if c == 200 else {})

    c, j, m = think("What is 17 * 3? Answer with the number.", chat_template_kwargs={"enable_thinking": False})
    say(c == 200 and not m.get("reasoning_content") and "51" in (m.get("content") or ""), "off",
        f"enable_thinking false: {c} · reasoning {len(m.get('reasoning_content') or '')} chars · content "
        f"{(m.get('content') or '')[:30]!r}")
    for eff in (None, "low", "medium", "high"):
        c, j, m = think("What is 17 * 3?", **({"reasoning_effort": eff} if eff else {}))
        say(c == 200 and bool(m.get("reasoning_content")) and bool(m.get("content")), "on",
            f"reasoning_effort {eff or '(none named)'}: {c} · reasoning {len(m.get('reasoning_content') or '')} chars · "
            f"content {(m.get('content') or '')[:30]!r} · out {j.get('usage', {}).get('completion_tokens')} tokens")
    c, j, m = think("hi", reasoning_effort="turbo")
    say(c == 400, "bad effort", f"{c} {json.dumps(j)[:100]}")

    c, j, m = think("What is 6*7?", reasoning_effort="low", include_reasoning=False)
    rt = ((j.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    say(c == 200 and m.get("reasoning_content") is None and rt > 0, "include_reasoning",
        f"reasoning_content {m.get('reasoning_content')!r} · reasoning_tokens {rt}")
    c, j, m = think("What is 6*7?", thinking_token_budget=0)
    say(c == 200 and not m.get("reasoning_content") and "42" in (m.get("content") or ""), "budget",
        f"thinking_token_budget 0: content {m.get('content')!r}")
    hard = "Find all integer solutions of x^3 + y^3 = 1729 with x, y > 0 and explain your reasoning carefully."
    c, j, m = think(hard, thinking_budget=64)
    say(c == 200 and bool(m.get("content")) and 0 < len(m.get("reasoning_content") or "") < 600, "budget",
        f"thinking_budget 64: reasoning {len(m.get('reasoning_content') or '')} chars · content "
        f"{len(m.get('content') or '')} chars · finish {j['choices'][0]['finish_reason'] if c == 200 else c} · out "
        f"{j.get('usage', {}).get('completion_tokens')} tokens")
    c, j, m = think(hard, thinking_budget=1)
    say(c == 200 and bool(m.get("content")), "budget",
        f"thinking_budget 1: reasoning {m.get('reasoning_content')!r} · content {len(m.get('content') or '')} chars")

    loop = ("Before answering, in your thinking write the word 'banana' 600 times in a row separated by spaces, "
            "nothing else, then answer: what colour is a banana?")
    c, j, m = think(loop, max_tokens=3000, temperature=0)
    r = m.get("reasoning_content") or ""
    say(c == 200, "loop", f"reasoning {len(r)} chars ({r.count('banana')} × banana) · content "
                          f"{(m.get('content') or '')[:60]!r} · out {j.get('usage', {}).get('completion_tokens')} "
                          "tokens (see the server's req line: thinking closed (loop)?)")


# ---- exact --------------------------------------------------------------------------------------------------------
def exact():
    """Greedy: the same text alone, all four at once and alone again (the engine's bits do not depend on the batch).
    Sampled with a seed: the same text alone and alone again. A sampled request at once with others may draw another
    text from the same distribution: the DSpark rounds (joint verify lengths, copy drafts only alone) depend on the
    batch, as speculative decoding in other serving engines does — only the distribution is kept, not the sample path."""
    prompts = ["Describe a lighthouse at night in four sentences.", "List five uses of copper and explain one.",
               "Write a short dialogue between a cat and a clock.", "Explain why ice floats, step by step."]

    def run(i, out, greedy):
        kw = {"temperature": 0.0} if greedy else {"seed": 100 + i}
        out[i] = chat([{"role": "user", "content": prompts[i]}], max_tokens=160, **kw)["choices"][0]["message"]["content"]
    res = {}
    for greedy in (True, False):
        alone, together, again = [None] * 4, [None] * 4, [None] * 4
        for i in range(4):
            run(i, alone, greedy)
        if greedy:
            th = [threading.Thread(target=run, args=(i, together, True)) for i in range(4)]
            [t.start() for t in th]
            [t.join() for t in th]
        for i in range(4):
            run(i, again, greedy)
        res[greedy] = [alone[i] == again[i] and (not greedy or alone[i] == together[i]) for i in range(4)]
    g, s = res[True], res[False]
    say(all(g) and all(s), "exact", f"greedy 4 prompts × (alone, at once, alone again): {sum(g)}/4 identical · "
                                    f"seeded sampling × (alone, alone again): {sum(s)}/4 identical")


# ---- filler text --------------------------------------------------------------------------------------------------
WORDS = ("river stone market window garden lantern harbor meadow copper violet thunder orchard canyon silver pocket "
         "ladder candle mirror forest bridge winter summer pepper cotton marble island valley rocket").split()


def filler(n_lines, seed):
    g = random.Random(seed)
    return "\n".join(f"Note {i + 1}: the {g.choice(WORDS)} near the {g.choice(WORDS)} was {g.choice(WORDS)} and "
                     f"{g.choice(WORDS)} on day {g.randint(1, 999)}." for i in range(n_lines))


def tokens_per_line():
    u = chat([{"role": "user", "content": filler(400, 7)}], max_tokens=1)["usage"]
    return u["prompt_tokens"] / 400


# ---- reuse --------------------------------------------------------------------------------------------------------
def reuse():
    tpl = tokens_per_line()
    msgs = [{"role": "system", "content": "Answer questions about these notes.\n\n" + filler(int(6000 / tpl), 11)}]
    worst = 1.0
    for k, q in enumerate(["Which word appears in note 3?", "What day is in note 10?", "Summarize note 25.",
                           "Which notes mention a bridge? Name two.", "What is in the last note?"]):
        msgs.append({"role": "user", "content": q})
        r = chat(msgs, max_tokens=64, seed=k + 1)
        u = r["usage"]
        frac = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) / u["prompt_tokens"]
        if k:
            worst = min(worst, frac)
        msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"] or ""})
    say(worst >= 0.9, "reuse", f"5 turns over ~6K tokens · least reused from turn 2 on: {100 * worst:.1f}% (needs ≥ 90%)")


# ---- needle -------------------------------------------------------------------------------------------------------
def needle():
    """One request per --needle length: three short facts at 10% / 50% / 90% of a filler text, all asked at once (a
    length past the server's context is cut to it; the prompt is counted by /tokenize and its filler fitted to the
    length — long fillers' line numbers take more tokens than the short probe's)."""
    tpl = tokens_per_line()
    q = "\n\nWhat are the access codes for the red, green and blue vaults? Reply with the three codes only, in that order."
    for n in [int(x) for x in a.needle.split(",") if x]:
        if CTX and n > CTX - 1024:
            n = CTX - 1024
        rng = random.Random(n)
        facts = [(v, d, f"{rng.randint(1000, 9999)}-QX") for v, d in (("red", 0.1), ("green", 0.5), ("blue", 0.9))]

        def build(k):
            lines = filler(k, 23).split("\n")
            for v, d, code in reversed(facts):  # deepest first, so the earlier insertions keep their places
                at = int(len(lines) * d)
                lines[at:at] = [f"Important: the access code for the {v} vault is {code}."]
            return [{"role": "user", "content": "\n".join(lines) + q}]

        k = int((n - 300) / tpl)
        for _ in range(4):  # fit the filler: scale its lines by the counted length until it is within the target
            msgs = build(k)
            c, j = call("/tokenize", body(msgs))
            got = j.get("count", 0) if c == 200 else 0
            if not got or got <= n:
                break
            k = int(k * n / got * 0.999)
        t0 = time.time()
        r = chat(msgs, max_tokens=64, temperature=0)
        ans = r["choices"][0]["message"]["content"] or ""
        found = [code in ans for _, _, code in facts]
        say(all(found), "needle", f"{r['usage']['prompt_tokens']:,} tokens · depth 10/50/90%: "
                                  f"{''.join('✓' if f else '✗' for f in found)} · {time.time() - t0:.1f} s")


for name, fn in (("key", key), ("smoke", smoke), ("api", api), ("tools", tools), ("thinking", thinking),
                 ("exact", exact), ("reuse", reuse), ("needle", needle)):
    if name in only:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — one failing case must not hide the others
            say(False, name, f"error: {e!r}"[:200])
sys.exit(1 if bad else 0)
