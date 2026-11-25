"""Chat side of the server: ERNIE's protocol as the checkpoint ships it (server/ernie_proto: encoding_ernie.py renders a
conversation into segments, tokens.py turns segments into ids, decoding_ernie.py parses a completion), plus what the
HTTP front needs around it: the request's thinking decision, tool normalisation, incremental detokenising and a
streaming parser. Pure Python on the head node; the engine never sees text.

Client text can never become a control token: the renderer marks only its own literals as control segments, and
everything else is BPE-encoded without the added tokens (tokens.py) — the checkpoint's own rule, so plain_specials
needs no quoting here.

Parser (decoding_ernie.py's grammar, on token ids so a written marker stays text): the output starts in reasoning when
thinking is on (the generation prompt ends with <ernie:think>), </ernie:think> switches to content, <ernie:tool_call>
starts a tool block of <invoke name="…"><parameter name="…">value</parameter>…</invoke> that ends at
</ernie:tool_call>; arguments are resolved against the tool's JSON schema as decoding_ernie does (string vs JSON
value), raw strings when the schema cannot decide. <|ernie:im_end|> / <|ERNIE_MASK|> end the reply (generation_config
eos 8 and 2; the scheduler stops there)."""
import json
import os

from tokenizers import Tokenizer

from ernie_proto import decoding_ernie as dec
from ernie_proto.encoding_ernie import (CALL, CALL_END, END, MASK, START, THINK, THINK_END, EncodeSegment,
                                        render_chat)
from ernie_proto.tokens import SegmentTokenizer

try:
    import qlog  # tools/qlog.py (the server puts tools/ on the path)
except ImportError:  # pragma: no cover — chat.py used on its own
    qlog = None

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_ALIAS = {"minimal": "low", "xhigh": "high", "max": "high"}  # ERNIE's template: low / medium / high


class TemplateError(ValueError):
    pass


def thinking_mode(body):
    """Thinking on/off and level from any of the request forms clients use, read in one place so the template and
    the parser see the same decision: reasoning_effort (OpenAI; as vLLM it overrides chat_template_kwargs'),
    chat_template_kwargs {enable_thinking (wins for on/off), reasoning_effort}, reasoning {enabled, effort}
    (OpenRouter style), think (Ollama style: bool or a level).
    Effort "none" turns thinking off wherever it comes from. Levels: minimal → low, xhigh / max → high, unset → high.
    Returns (enabled, level or None)."""
    kw = body.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise ValueError("chat_template_kwargs must be an object")
    r = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
    think = body.get("think")
    eff = next((v for v in (body.get("reasoning_effort"), kw.get("reasoning_effort"), r.get("effort"),
                            think if isinstance(think, str) else None) if v is not None), None)
    if eff is not None and (not isinstance(eff, str) or eff.lower() not in EFFORTS):
        raise ValueError(f"reasoning effort must be one of {', '.join(EFFORTS)}, got {eff!r}")
    eff = eff.lower() if eff is not None else None
    if "enable_thinking" in kw:
        en = kw["enable_thinking"] not in (False, 0, "false")
    elif r.get("enabled") is not None:
        en = bool(r["enabled"])
    elif isinstance(think, bool):
        en = think
    else:
        en = True
    if eff == "none":
        en = False
    return en, ((EFFORT_ALIAS.get(eff, eff) or "high") if en else None)


def normalize_tools(tools):
    """Tools as vLLM's request model dumps them: {"type": "function", "function": {name, description, parameters}}
    (description / parameters null when absent; strict only when set; other keys dropped)."""
    if tools is None:
        return None
    if not isinstance(tools, list) or not tools:
        raise ValueError("`tools` must not be an empty array. Either provide at least one tool or omit the field "
                         "entirely.")
    out = []
    for t in tools:
        f = (t or {}).get("function") or {}
        if not isinstance(f.get("name"), str):
            raise ValueError("each tool needs function.name")
        g = {"name": f["name"], "description": f.get("description"), "parameters": f.get("parameters")}
        if f.get("strict") is not None:
            g["strict"] = f["strict"]
        out.append({"type": "function", "function": g})
    return out


def normalize_messages(messages):
    """OpenAI chat messages → the renderer's: "developer" is a system message; text parts stay; media parts are
    refused (this checkpoint reads text only); a null content is empty."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"messages[{i}] must be an object")
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if not isinstance(p, dict) or p.get("type") not in ("text", "input_text"):
                    raise ValueError(f"messages[{i}]: only text content parts are supported (this model reads text "
                                     "only)")
            m["content"] = [{"type": "text", "text": p.get("text", "")} for p in c]
        elif c is None:
            m["content"] = ""
        out.append(m)
    return out


class Chat:
    def __init__(self, model_dir):
        self.seg = SegmentTokenizer(model_dir)
        self.tok = self.seg.full  # all added tokens (ids ↔ text)
        sp = self.seg.special
        self.ids = {s: sp[s] for s in (MASK, START, END, THINK, THINK_END, CALL, CALL_END)}
        self.think_end, self.call, self.call_end = sp[THINK_END], sp[CALL], sp[CALL_END]
        self.markers = set(self.ids.values())
        gc = json.load(open(os.path.join(model_dir, "generation_config.json")))
        eos = gc.get("eos_token_id", [])
        self.eos = set(eos if isinstance(eos, list) else [eos])
        # server defaults = generation_config (vLLM: request → these → OpenAI defaults)
        self.defaults = {"temperature": gc.get("temperature", 1.0), "top_k": gc.get("top_k", -1),
                         "top_p": gc.get("top_p", 1.0)}

    def render(self, messages, tools=None, thinking=True, level="high", kwargs=None, add_generation_prompt=True):
        """→ encoding_ernie.RenderedChat. kwargs: the client's chat_template_kwargs (system_settings, agent_storage,
        examples, options pass through; enable_thinking / reasoning_effort are already decided in thinking / level)."""
        kw = {k: v for k, v in (kwargs or {}).items() if k in ("system_settings", "agent_storage", "examples", "options")}
        try:
            return render_chat(messages, tools, add_generation_prompt=add_generation_prompt, enable_thinking=thinking,
                               reasoning_effort=level if thinking else "high", **kw)
        except (TypeError, KeyError, json.JSONDecodeError) as e:
            raise TemplateError(str(e)) from e

    def encode_chat(self, rendered):
        return self.seg.encode_segments(rendered.segments)

    def force_ids(self, named=None):
        """tool_choice forcing: required — the <ernie:tool_call> token alone (what follows it is one BPE run in the
        template's text, so any longer cut would split it where the model never sees a split); named — the template's
        own tokens of a call to `named` (a dummy parameter after it), cut to the longest token prefix whose text stays
        within '<ernie:tool_call>\n<invoke name="NAME' — a cut the model has seen, which leaves the name's end and the
        arguments to it."""
        if named is None:
            return [self.call]
        segs = [EncodeSegment(CALL + "\n", True), EncodeSegment('<invoke name="', True), EncodeSegment(named),
                EncodeSegment('">', True), EncodeSegment('<parameter name="', True), EncodeSegment("x"),
                EncodeSegment('">', True)]
        ids = self.seg.encode_segments(segs)
        want, k = CALL + '\n<invoke name="' + named, 1
        while k < len(ids) and want.startswith(self.decode(ids[:k + 1], False)):
            k += 1
        return ids[:k]

    def encode(self, text):
        """A raw prompt (/v1/completions, /tokenize): the tokenizer with its added tokens, as vLLM."""
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=True):
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)


class Detok:
    """Incremental detokeniser (vLLM's prefix/read offsets): each step decodes only a short window, so a long answer
    costs O(n); an unfinished UTF-8 sequence (replacement character at the end) is held back until it completes.
    skip: drop special tokens (completions); the parser decodes ordinary ids with them kept (decoding_ernie)."""

    def __init__(self, chat, skip=True):
        self.chat, self.skip, self.ids, self.pre, self.read = chat, skip, [], 0, 0

    def _d(self, ids):
        return self.chat.decode(ids, self.skip)

    def add(self, ids):
        self.ids += ids
        head = self._d(self.ids[self.pre:self.read])
        txt = self._d(self.ids[self.pre:])
        if len(txt) <= len(head) or txt.endswith("�"):
            return ""
        self.pre, self.read = self.read, len(self.ids)
        return txt[len(head):]

    def flush(self):
        head = self._d(self.ids[self.pre:self.read])
        txt = self._d(self.ids[self.pre:])
        self.ids, self.pre, self.read = [], 0, 0
        return txt[len(head):]


class Parser:
    """Token ids of a reply → events (kind, value): ("reasoning", text), ("content", text), ("tool_delta", chunk),
    ("tool", call). tools: the request's tool list (None: tool blocks stay text); suppress: tool_choice "none"."""

    def __init__(self, chat, thinking, tools=None, suppress=False, max_calls=None):
        # suppress (tool_choice "none"): a tool block the model writes anyway is dropped (neither a call nor text)
        self.c, self.tools, self.suppress = chat, (tools or None), bool(tools) and suppress
        self.state = "reasoning" if thinking else "content"
        self.det = Detok(chat, skip=False)
        self.block, self.ncall, self.max_calls = [], 0, max_calls  # max_calls: parallel_tool_calls false / named
        # → the first call only (vLLM keeps the first)

    def _text(self, ev, txt):
        if txt:
            kind = "reasoning" if self.state == "reasoning" else "content"
            if ev and ev[-1][0] == kind:
                ev[-1] = (kind, ev[-1][1] + txt)
            else:
                ev.append((kind, txt))

    def _calls(self, ev):
        if self.suppress:
            return
        text = self.c.decode(self.block, False)
        calls = None
        for mode in ("schema", "raw"):
            try:
                calls = dec._parse_calls(text, self.tools, mode)
                break
            except (dec.CompletionParseError, ValueError, TypeError) as e:
                if qlog and mode == "raw":
                    qlog.warn("parse", f"tool block kept as text: {e}")
        if calls is None:  # unparseable: the client gets the text
            self.state = "content"
            self._text(ev, CALL + text + CALL_END)
            return
        for c in calls:
            if self.max_calls is not None and self.ncall >= self.max_calls:
                break
            d = {"index": self.ncall, "id": c["id"], "type": "function", "function": dict(c["function"])}
            ev.append(("tool_delta", d))
            ev.append(("tool", c))
            self.ncall += 1

    def feed(self, ids, final=False):
        ev = []
        for t in ids:
            if self.state == "tool":
                if t == self.c.call_end:
                    self._calls(ev)
                    if self.state == "tool":
                        self.state = "after"
                elif t not in self.c.markers:
                    self.block.append(t)
                continue
            if t in self.c.markers:
                self._text(ev, self.det.flush())
                if t == self.c.think_end and self.state == "reasoning":
                    self.state = "content"
                elif t == self.c.call and self.tools is not None and self.state in ("content", "after"):
                    self.state, self.block = "tool", []
                elif t == self.c.call and self.state != "reasoning":  # no tools offered: the block is text
                    self._text(ev, CALL)
                elif t == self.c.call_end and self.state != "reasoning":
                    self._text(ev, CALL_END)
                continue  # other markers (the end, a stray role or think tag) are not text
            if self.state == "reasoning" or self.state in ("content", "after"):
                self._text(ev, self.det.add([t]))
        if final:
            if self.state == "tool":  # cut off inside a tool block (length): the client gets the text
                self.state = "content"
                if not self.suppress:
                    self._text(ev, CALL + self.c.decode(self.block, False))
            self._text(ev, self.det.flush())
        return ev
