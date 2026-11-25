# Copied unchanged from the checkpoint baidu/ERNIE-5.5-Flash-Preview @ de3ffd3e (decoding_ernie.py), Apache-2.0,
# Copyright 2026 Baidu. The served template and output parser are these files as shipped.
"""Completion parsing for the existing ERNIE protocol.

Token-ID input preserves the distinction between ordinary marker-looking text
and real control IDs. String-only input cannot recover that distinction.
The legacy parameter format has no type/escaping field: ambiguous values are
rejected unless a tool schema resolves them or raw-string mode is requested.
"""

from __future__ import annotations

import itertools
import json
import re
import uuid
from dataclasses import dataclass

from .encoding_ernie import CALL, CALL_END, END, MASK, START, THINK, THINK_END, RenderedChat

_MARKERS = (THINK, THINK_END, CALL, CALL_END, END, MASK, START)
_MARKER_RE = re.compile("|".join(re.escape(s) for s in _MARKERS))
_INVOKE = re.compile(r'<invoke\s+name="([^"]+)">')
_PARAMETER = re.compile(r'<parameter\s+name="([^"]+)">')


class CompletionParseError(ValueError):
    """Malformed, truncated, or inherently ambiguous completion."""


class AmbiguousArgumentsError(CompletionParseError):
    """The legacy wire format does not uniquely identify argument types."""


@dataclass(frozen=True)
class _Document:
    text: str
    markers: tuple[tuple[int, str], ...]


def _from_text(text: str) -> _Document:
    return _Document(text, tuple((m.start(), m.group()) for m in _MARKER_RE.finditer(text)))


def _from_ids(token_ids, tokenizer) -> _Document:
    marker_ids = {tokenizer.convert_tokens_to_ids(s): s for s in _MARKERS}
    if None in marker_ids or len(marker_ids) != len(_MARKERS):
        raise CompletionParseError("Tokenizer does not contain the ERNIE protocol markers")
    text = []
    markers = []
    ordinary = []
    length = 0

    def flush():
        nonlocal length
        if ordinary:
            value = tokenizer.decode(ordinary, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            text.append(value)
            length += len(value)
            ordinary.clear()

    for token_id in token_ids:
        if token_id in marker_ids:
            flush()
            marker = marker_ids[token_id]
            markers.append((length, marker))
            text.append(marker)
            length += len(marker)
        else:
            ordinary.append(token_id)
    flush()
    return _Document("".join(text), tuple(markers))


def _argument_candidates(raw):
    values = [raw]
    try:
        parsed = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError):
        return values
    # String parameters are emitted raw by the encoder; literal quotes belong
    # to the string. Only non-string JSON creates a second interpretation.
    if not isinstance(parsed, str):
        values.append(parsed)
    return values


def _resolve_arguments(name, raw, tools, argument_mode):
    if argument_mode == "raw":
        return dict(raw)
    schema = None
    if tools is not None:
        found = False
        for tool in tools:
            function = tool.get("function", tool)
            if function.get("name") == name:
                schema = function.get("parameters", {})
                found = True
                break
        if not found:
            raise CompletionParseError(f"Generated unknown tool {name!r}")
    candidates = [_argument_candidates(value) for value in raw.values()]
    possibilities = 1
    for values in candidates:
        possibilities *= len(values)
    # Filter each property first; this makes ordinary schemas linear instead
    # of exploring 2**N combinations of string-vs-number interpretations.
    validator = None
    if schema is not None:
        from jsonschema.validators import validator_for

        cls = validator_for(schema)
        cls.check_schema(schema)
        validator = cls(schema)
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        for i, key in enumerate(raw):
            prop = properties.get(key)
            if prop is not None:
                candidates[i] = [v for v in candidates[i] if validator.evolve(schema=prop).is_valid(v)]
        possibilities = 1
        for values in candidates:
            possibilities *= len(values)
    if possibilities > 256:
        raise AmbiguousArgumentsError("Too many unresolved argument types; provide a more specific tool schema")
    valid = []
    for values in itertools.product(*candidates):
        candidate = dict(zip(raw, values))
        if validator is None or validator.is_valid(candidate):
            valid.append(candidate)
        if len(valid) > 1:
            raise AmbiguousArgumentsError(
                f"Tool {name!r} has ambiguous string/JSON argument types; "
                "provide an unambiguous schema or use argument_mode='raw'"
            )
    if not valid:
        raise CompletionParseError(f"Arguments for tool {name!r} do not match its schema")
    return valid[0]


def _parse_calls(text, tools, argument_mode):
    calls = []
    i = 0

    def skip_space(position):
        return re.compile(r"\s*").match(text, position).end()

    while (i := skip_space(i)) < len(text):
        match = _INVOKE.match(text, i)
        if match is None:
            raise CompletionParseError(f"Expected invoke tag at tool block offset {i}")
        name = match.group(1)
        i = match.end()
        arguments = {}
        while True:
            i = skip_space(i)
            if text.startswith("</invoke>", i):
                i += len("</invoke>")
                break
            match = _PARAMETER.match(text, i)
            if match is None:
                raise CompletionParseError(f"Expected parameter or closing invoke at tool block offset {i}")
            key = match.group(1)
            if key in arguments:
                raise CompletionParseError(f"Duplicate parameter {key!r}")
            end = text.find("</parameter>", match.end())
            if end < 0:
                raise CompletionParseError(f"Missing closing parameter tag for {key!r}")
            arguments[key] = text[match.end():end]
            i = end + len("</parameter>")
        values = _resolve_arguments(name, arguments, tools, argument_mode)
        calls.append({
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(values, ensure_ascii=False, allow_nan=False)},
        })
    if not calls:
        raise CompletionParseError("Empty tool-call block")
    return calls


def _parse_document(document, tools, argument_mode, allow_incomplete):
    text = document.text
    markers = list(document.markers)
    reasoning = ""
    content = ""
    calls = []
    pending = ""
    i = 0
    terminated = False

    def next_marker(position):
        return next(((at, marker) for at, marker in markers if at >= position), (len(text), None))

    at, marker = next_marker(0)
    if at == 0 and marker == THINK:
        i = len(THINK)
        at, marker = next_marker(i)
        reasoning = text[i:at]
        if marker is None:
            if not allow_incomplete:
                raise CompletionParseError("Missing closing think tag")
            return dict(role="assistant", reasoning_content=reasoning, content="", tool_calls=[],
                        complete=False, raw_text=text, unparsed_tool_text="")
        if marker != THINK_END:
            raise CompletionParseError(f"Expected closing think tag, found {marker}")
        i = at + len(THINK_END)
    at, marker = next_marker(i)
    content = text[i:at]
    i = at
    if marker == CALL:
        body_start = i + len(CALL)
        at, marker = next_marker(body_start)
        if marker is None and allow_incomplete:
            pending = text[body_start:]
            i = len(text)
        elif marker != CALL_END:
            raise CompletionParseError("Missing closing tool-call tag or unexpected control token inside tools")
        else:
            calls = _parse_calls(text[body_start:at], tools, argument_mode)
            i = at + len(CALL_END)
            at, marker = next_marker(i)
            if text[i:at].strip():
                raise CompletionParseError("Unexpected text after the tool-call block")
            i = at
    if marker in (END, MASK):
        terminated = True
        i = at + len(marker)
        if text[i:].strip():
            raise CompletionParseError("Unexpected text after assistant end")
    elif marker is not None:
        raise CompletionParseError(f"Unexpected control marker {marker}")
    if not terminated and not allow_incomplete:
        raise CompletionParseError("Missing assistant end token; pass allow_incomplete=True for truncated output")
    return dict(
        role="assistant", reasoning_content=reasoning, content=content, tool_calls=calls,
        complete=terminated, raw_text=text, unparsed_tool_text=pending,
    )


def parse_completion(
    completion,
    *,
    tokenizer=None,
    rendered: RenderedChat | None = None,
    tools=None,
    argument_mode="schema",
    allow_incomplete=False,
):
    """Parse newly generated assistant output, optionally restoring its prefill.

    Pass generated IDs plus tokenizer for source-preserving decoding. Do not
    pass the whole prompt+completion sequence. `rendered` is the result of
    render_chat(..., add_generation_prompt=True). String input is supported
    for legacy servers, but marker-looking literal text is ambiguous there.

    Default argument_mode='schema' refuses unresolved type ambiguity. 'raw'
    returns every tool parameter as a string, for inspection only.
    """
    if argument_mode not in ("schema", "raw"):
        raise ValueError("argument_mode must be 'schema' or 'raw'")
    if rendered is not None and not rendered.has_generation_prompt:
        raise ValueError("rendered must have add_generation_prompt=True")
    prefix = rendered.generation_prefix if rendered is not None else ()
    if isinstance(completion, str):
        prefix_text = "".join(s.text for s in prefix)
        document = _from_text(prefix_text + completion)
    else:
        if tokenizer is None:
            raise ValueError("Token-ID completion parsing requires tokenizer")
        ids = completion.tolist() if hasattr(completion, "tolist") else list(completion)
        if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(tokenizer) for i in ids):
            raise ValueError("completion must be a one-dimensional sequence of valid token IDs")
        while ids and ids[-1] == tokenizer.pad_token_id:
            ids.pop()
        prefix_ids = tokenizer.encode_segments(prefix) if prefix else []
        document = _from_ids(prefix_ids + ids, tokenizer)
    return _parse_document(document, tools, argument_mode, allow_incomplete)
