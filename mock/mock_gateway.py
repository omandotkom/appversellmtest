"""Mock OpenAI-compatible providers for local development and smoke tests.

Run:  python mock/mock_gateway.py     (listens on http://127.0.0.1:4010)

Two independent providers are mounted:

* ``http://127.0.0.1:4010/a`` — models such as ``mock-good``, ``mock-flaky``,
  ``mock-no-tools`` and ``mock-blocked``.
* ``http://127.0.0.1:4010/b`` — a second catalog, also containing
  ``mock-good`` so duplicate model names across providers are exercised.

Point a provider's base URL at one of those prefixes and give it any dummy API
key. No real gateway is required.
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ~1.4KB of valid HTML so the landing-page task's 800-byte artifact check passes.
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nimbus Notes — AI note-taking</title>
<style>
:root{--bg:#0f172a;--fg:#e2e8f0;--acc:#38bdf8}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);line-height:1.6}
header{padding:24px;border-bottom:1px solid #1e293b;display:flex;justify-content:space-between;align-items:center}
.brand{font-weight:700;font-size:20px}
.hero{max-width:760px;margin:0 auto;padding:80px 24px;text-align:center}
.hero h1{font-size:44px;margin-bottom:16px}
.hero p{font-size:18px;color:#94a3b8;margin-bottom:28px}
.cta{display:inline-block;background:var(--acc);color:#0f172a;font-weight:600;padding:14px 28px;border-radius:10px;text-decoration:none}
.features{max-width:960px;margin:0 auto;padding:40px 24px;display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px}
.feature{background:#1e293b;border-radius:12px;padding:24px}
.feature h2{font-size:18px;margin-bottom:8px;color:var(--acc)}
footer{padding:28px;text-align:center;color:#64748b;border-top:1px solid #1e293b;margin-top:40px}
</style>
</head>
<body>
<header><span class="brand">Nimbus Notes</span><nav>Product &nbsp; Pricing &nbsp; Docs</nav></header>
<main>
<section class="hero"><h1>Notes that think with you</h1>
<p>Capture ideas and let Nimbus summarize, connect, and resurface them automatically.</p>
<a class="cta" href="#">Start free</a></section>
<section class="features">
<div class="feature"><h2>AI summaries</h2><p>Distill long notes into clear action items in one click.</p></div>
<div class="feature"><h2>Smart search</h2><p>Find any note by meaning, not just keywords.</p></div>
<div class="feature"><h2>Sync everywhere</h2><p>Real-time sync across web, desktop, and mobile.</p></div>
</section>
</main>
<footer>© 2026 Nimbus Notes. All rights reserved.</footer>
</body>
</html>
"""


def _entry(name, mode="chat", blocked=False, tools=True, json_mode=True,
           cost_in=1e-6, cost_out=3e-6, context=128000):
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"mock/{name}",
            "input_cost_per_token": cost_in,
            "output_cost_per_token": cost_out,
        },
        "model_info": {
            "id": f"id-{name}",
            "mode": mode,
            "blocked": blocked,
            "litellm_provider": "mock",
            "supports_function_calling": tools,
            "supports_response_schema": json_mode,
            "max_input_tokens": context,
            "supported_openai_params": ["tools", "response_format", "max_tokens",
                                        "stream", "temperature", "stream_options"],
        },
    }


PROFILE_MODELS = {
    "a": [
        _entry("mock-good"),
        _entry("mock-flaky"),
        _entry("mock-lazy"),
        _entry("mock-broken"),
        _entry("mock-no-tools", tools=False),
        _entry("mock-no-json", json_mode=False),
        _entry("mock-legacy-embed", mode="embedding"),
        _entry("mock-blocked", blocked=True),
        _entry("mock-slow", cost_in=1.5e-6, cost_out=4.5e-6),
    ],
    "b": [
        _entry("mock-good", cost_in=2e-6, cost_out=5e-6),
        _entry("mock-flaky", cost_in=2e-6, cost_out=5e-6),
        _entry("mock-lazy", cost_in=2e-6, cost_out=5e-6),
        _entry("mock-rate-limited"),
        _entry("mock-small-context", context=4096),
        _entry("mock-no-json", json_mode=False),
        _entry("mock-slow", cost_in=2.5e-6, cost_out=6e-6),
    ],
}


def _tool_call(name, args):
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def _usage_for(messages):
    last_user = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            last_user = message.get("content") or ""
            break
    prompt_tokens = max(1, len(last_user) // 4)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": 45,
            "total_tokens": prompt_tokens + 45}


def _logical(model, messages, body):
    """Compute the assistant message for a request (shared by stream/non-stream)."""
    first = messages[0] if messages else {}
    is_bench = first.get("role") == "system" and "coding agent" in str(first.get("content") or "")

    if is_bench:
        tool_msgs = sum(1 for message in messages if message.get("role") == "tool")
        if model == "mock-good":
            if tool_msgs == 0:
                return {"role": "assistant", "content": None,
                        "tool_calls": [_tool_call("write_file", {"path": "index.html", "content": HTML})]}
            return {"role": "assistant", "content": "Selesai. index.html sudah dibuat."}
        if model == "mock-flaky":
            if tool_msgs == 0:
                return {"role": "assistant", "content": None,
                        "tool_calls": [_tool_call("edit_file", "{this is not valid json")]}
            if tool_msgs == 1:
                return {"role": "assistant", "content": None,
                        "tool_calls": [_tool_call("read_file", {"path": "styles.css"})]}
            if tool_msgs == 2:
                return {"role": "assistant", "content": None,
                        "tool_calls": [_tool_call("write_file", {"path": "index.html", "content": HTML})]}
            return {"role": "assistant", "content": "Recovered and finished."}
        if model == "mock-lazy":
            return {"role": "assistant", "content": "Tentu, ini landing page Anda!"}
        if model == "mock-broken":
            return {"role": "assistant", "content": None,
                    "tool_calls": [_tool_call("run_command", {"command": "ls"})]}
        if tool_msgs == 0:
            return {"role": "assistant", "content": None,
                    "tool_calls": [_tool_call("write_file", {"path": "index.html", "content": HTML})]}
        return {"role": "assistant", "content": "Done."}

    if body.get("tools"):
        return {"role": "assistant", "content": None,
                "tool_calls": [_tool_call("get_weather", {"city": "Jakarta"})]}
    if body.get("response_format"):
        return {"role": "assistant", "content": '{"status": "ok"}'}
    return {"role": "assistant", "content": "OK"}


def _msg(model, message, usage):
    return JSONResponse({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage,
    })


def _chunk(model, delta):
    return f"data: {json.dumps({'id': 'chatcmpl-mock', 'object': 'chat.completion.chunk', 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"


def _stream_response(model, message, usage, stream_options):
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []

    async def gen():
        if content:
            for index in range(0, len(content), 12):
                yield _chunk(model, {"content": content[index:index + 12]})
                await asyncio.sleep(0.02)
        for call in tool_calls:
            function = call["function"]
            yield _chunk(model, {"tool_calls": [{"index": 0, "id": call["id"], "type": "function",
                                                 "function": {"name": function["name"], "arguments": ""}}]})
            await asyncio.sleep(0.02)
            arguments = function["arguments"]
            for index in range(0, len(arguments), 16):
                yield _chunk(model, {"tool_calls": [{"index": 0, "function": {"arguments": arguments[index:index + 16]}}]})
                await asyncio.sleep(0.02)
        if stream_options and stream_options.get("include_usage"):
            yield f"data: {json.dumps({'id': 'chatcmpl-mock', 'object': 'chat.completion.chunk', 'model': model, 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _error(message, status=400):
    return JSONResponse({"error": {"message": message, "type": "mock_error"}}, status_code=status)


def make_app(profile: str) -> FastAPI:
    models = PROFILE_MODELS[profile]
    names = {model["model_name"] for model in models}
    sub = FastAPI(title=f"Mock Gateway {profile.upper()}")

    @sub.get("/v2/model/info")
    def model_info():
        return {"data": models, "total_count": len(models), "current_page": 1, "total_pages": 1}

    @sub.get("/v1/models")
    def list_models():
        return {"object": "list",
                "data": [{"id": model["model_name"], "object": "model"} for model in models]}

    @sub.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        model = body.get("model") or ""
        messages = body.get("messages") or []

        if model not in names:
            return _error(f"model '{model}' not found", 404)
        if model == "mock-blocked":
            return _error("model is blocked", 403)
        if model == "mock-legacy-embed":
            return _error("This is not a chat model (embedding deployment)", 400)
        if model == "mock-rate-limited":
            return _error("rate limit exceeded (mock)", 429)
        if model == "mock-small-context":
            prompt = messages[-1].get("content") or "" if messages else ""
            if len(prompt) > 40000:
                return _error("context length exceeded for this model", 400)
        if body.get("tools") and model == "mock-no-tools":
            return _error("tools are not supported for this model")
        if body.get("response_format") and model == "mock-no-json":
            return _error("response_format is not supported for this model")

        message = _logical(model, messages, body)
        usage = _usage_for(messages)

        if model == "mock-slow":
            await asyncio.sleep(0.4)

        if body.get("stream"):
            return _stream_response(model, message, usage, body.get("stream_options"))
        return _msg(model, message, usage)

    return sub


app = FastAPI(title="Mock Gateway")
app.mount("/a", make_app("a"))
app.mount("/b", make_app("b"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=4010, log_level="warning")