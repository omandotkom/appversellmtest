from __future__ import annotations

from app.providers.base import ChatResult
from app import runner


class _StreamingProvider:
    async def complete(self, model, messages, **kwargs):
        return ChatResult(
            status_code=200,
            message={"content": "OK"},
            usage={"prompt_tokens": 5, "completion_tokens": 10},
            ttft_ms=0,
            chunks=10,
            done=True,
            latency_ms=100,
        )


async def test_streaming_check_handles_zero_ms_ttft():
    result = await runner.check_streaming(_StreamingProvider(), "m", 60, {})
    assert result["status"] == "pass"
    assert result["ttft_ms"] == 0
    assert result["generation_ms"] == 100
    assert result["tok_per_s"] == 100.0
    assert result["itl_ms"] == 11.1