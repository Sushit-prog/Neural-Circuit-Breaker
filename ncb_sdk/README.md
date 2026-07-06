# ncb-sdk

Python SDK for the [Neural Circuit Breaker](https://github.com/your-org/neural-circuit-breaker) safety API.

## Quickstart

```python
from ncb_sdk import NeuralCircuitBreaker

ncb = NeuralCircuitBreaker(base_url="http://localhost:8000", circuit_id="demo")

result = ncb.check("some user text")
if result.allowed:
    # proceed with LLM call
else:
    print(result.fallback_message)
```

## Decorator usage

```python
@ncb.protect
def call_llm(prompt: str) -> str:
    return llm.complete(prompt)

# Raises NCBBlockedError if blocked, NCBConnectionError if backend unreachable
response = call_llm("Hello!")
```

## Fail-closed design

If the NCB backend is unreachable or times out, the SDK raises an exception
rather than silently allowing requests through. A broken connection to the
safety backend must never result in unprotected calls — this is a deliberate
fail-closed design choice, matching the backend's own philosophy.

## Installation

```bash
pip install -e ./ncb_sdk
```

## Exceptions

- `NCBError` — base exception
- `NCBConnectionError` — backend unreachable
- `NCBTimeoutError` — request timed out
- `NCBBlockedError` — request blocked (carries `.result: CheckResult`)

## License

MIT
