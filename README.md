# Neural Circuit Breaker

A circuit-breaker state machine for LLM prompt-injection safety. Unlike a static
classifier that runs every request through the same pipeline regardless of load,
this system applies the circuit breaker pattern — a well-understood reliability
mechanism from distributed systems — to content safety. When a user (or attacker)
repeatedly sends blocked content, the system trips open and stops running the
detector entirely, returning a safe fallback response without the latency cost of
inference. This is real-time detection with graceful degradation, not just a
wrapper around a model.

## Architecture

```
                    ┌──────────────────────┐
                    │   POST /v1/check     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ CircuitBreaker        │
                    │ .should_allow()       │
                    └──────┬───────┬───────┘
                     OPEN  │       │  CLOSED / HALF_OPEN
                           │       │
              ┌────────────▼─┐     │
              │  Block +      │     │
              │  Fallback     │     │
              └───────────────┘     │
                                    │
                    ┌───────────────▼───────────────┐
                    │  FastFilter.check()            │
                    │  (regex, <15ms)                │
                    └───────┬───────────┬───────────┘
                     unsafe │           │ safe
                            │           │
                   ┌────────▼──┐  ┌─────▼──────────────────┐
                   │ Block +   │  │ DeepClassifier.check()  │
                   │ Fallback  │  │ (ML, ~450ms on CPU)     │
                   └───────────┘  └─────┬───────────┬───────┘
                              unsafe    │           │ safe
                                        │           │
                               ┌────────▼──┐  ┌─────▼─────────────┐
                               │ Block +   │  │ record_success()   │
                               │ Fallback  │  │ -> pass            │
                               └───────────┘  └───────────────────┘
```

Every blocked path (circuit OPEN, fast filter match, deep classifier flag)
attaches a safe fallback response via `FallbackRouter`. The caller always
receives something usable — never a bare rejection.

## Key Engineering Decisions

| Decision | Why |
|---|---|
| WATCH/MULTI for state transitions | Eliminates read-then-write races under concurrent requests; two workers can't both claim the HALF_OPEN probe slot |
| Fail-closed on Redis loss | If the state store is unreachable, the system blocks rather than silently allowing potentially unsafe content through |
| Fail-closed on model load failure | DeepClassifier exposes `is_ready`; routes check it explicitly and block with a 503-class response if the model didn't load |
| Two-tier detection (regex then ML) | Regex catches obvious patterns in <15ms; ML only runs on the subset that slips past, keeping average-case latency low |
| Fallback never receives raw unsafe text | Only the block reason and circuit ID are sent to the fallback model, preventing re-injection through the fallback path itself |

## Performance

Measured on an Intel i5-1235U / 8GB RAM (3.8GB available under WSL2), CPU-only,
no GPU. Numbers include HTTP round-trip overhead — not isolated in-process
timing, but representative of what a real client observes.

| Layer | Cold start | Warm steady-state |
|---|---|---|
| Fast filter (regex match) | N/A (no model loading required) | ~15-17ms |
| Deep classifier (DeBERTa on CPU) | ~5.2s (one-time, first inference only) | ~420-460ms |

The deep classifier cold start happens once per process lifetime on the first
inference call — not per-request. Subsequent requests use the warmed model.

The two-tier design keeps average-case latency practical: the fast filter
handles the majority of obvious jailbreak attempts and short-circuits before
the deep classifier runs. The ~450ms ML inference is reserved for the subset
of requests where regex alone isn't sufficient.

## Engineering Highlight: The HALF_OPEN Race Condition

During manual concurrent load testing (2-3 simultaneous requests timed to land
right as the OPEN-to-HALF_OPEN cooldown expired), multiple requests were all
observing "cooldown expired" and transitioning to HALF_OPEN concurrently — each
one believing it was the probe. The detector ran 2-3 times instead of exactly
once.

The fix: the OPEN->HALF_OPEN transition now uses a WATCH/MULTI optimistic
locking pattern. Each request WATCHes the state key, reads the current value,
and attempts an atomic MULTI transaction to claim the probe slot. If another
request already transitioned the state (WATCH detects the change), the loser
rejects immediately. This guarantees exactly one probe per cooldown window
under any level of concurrency.

This was not caught by the unit test suite, which tested `should_allow()` in
isolation. It surfaced through manual testing that exercised the full HTTP
stack with concurrent clients — a reminder that integration-level concurrency
bugs often hide behind correct unit-level logic.

## Setup

The verified working path is native Python on WSL (not Docker — see Known
Limitations).

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd Neural_circuit_breaker

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install CPU-only PyTorch first (saves ~2GB vs full CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Install and start Redis (WSL)
sudo apt update && sudo apt install -y redis-server
redis-server --daemonize yes

# 6. Copy and review environment config
cp .env.example .env

# 7. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On first startup, the deep classifier downloads `protectai/deberta-v3-base-prompt-injection-v2`
(~440MB from HuggingFace). Subsequent starts use the cached model. Set
`DEEP_CLASSIFIER_ENABLED=False` in `.env` to skip the model entirely.

```bash
# Run the test suite (no Redis or model download required — uses fakeredis and mocks)
pytest
```

`docker-compose.yml` is present in the repo but has not been re-validated
since Milestone 2, when the deep classifier and fallback routing dependencies
were added. The native path above is the primary working path.

## API Reference

### POST /v1/check

Evaluate text through the circuit breaker and detector pipeline.

**Request:**
```json
{"text": "What is machine learning?", "circuit_id": "demo"}
```

**Pass response** (both detectors cleared, circuit closed):
```json
{
  "status": "pass",
  "circuit_state": "CLOSED",
  "detail": null,
  "detector": null,
  "fallback": null
}
```

**Blocked response** (caught by deep classifier):
```json
{
  "status": "blocked",
  "circuit_state": "CLOSED",
  "detail": "Deep classifier flagged as prompt injection (score: 0.9200)",
  "detector": "deep_classifier",
  "fallback": {
    "message": "This request could not be processed safely. Please rephrase and try again.",
    "strategy": "static",
    "fallback_triggered": true
  }
}
```

**Blocked response** (circuit tripped open after repeated failures):
```json
{
  "status": "blocked",
  "circuit_state": "OPEN",
  "detail": "Circuit is OPEN — requests blocked during cooldown",
  "detector": null,
  "fallback": {
    "message": "This request could not be processed safely. Please rephrase and try again.",
    "strategy": "static",
    "fallback_triggered": true
  }
}
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `status` | `"pass"` \| `"blocked"` | Whether the request was allowed |
| `circuit_state` | string | Current circuit state: `CLOSED`, `OPEN`, or `HALF_OPEN` |
| `detail` | string \| null | Human-readable reason for the decision |
| `detector` | `"fast_filter"` \| `"deep_classifier"` \| null | Which layer made the block decision |
| `fallback` | object \| null | Safe fallback message (only present when `status` is `"blocked"`) |

### GET /v1/circuit/{circuit_id}/state

Debug endpoint showing current circuit state and failure count.

```bash
curl http://localhost:8000/v1/circuit/demo/state
```

```json
{"circuit_id": "demo", "state": "CLOSED", "failure_count": 0}
```

### GET /health

Liveness probe. Returns `{"status": "ok"}`.

## Configuration

All settings are loaded from `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `CIRCUIT_FAILURE_THRESHOLD` | `5` | Failures within the window before circuit opens |
| `CIRCUIT_WINDOW_SECONDS` | `60` | Sliding window for failure count |
| `CIRCUIT_COOLDOWN_SECONDS` | `30` | Seconds before OPEN transitions to HALF_OPEN |
| `DEEP_CLASSIFIER_ENABLED` | `True` | Enable ML classifier (`False` for fast-filter-only) |
| `DEEP_CLASSIFIER_MODEL` | `protectai/deberta-v3-base-prompt-injection-v2` | HuggingFace model ID |
| `DEEP_CLASSIFIER_THRESHOLD` | `0.5` | Score above this = unsafe |
| `FALLBACK_STRATEGY` | `static` | `static` (fixed message) or `model` (Groq/OpenRouter) |
| `FALLBACK_STATIC_MESSAGE` | _(safe default)_ | Message returned when strategy is `static` |
| `FALLBACK_MODEL_PROVIDER` | `groq` | `groq` or `openrouter` (only for `model` strategy) |
| `FALLBACK_MODEL_NAME` | `llama-3.1-8b-instant` | Model to call for the `model` strategy |
| `FALLBACK_MODEL_API_KEY` | _(empty)_ | API key for the model provider |

## Known Limitations / Path to Production

- **Deep classifier latency on CPU (~450ms)** exceeds a production <50ms
  budget. GPU inference (same DeBERTa model on CUDA — estimated ~30-50ms based
  on community benchmarks for this model class, not verified on this project's
  hardware) or a hosted
  inference endpoint (HuggingFace Inference API, Triton server) would close
  this gap. The two-tier design already mitigates average-case impact.

- **`FALLBACK_STRATEGY=model`** is implemented and unit-tested against mocked
  API responses, but was not verified against live Groq/OpenRouter calls during
  development (no API key was available). The default `static` strategy is
  fully verified. This is a straightforward integration test away from
  production readiness.

- **Docker Compose path** has not been re-validated since Milestone 2. The
  addition of the deep classifier (torch, transformers) and fallback routing
  (httpx calls) changed the dependency graph. The native WSL path is the
  primary working path until Docker is re-verified.

- **Single-instance Redis** with no clustering or HA. Fine for a portfolio
  demo or single-server deployment. Production multi-region would need
  Redis Sentinel or a managed service.

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| API framework | FastAPI | Async-native, Pydantic validation built in, minimal boilerplate |
| State store | Redis (async) | Atomic WATCH/MULTI for concurrency-safe state transitions; survives process restarts |
| Fast filter | Regex (20 patterns) | <15ms, zero dependencies, catches obvious patterns; short-circuits before ML |
| Deep classifier | `protectai/deberta-v3-base-prompt-injection-v2` | Fine-tuned specifically for prompt injection detection; runs on CPU |
| ML runtime | PyTorch + HuggingFace Transformers | Industry-standard; CPU-only install for development machines without GPU |
| Fallback routing | httpx (Groq/OpenRouter API) | 3s timeout, auto-degrades to static on failure |
| Testing | pytest + fakeredis + httpx | No real Redis or model download needed; mocked classifiers for fast CI |
| Config | pydantic-settings + .env | Type-safe, IDE-friendly, works with Docker and native setups |

## Test Suite

53 tests covering circuit breaker state transitions, detector accuracy,
concurrency safety, API integration, fallback routing, and regression guards.

```bash
pytest                    # run all tests
pytest tests/test_circuit_breaker.py   # state machine unit tests
pytest tests/test_integration.py       # full API integration tests
```

Tests use `fakeredis` (in-memory Redis mock) and stubbed ML pipelines — no
Redis server or model download required.
