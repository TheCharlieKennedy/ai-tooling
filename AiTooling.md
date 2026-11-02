# ai-tooling

A provider-neutral Python library for calling LLMs from application code. It abstracts over provider SDKs, handles auth and transient failures, and exposes a small set of composable workflows — so callers never touch raw API calls.

Designed to be imported as an optional dependency. If it's not installed, callers fall back to stubs automatically.

---

## Design goals

- **Provider-agnostic** — swap Anthropic, OpenAI-compatible, or the Claude CLI without changing call sites
- **Composable** — pick the workflow that fits the task: single-shot analysis, multi-turn chat, system prompts, tool use
- **Optional** — callers install it only when they need LLM features; absence is handled gracefully
- **No framework lock-in** — no global state, no config files, no decorators; just functions and dataclasses

---

## Structure

```
src/ai_tooling/
├── protocols.py          # LlmClient — the single contract everything depends on
├── config.py             # ProviderConfig — typed config with env-var overrides
├── providers.py          # AnthropicProvider, OpenAICompatibleProvider, ClaudeCliProvider
├── session.py            # create_session() — factory; the main entry point
├── response.py           # extract_json(), serialize_bundle() — response utilities
└── workflows/
    ├── analyze.py        # run_analysis()  — prompt + evidence bundle → response
    ├── chat.py           # ChatSession     — multi-turn conversation with history
    ├── prompted.py       # run_prompted()  — system prompt + user message
    └── tools.py          # ToolDefinition, ToolRunner — structured tool dispatch
```

---

## Core concepts

### `LlmClient` protocol (`protocols.py`)

Every provider and stub satisfies this two-method contract:

```python
class LlmClient(Protocol):
    @property
    def available(self) -> bool: ...          # False → backend not configured
    def call(self, prompt: str, *, max_tokens: int = 1024) -> str: ...
```

Callers check `available` before calling. If it's `False`, skip gracefully — never raise.

### `ProviderConfig` (`config.py`)

Typed configuration for a provider session. Defaults to Anthropic with `ANTHROPIC_API_KEY`. Override at runtime via env vars:

| Env var | Effect |
|---|---|
| `AI_TOOLING_PROVIDER` | `anthropic` / `openai` / `claude-cli` |
| `AI_TOOLING_MODEL` | model string |
| `AI_TOOLING_BASE_URL` | custom base URL (for proxies / local models) |

`config.resolve()` applies these overrides and returns a new config — the original is not mutated.

### `create_session()` (`session.py`)

The only entry point you normally need. Pass a `ProviderConfig` or nothing:

```python
from ai_tooling.session import create_session

client = create_session()  # Anthropic, reads ANTHROPIC_API_KEY from env
```

Returns an `LlmClient`. If the API key is absent, `client.available` is `False` — no exception.

### Providers (`providers.py`)

Three implementations, all satisfying `LlmClient`:

| Class | Backend | SDK required |
|---|---|---|
| `AnthropicProvider` | Anthropic Messages API | `anthropic` |
| `OpenAICompatibleProvider` | OpenAI / any compatible API | `openai` |
| `ClaudeCliProvider` | `claude` CLI on PATH | none |

All providers:
- Set `available = False` silently when the SDK or key is missing
- Raise `ProviderUnavailableError` (not a generic exception) if called while unavailable
- Retry transient errors (429, 503, 529, timeout, connection reset) with exponential backoff, up to 3 attempts

---

## Workflows

Workflows are thin functions that combine a client with a calling pattern. Import only what you need.

### `run_analysis` — prompt + structured evidence

```python
from ai_tooling.workflows.analyze import run_analysis

result = run_analysis(client, prompt, evidence_list)
```

Serializes `evidence_list` (a `list[dict]`) to JSON, appends it to the prompt, and calls `client.call()`. Truncates the bundle if it would exceed the byte limit. This is the pattern for any rule that collects structured evidence and asks the LLM to judge it.

### `ChatSession` — multi-turn conversation

```python
from ai_tooling.workflows.chat import ChatSession

session = ChatSession(client)
reply1 = session.send("What's the capital of France?")
reply2 = session.send("And its population?")
session.reset()
```

Maintains history in memory. Each `send()` renders the full history into a single prompt (since `LlmClient.call()` is stateless).

### `run_prompted` — system prompt + user message

```python
from ai_tooling.workflows.prompted import run_prompted

result = run_prompted(client, system_prompt="You are a code reviewer.", user_message=code)
```

Combines system and user text into one call. For single-shot tasks with a persona or instruction set.

### `ToolRunner` — structured tool dispatch

```python
from ai_tooling.workflows.tools import ToolDefinition, ToolRunner

runner = ToolRunner()
runner.register(ToolDefinition(name="search", description="...", parameters={...}, handler=do_search))
runner.dispatch("search", {"query": "foo"})
```

Base implementation for LLM-driven tool use. Subclass and override `_execute_call()` for provider-native tool APIs (e.g. Anthropic's `tool_use` content blocks).

---

## Installation

```bash
# Core + Anthropic backend
pip install ai-tooling[anthropic]

# Core + OpenAI-compatible backend
pip install ai-tooling[openai]

# All backends
pip install ai-tooling[all]
```

The core package (`pip install ai-tooling`) installs without any LLM SDK — useful for the Claude CLI backend or for callers that implement their own provider.

---

## How nfr-review uses it

nfr-review imports ai-tooling as an optional dependency. The integration lives entirely in one file: `src/nfr_review/llm_integration.py`.

**Install with LLM support:**
```bash
pip install nfr-review[llm]   # pulls in ai-tooling[anthropic]
```

**Without it**, `llm_integration.py` falls back to inert stubs — `extract_json`, `serialize_bundle`, and `run_analysis` are re-implemented locally so every import still resolves. Band 2 rules that need an LLM will check `client.available` and skip with a recorded reason rather than failing.

### The integration boundary

`llm_integration.py` maps nfr-review's `LlmConfig` onto ai-tooling's `ProviderConfig` and calls `create_session()`:

```python
from nfr_review.llm_integration import create_llm_client

client = create_llm_client(config.llm)   # LlmConfig → ProviderConfig → LlmClient
```

### A Band 2 rule end-to-end

```python
# src/nfr_review/rules/pii_logging.py (simplified)

from nfr_review.llm_integration import ProviderUnavailableError, run_analysis, extract_json
from nfr_review.protocols import LlmClient

class PiiInLogStatementsRule:
    def __init__(self, llm_client: LlmClient) -> None:
        self._llm = llm_client

    def evaluate(self, evidence, config):
        if not self._llm.available:
            return skip("LLM not available")

        log_statements = [e for e in evidence if e["kind"] == "log_statement"]
        if not log_statements:
            return skip("no log statements found")

        prompt = (
            "Review each log statement below. For each one, return JSON: "
            '[{"index": N, "is_pii": true/false, "reason": "..."}]'
        )
        raw = run_analysis(self._llm, prompt, log_statements)
        verdicts = extract_json(raw, expect="array") or []

        findings = []
        for v in verdicts:
            if v.get("is_pii"):
                stmt = log_statements[v["index"]]
                findings.append(Finding(
                    rule_id="pii-in-log-statements",
                    rag="red",
                    confidence=0.85,
                    message=v["reason"],
                    location=stmt["file"],
                ))
        return findings
```

The rule never calls `client.call()` directly. It uses `run_analysis()` to handle bundling, and `extract_json()` to parse the response robustly regardless of how the LLM formats it.

---

## Adding a new workflow

1. Create `src/ai_tooling/workflows/my_workflow.py`
2. Accept an `LlmClient` as a parameter — do not import a specific provider
3. Use `client.call()` for the actual LLM call
4. Export from `workflows/__init__.py`

The `LlmClient` protocol is the only coupling point. Your workflow works against any backend automatically.
