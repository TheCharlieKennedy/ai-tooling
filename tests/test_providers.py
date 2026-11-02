"""Tests for ai_tooling providers — backends, retry, factory."""

from __future__ import annotations

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

import ai_tooling.providers as _providers_mod
from ai_tooling.config import ProviderConfig
from ai_tooling.providers import (
    AnthropicProvider,
    ClaudeCliProvider,
    OpenAICompatibleProvider,
    ProviderUnavailableError,
    _is_transient,
    _retry_with_backoff,
)
from ai_tooling.session import create_session


# ---------------------------------------------------------------------------
# _is_transient
# ---------------------------------------------------------------------------


class TestIsTransient:
    def test_429_is_transient(self) -> None:
        exc = Exception("rate limit")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert _is_transient(exc) is True

    def test_503_is_transient(self) -> None:
        exc = Exception("service unavailable")
        exc.status_code = 503  # type: ignore[attr-defined]
        assert _is_transient(exc) is True

    def test_529_is_transient(self) -> None:
        exc = Exception("overloaded")
        exc.status_code = 529  # type: ignore[attr-defined]
        assert _is_transient(exc) is True

    def test_400_is_not_transient(self) -> None:
        exc = Exception("bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        assert _is_transient(exc) is False

    def test_timeout_error_is_transient(self) -> None:
        assert _is_transient(TimeoutError("timed out")) is True

    def test_connection_error_is_transient(self) -> None:
        assert _is_transient(ConnectionError("refused")) is True

    def test_class_name_ratelimit_is_transient(self) -> None:
        class RateLimitError(Exception):
            pass

        assert _is_transient(RateLimitError("too fast")) is True

    def test_generic_exception_not_transient(self) -> None:
        assert _is_transient(ValueError("bad value")) is False


# ---------------------------------------------------------------------------
# _retry_with_backoff
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    def test_succeeds_on_first_try(self) -> None:
        assert _retry_with_backoff(lambda: "ok") == "ok"

    def test_retries_on_transient_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_providers_mod.time, "sleep", lambda _: None)
        calls = []

        def flaky() -> str:
            calls.append(1)
            if len(calls) < 3:
                exc = Exception("overloaded")
                exc.status_code = 503  # type: ignore[attr-defined]
                raise exc
            return "recovered"

        assert _retry_with_backoff(flaky) == "recovered"
        assert len(calls) == 3

    def test_raises_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_providers_mod.time, "sleep", lambda _: None)

        def always_fail() -> str:
            exc = Exception("overloaded")
            exc.status_code = 503  # type: ignore[attr-defined]
            raise exc

        with pytest.raises(Exception, match="overloaded"):
            _retry_with_backoff(always_fail)

    def test_does_not_retry_non_transient(self) -> None:
        calls = []

        def permanent_fail() -> str:
            calls.append(1)
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            _retry_with_backoff(permanent_fail)
        assert len(calls) == 1

    def test_does_not_retry_provider_unavailable(self) -> None:
        calls = []

        def unavailable() -> str:
            calls.append(1)
            raise ProviderUnavailableError("no backend")

        with pytest.raises(ProviderUnavailableError):
            _retry_with_backoff(unavailable)
        assert len(calls) == 1

    def test_exponential_delays(self, monkeypatch: pytest.MonkeyPatch) -> None:
        delays: list[float] = []
        monkeypatch.setattr(_providers_mod.time, "sleep", lambda d: delays.append(d))
        call_count = 0

        def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                exc = Exception("overloaded")
                exc.status_code = 503  # type: ignore[attr-defined]
                raise exc
            return "ok"

        _retry_with_backoff(fail_twice)
        assert len(delays) == 2
        assert delays[0] == pytest.approx(1.0)
        assert delays[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def test_unavailable_when_sdk_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_providers_mod, "_anthropic_sdk", None)
        client = AnthropicProvider(model="test", api_key="key", base_url=None, timeout_seconds=120)
        assert client.available is False

    def test_unavailable_when_no_key(self) -> None:
        client = AnthropicProvider(model="test", api_key="", base_url=None, timeout_seconds=120)
        assert client.available is False

    def test_raises_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_providers_mod, "_anthropic_sdk", None)
        client = AnthropicProvider(model="test", api_key="key", base_url=None, timeout_seconds=120)
        with pytest.raises(ProviderUnavailableError):
            client.call("prompt")

    @patch("ai_tooling.providers._anthropic_sdk", create=True)
    def test_call_returns_text(self, mock_sdk: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_providers_mod.time, "sleep", lambda _: None)
        mock_block = MagicMock()
        mock_block.text = "result text"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_instance = MagicMock()
        mock_instance.messages.create.return_value = mock_response
        mock_sdk.Anthropic.return_value = mock_instance

        client = AnthropicProvider(model="claude-test", api_key="key", base_url=None, timeout_seconds=120)
        assert client.call("prompt") == "result text"
        mock_instance.messages.create.assert_called_once_with(
            model="claude-test",
            max_tokens=1024,
            messages=[{"role": "user", "content": "prompt"}],
        )

    @patch("ai_tooling.providers._anthropic_sdk", create=True)
    def test_retries_transient_error(self, mock_sdk: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_providers_mod.time, "sleep", lambda _: None)
        mock_block = MagicMock()
        mock_block.text = "recovered"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        transient = Exception("overloaded")
        transient.status_code = 529  # type: ignore[attr-defined]
        mock_instance = MagicMock()
        mock_instance.messages.create.side_effect = [transient, mock_response]
        mock_sdk.Anthropic.return_value = mock_instance

        client = AnthropicProvider(model="test", api_key="key", base_url=None, timeout_seconds=120)
        assert client.call("prompt") == "recovered"
        assert mock_instance.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider
# ---------------------------------------------------------------------------


class TestOpenAICompatibleProvider:
    def test_unavailable_when_sdk_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_providers_mod, "_openai_sdk", None)
        client = OpenAICompatibleProvider(model="gpt-4o", api_key="key", base_url=None, timeout_seconds=120)
        assert client.available is False

    def test_raises_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_providers_mod, "_openai_sdk", None)
        client = OpenAICompatibleProvider(model="gpt-4o", api_key="key", base_url=None, timeout_seconds=120)
        with pytest.raises(ProviderUnavailableError):
            client.call("prompt")

    @patch("ai_tooling.providers._openai_sdk", create=True)
    def test_call_returns_text(self, mock_sdk: MagicMock) -> None:
        mock_message = MagicMock()
        mock_message.content = "OpenAI says hello"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_sdk.OpenAI.return_value = mock_instance

        client = OpenAICompatibleProvider(model="gpt-4o", api_key="key", base_url=None, timeout_seconds=120)
        assert client.call("prompt") == "OpenAI says hello"

    @patch("ai_tooling.providers._openai_sdk", create=True)
    def test_raises_on_null_content(self, mock_sdk: MagicMock) -> None:
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_sdk.OpenAI.return_value = mock_instance

        client = OpenAICompatibleProvider(model="gpt-4o", api_key="key", base_url=None, timeout_seconds=120)
        with pytest.raises(TypeError, match="no content"):
            client.call("prompt")


# ---------------------------------------------------------------------------
# ClaudeCliProvider
# ---------------------------------------------------------------------------


class TestClaudeCliProvider:
    def test_unavailable_when_missing(self) -> None:
        with patch("ai_tooling.providers.shutil.which", return_value=None):
            client = ClaudeCliProvider(timeout_seconds=120)
        assert client.available is False

    def test_available_when_found(self) -> None:
        with patch("ai_tooling.providers.shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCliProvider(timeout_seconds=120)
        assert client.available is True

    def test_returns_stdout(self) -> None:
        with patch("ai_tooling.providers.shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCliProvider(timeout_seconds=120)
        mock_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="  result\n", stderr="")
        with patch("ai_tooling.providers.subprocess.run", return_value=mock_result):
            assert client.call("prompt") == "result"

    def test_raises_on_nonzero_exit(self) -> None:
        with patch("ai_tooling.providers.shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCliProvider(timeout_seconds=120)
        mock_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="broke")
        with patch("ai_tooling.providers.subprocess.run", return_value=mock_result):
            with pytest.raises(ProviderUnavailableError, match="exited 1"):
                client.call("prompt")

    def test_raises_on_timeout(self) -> None:
        with patch("ai_tooling.providers.shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCliProvider(timeout_seconds=120)
        with patch(
            "ai_tooling.providers.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=120),
        ):
            with pytest.raises(ProviderUnavailableError, match="timed out"):
                client.call("prompt")


# ---------------------------------------------------------------------------
# create_session factory
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_default_creates_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AI_TOOLING_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = create_session()
        assert isinstance(client, AnthropicProvider)

    def test_openai_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = ProviderConfig(provider="openai", model="gpt-4o", api_key_env_var="OPENAI_API_KEY")
        client = create_session(cfg)
        assert isinstance(client, OpenAICompatibleProvider)

    def test_claude_cli_config(self) -> None:
        cfg = ProviderConfig(provider="claude-cli")
        with patch("ai_tooling.providers.shutil.which", return_value="/usr/bin/claude"):
            client = create_session(cfg)
        assert isinstance(client, ClaudeCliProvider)
        assert client.available is True
