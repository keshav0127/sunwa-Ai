"""Tests: Groq LLM client — missing key, auth error handling, no real API calls."""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── API Status ───────────────────────────────────────────────────────────────

def test_api_status_no_key():
    with patch("dotenv.load_dotenv"):
        with patch.dict(os.environ, {}, clear=True):
            # Remove all API key env vars
            for k in ["GROQ_API", "GROQ_API_KEY", "GROK_API"]:
                os.environ.pop(k, None)
            import importlib
            import engine.llm_client as llm
            importlib.reload(llm)
            status = llm.get_api_status()
            assert status["key_present"] is False


def test_api_status_with_key():
    import importlib
    import engine.llm_client as llm
    with patch.dict(os.environ, {"GROQ_API": "gsk_testkey123"}, clear=False):
        importlib.reload(llm)
        status = llm.get_api_status()
        assert status["key_present"] is True


# ── Is Auth Error ────────────────────────────────────────────────────────────

def test_is_auth_error_401():
    import engine.llm_client as llm
    exc = Exception("invalid api key provided (HTTP 401)")
    assert llm._is_auth_error(exc) is True


def test_is_auth_error_forbidden():
    import engine.llm_client as llm
    exc = Exception("403 Forbidden — authentication failed")
    assert llm._is_auth_error(exc) is True


def test_is_auth_error_network():
    import engine.llm_client as llm
    exc = Exception("Connection timeout after 30s")
    assert llm._is_auth_error(exc) is False


# ── resolve_location_with_llm — No API Calls in Tests ──────────────────────

def test_no_llm_call_when_no_key():
    import importlib
    import engine.llm_client as llm
    importlib.reload(llm)
    for k in ["GROQ_API", "GROQ_API_KEY", "GROK_API"]:
        os.environ.pop(k, None)
    result = llm.resolve_location_with_llm(
        complaint_text="SIES College ke samne khadda",
        region="Nerul",
        candidates=["SIES Graduate School of Technology"],
    )
    assert result is None


def test_llm_call_with_mocked_client():
    import importlib
    import engine.llm_client as llm
    importlib.reload(llm)
    # Reset disabled state
    llm._API_DISABLED = False

    with patch.dict(os.environ, {"GROQ_API": "gsk_fakekey"}):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "SIES Graduate School of Technology"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("engine.llm_client.Groq", return_value=mock_client):
            result = llm.resolve_location_with_llm(
                complaint_text="SIES College ke samne khadda hai",
                region="Nerul",
                candidates=["SIES Graduate School of Technology", "Nerul Railway Station"],
            )
        assert result == "SIES Graduate School of Technology"


def test_llm_returns_none_on_none_response():
    import importlib
    import engine.llm_client as llm
    importlib.reload(llm)
    llm._API_DISABLED = False

    with patch.dict(os.environ, {"GROQ_API": "gsk_fakekey"}):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "NONE"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("engine.llm_client.Groq", return_value=mock_client):
            result = llm.resolve_location_with_llm(
                complaint_text="some vague text",
                region="Nerul",
                candidates=["SIES Graduate School of Technology"],
            )
        assert result is None


def test_llm_disabled_after_auth_error():
    import importlib
    import engine.llm_client as llm
    importlib.reload(llm)
    llm._API_DISABLED = False

    with patch.dict(os.environ, {"GROQ_API": "gsk_badkey"}):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("incorrect api key provided")

        with patch("engine.llm_client.Groq", return_value=mock_client):
            result = llm.resolve_location_with_llm(
                complaint_text="SIES College ke pass khadda",
                region="Nerul",
                candidates=["SIES Graduate School of Technology"],
            )
        assert result is None
        assert llm._API_DISABLED is True


def test_no_retry_after_disabled():
    """Once disabled, subsequent calls must return None immediately without calling Groq."""
    import importlib
    import engine.llm_client as llm
    importlib.reload(llm)
    llm._API_DISABLED = True

    call_count = []
    def counting_llm(*args, **kwargs):
        call_count.append(1)
        return MagicMock()

    with patch.dict(os.environ, {"GROQ_API": "gsk_key"}):
        with patch("engine.llm_client.Groq", side_effect=counting_llm):
            result = llm.resolve_location_with_llm("test", "Nerul", ["SIES College"])
    assert result is None
    assert len(call_count) == 0, "Groq should not be instantiated when API is disabled"
