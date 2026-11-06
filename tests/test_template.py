"""Offline tests for the Day 1 OpenRouter implementation."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


TEMPLATE_PATH = Path(__file__).parents[1] / "starter-code" / "template.py"
spec = importlib.util.spec_from_file_location("day1_template", TEMPLATE_PATH)
template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template)


def completion(text="answer", prompt_tokens=10, completion_tokens=20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


@pytest.mark.parametrize(
    "function,model",
    [
        ("call_openai", "openai/gpt-4o-mini"),
        ("call_gemini", "google/gemini-3.1-flash-lite"),
        ("call_anthropic", "anthropic/claude-haiku-5.5"),
    ],
)
def test_provider_calls_use_one_openrouter_key(function, model):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), patch("openai.OpenAI") as client_class:
        client_class.return_value.chat.completions.create.return_value = completion()
        text, latency, usage = getattr(template, function)(
            "Hello", temperature=0.4, top_p=0.8, max_tokens=42
        )

    assert text == "answer"
    assert latency >= 0
    assert usage == {"input_tokens": 10, "output_tokens": 20}
    assert client_class.call_args.kwargs == {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
    }
    call = client_class.return_value.chat.completions.create.call_args
    assert call.kwargs["model"] == model
    assert call.kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert call.kwargs["temperature"] == 0.4
    assert call.kwargs["top_p"] == 0.8
    assert call.kwargs["max_completion_tokens"] == 42


def test_missing_openrouter_key_fails_before_request():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), patch("openai.OpenAI") as client_class:
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            template.call_openai("Hello")
    client_class.assert_not_called()


def test_direct_system_python_run_uses_project_venv():
    env = {**os.environ, "OPENROUTER_API_KEY": ""}
    result = subprocess.run(
        [sys._base_executable, "-S", str(TEMPLATE_PATH)],
        cwd=TEMPLATE_PATH.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "Set OPENROUTER_API_KEY" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_compare_models_uses_three_providers_and_current_estimate_rates():
    usage = {"input_tokens": 10, "output_tokens": 20}
    with patch.object(template, "call_openai", return_value=("O", 0.1, usage)), \
         patch.object(template, "call_gemini", return_value=("G", 0.2, usage)), \
         patch.object(template, "call_anthropic", return_value=("A", 0.3, usage)):
        result = template.compare_models("Prompt")

    assert set(result) == {"openai", "gemini", "anthropic"}
    assert result["openai"]["cost"] == pytest.approx(0.0000135)
    assert result["gemini"]["cost"] == pytest.approx(0.0000325)
    assert result["anthropic"]["cost"] == pytest.approx(0.000011)
    assert result["anthropic"]["response"] == "A"


def test_streaming_chatbot_keeps_last_three_turns(capsys):
    inputs = iter(["Q1", "Q2", "Q3", "Q4", "Q5", "exit"])

    def stream_for_call(**kwargs):
        answer = "A" + kwargs["messages"][-1]["content"][1:]
        return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=answer))])]

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
         patch("openai.OpenAI") as client_class, \
         patch("builtins.input", side_effect=lambda _: next(inputs)):
        client_class.return_value.chat.completions.create.side_effect = stream_for_call
        template.streaming_chatbot()

    calls = client_class.return_value.chat.completions.create.call_args_list
    assert len(calls) == 5
    assert all(call.kwargs["stream"] is True for call in calls)
    assert [item["content"] for item in calls[-1].kwargs["messages"]] == [
        "Q2", "A2", "Q3", "A3", "Q4", "A4", "Q5"
    ]
    assert "A5" in capsys.readouterr().out


def test_retry_with_backoff_uses_exponential_delays():
    attempts = []

    def fail():
        attempts.append(1)
        raise RuntimeError("failure")

    with patch.object(template.time, "sleep") as sleep:
        with pytest.raises(RuntimeError, match="failure"):
            template.retry_with_backoff(fail, max_retries=3, base_delay=0.25)
    assert len(attempts) == 4
    assert [call.args[0] for call in sleep.call_args_list] == [0.25, 0.5, 1.0]


def test_batch_and_table_include_all_models():
    comparison = {
        key: {"response": key, "latency": 0.1, "cost": 0.00001, "input_tokens": 10, "output_tokens": 20}
        for key in ("openai", "gemini", "anthropic")
    }
    with patch.object(template, "compare_models", return_value=comparison):
        results = template.batch_compare(["Question"])
    assert results[0]["prompt"] == "Question"
    table = template.format_comparison_table(results)
    assert "Question" in table
    assert "GPT-4o mini" in table
    assert "Gemini 3.1 Flash Lite" in table
    assert "Claude Haiku 5.5" in table
