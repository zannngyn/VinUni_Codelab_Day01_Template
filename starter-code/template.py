"""Day 1 — compare inexpensive OpenAI, Google, and Anthropic models via OpenRouter.

Set OPENROUTER_API_KEY in the environment or in a .env file before calling an API.
Run ``python -m pytest tests/ -v`` for offline tests.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Allow ``python starter-code/template.py`` to use the already-installed project venv.
if __name__ == "__main__" and sys.prefix == sys.base_prefix:
    venv_python = Path(__file__).resolve().parent.parent / ".venv"
    venv_python /= "Scripts/python.exe" if os.name == "nt" else "bin/python"
    if venv_python.is_file():
        os.execv(str(venv_python), [str(venv_python), *sys.argv])

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Standard text rates shown on OpenRouter, USD per million tokens, November 2026.
# These are estimates; routing, caching, and other token types can change billed cost.
OPENAI_MODEL = "openai/gpt-4o-mini"
GEMINI_MODEL = "google/gemini-3.1-flash-lite"
ANTHROPIC_MODEL = "anthropic/claude-haiku-5.5"

PRICING_1M_TOKENS = {
    OPENAI_MODEL: {"input": 0.15, "output": 0.60},
    GEMINI_MODEL: {"input": 0.25, "output": 1.50},
    ANTHROPIC_MODEL: {"input": 0.10, "output": 0.50},
}


def _openrouter_client():
    """Create an OpenAI-compatible client using only the OpenRouter key."""
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY in your environment or .env file")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _call_model(
    prompt: str, model: str, temperature: float, top_p: float, max_tokens: int
) -> tuple[str, float, dict]:
    client = _openrouter_client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,
        extra_body={"usage": {"include": True}},
    )
    latency = time.perf_counter() - start
    if response.usage is None:
        raise RuntimeError("OpenRouter did not return token usage")
    return (
        response.choices[0].message.content or "",
        latency,
        {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call an OpenAI model through OpenRouter; return text, latency, and usage."""
    return _call_model(prompt, model, temperature, top_p, max_tokens)


def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call a Google model through OpenRouter; return text, latency, and usage."""
    return _call_model(prompt, model, temperature, top_p, max_tokens)


def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call an Anthropic model through OpenRouter; return text, latency, and usage."""
    return _call_model(prompt, model, temperature, top_p, max_tokens)


def compare_models(prompt: str) -> dict:
    """Compare one inexpensive model from each provider using estimated USD cost."""
    responses = {
        "openai": (OPENAI_MODEL, call_openai(prompt)),
        "gemini": (GEMINI_MODEL, call_gemini(prompt)),
        "anthropic": (ANTHROPIC_MODEL, call_anthropic(prompt)),
    }
    comparison = {}
    for name, (model, (response, latency, usage)) in responses.items():
        rates = PRICING_1M_TOKENS[model]
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        comparison[name] = {
            "response": response,
            "latency": latency,
            "cost": (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    return comparison


def streaming_chatbot() -> None:
    """Stream Gemini replies and retain the last three complete exchanges."""
    client = _openrouter_client()
    history = []
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in {"quit", "exit"}:
            return
        if not prompt:
            continue

        user_message = {"role": "user", "content": prompt}
        stream = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=history + [user_message],
            stream=True,
        )
        print("Gemini: ", end="", flush=True)
        response_parts = []
        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                print(text, end="", flush=True)
                response_parts.append(text)
        print()
        history.extend([user_message, {"role": "assistant", "content": "".join(response_parts)}])
        history = history[-6:]


def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Retry up to max_retries times with exponentially increasing delays."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2 ** attempt)


def batch_compare(prompts: list[str]) -> list[dict]:
    """Compare each prompt and include its original text in the result."""
    return [{**compare_models(prompt), "prompt": prompt} for prompt in prompts]


def format_comparison_table(results: list[dict]) -> str:
    """Return a Markdown table with a row for each prompt and model."""
    headers = "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Est. Cost (USD) |"
    separator = "| --- | --- | --- | ---: | ---: | ---: |"
    labels = {
        "openai": "GPT-4o mini",
        "gemini": "Gemini 3.1 Flash Lite",
        "anthropic": "Claude Haiku 5.5",
    }
    rows = []
    for result in results:
        prompt = str(result["prompt"]).replace("|", "\\|").replace("\n", " ")
        for key, label in labels.items():
            stats = result[key]
            response = str(stats["response"]).replace("|", "\\|").replace("\n", " ")
            if len(response) > 50:
                response = response[:47] + "..."
            rows.append(
                f"| {prompt} | {label} | {response} | {stats['latency']:.2f}s | "
                f"{stats['input_tokens']}/{stats['output_tokens']} | ${stats['cost']:.8f} |"
            )
    return "\n".join([headers, separator, *rows])


if __name__ == "__main__":
    try:
        results = compare_models("Hãy chào bằng tiếng Việt trong một câu ngắn.")
        print(format_comparison_table([{**results, "prompt": "Lời chào"}]))
        print("\nChatbot Gemini (gõ 'quit' để thoát):")
        streaming_chatbot()
    except ValueError as exc:
        print(exc)
