import math
import time

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_EMBED_MODEL,
    GEMINI_INPUT_PRICE_PER_1M,
    GEMINI_OUTPUT_PRICE_PER_1M,
)

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add your Gemini API key to the .env file."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


def _normalize(values):
    """Normalize an embedding vector so cosine similarity works well."""
    norm = math.sqrt(sum(float(x) * float(x) for x in values))

    if norm == 0:
        return [0.0 for _ in values]

    return [float(x) / norm for x in values]


def embed_text(text: str, task_type: str):
    """
    Create Gemini embedding with simple retry handling.
    """

    max_retries = 5

    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=GEMINI_EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type
                ),
            )

            vector = result.embeddings[0].values

            return _normalize(vector)

        except Exception as exc:
            error_text = str(exc)

            if "429" not in error_text:
                raise

            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt

            print(
                f"Gemini rate limit reached. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

def generate_answer(prompt: str):
    """Generate answer using Gemini and track usage."""

    start = time.perf_counter()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    elapsed_ms = round(
        (time.perf_counter() - start) * 1000, 2
    )

    answer = (response.text or "").strip()

    usage = getattr(response, "usage_metadata", None)

    input_tokens = int(
        getattr(usage, "prompt_token_count", 0) or 0
    ) if usage else 0

    output_tokens = int(
        getattr(usage, "candidates_token_count", 0) or 0
    ) if usage else 0

    total_tokens = int(
        getattr(
            usage,
            "total_token_count",
            input_tokens + output_tokens
        ) or 0
    ) if usage else input_tokens + output_tokens

    estimated_cost = (
        (input_tokens / 1_000_000)
        * GEMINI_INPUT_PRICE_PER_1M
        +
        (output_tokens / 1_000_000)
        * GEMINI_OUTPUT_PRICE_PER_1M
    )

    return {
        "answer": answer,
        "latency_ms": elapsed_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_api_cost_usd": round(
            estimated_cost, 8
        ),
    }