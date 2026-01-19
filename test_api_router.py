import os
import asyncio
from dotenv import load_dotenv

from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Content

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

GEMINI_MODELS = [
    "models/gemini-3-flash-preview",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash-exp",
]

OPENROUTER_MODELS = [
    "xiaomi/mimo-v2-flash:free",
    "allenai/molmo-2-8b:free",
    "google/gemma-3-27b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "mistralai/devstral-2-2512:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-4-maverick:free",
    "qwen/qwen3-235b-a22b:free",
    "microsoft/phi-4:free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "deepseek/deepseek-v3-base:free",
    "xai/grok-3-mini:free",
]

# ─── Клиенты ────────────────────────────────────────────────────────────────
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        base_url="https://inspectorgpt.classname1984.workers.dev"
    )
)

or_client = OpenAI(
    api_key=OPEN_ROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

TEST_PROMPT = "Привет! Назови сегодняшнюю дату одним предложением."

SYSTEM = "Ты полезный ассистент. Отвечай кратко и по делу."


async def test_gemini_model(model_name: str):
    print(f"\n{'═' * 70}")
    print(f"Testing Gemini: {model_name}")
    print(f"{'─' * 70}")

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=[
                Content(role="model", parts=[types.Part(text=SYSTEM)]),
                Content(role="user", parts=[types.Part(text=TEST_PROMPT)])
            ],
            config=GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=80
            )
        )

        text = response.text.strip() if response.text else "[пустой ответ]"
        print("OK ✓")
        print(f"Ответ: {text[:120]}{'...' if len(text) > 120 else ''}")

    except Exception as e:
        print("✗ Ошибка:")
        print(str(e)[:300])


async def test_openrouter_model(model_name: str):
    print(f"\n{'═' * 70}")
    print(f"Testing OpenRouter: {model_name}")
    print(f"{'─' * 70}")

    try:
        response = or_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": TEST_PROMPT}
            ],
            temperature=0.0,
            max_tokens=80
        )

        text = response.choices[0].message.content.strip() if response.choices else "[пустой ответ]"
        print("OK ✓")
        print(f"Ответ: {text[:120]}{'...' if len(text) > 120 else ''}")

    except Exception as e:
        print("✗ Ошибка:")
        print(str(e)[:300])


async def main():
    print("Тестирование всех моделей...\n")

    print("GEMINI MODELS".center(70, "═"))
    for model in GEMINI_MODELS:
        await test_gemini_model(model)
        await asyncio.sleep(1.5)  # небольшая пауза, чтобы не нагружать

    print("\nOPENROUTER MODELS".center(70, "═"))
    for model in OPENROUTER_MODELS:
        await test_openrouter_model(model)
        await asyncio.sleep(2.0)  # OpenRouter иногда строже с рейт-лимитами


if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY не найден в .env")
        exit(1)
    if not OPEN_ROUTER_API_KEY:
        print("OPEN_ROUTER_API_KEY не найден в .env")
        exit(1)

    asyncio.run(main())