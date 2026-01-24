import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types

load_dotenv() # Загружаем переменные из .env

WORKER_URL = "https://inspectorgpt.classname1984.workers.dev"

# 1. Клиент для OpenRouter
# Берем РЕАЛЬНЫЙ ключ из .env через os.getenv
or_client = OpenAI(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"), # Без кавычек!
    base_url=f"{WORKER_URL}/v1"
)

# 2. Клиент для Gemini
# Берем РЕАЛЬНЫЙ ключ из .env через os.getenv
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"), # Без кавычек!
    http_options=types.HttpOptions(base_url=WORKER_URL),
)

# ─── ФУНКЦИИ ТЕСТИРОВАНИЯ ──────────────────────────────────────────────────

def test_openrouter():
    print("\n[1/2] ТЕСТ: OpenRouter...")
    try:
        # Используем одну из бесплатных моделей
        model_name = "xiaomi/mimo-v2-flash:free"

        response = or_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Скажи 'OpenRouter работает', если ты меня слышишь."}],
            max_tokens=50
        )
        print("✅ УСПЕХ!")
        print(f"Ответ модели ({model_name}): {response.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"❌ ОШИБКА OpenRouter: {e}")


def test_gemini():
    print("\n[2/2] ТЕСТ: Gemini...")
    try:
        # Используем стандартную модель
        model_name = "models/gemini-2.5-flash"

        response = gemini_client.models.generate_content(
            model=model_name,
            contents="Скажи 'Gemini работает', если ты меня слышишь."
        )
        print("✅ УСПЕХ!")
        print(f"Ответ модели ({model_name}): {response.text.strip()}")
    except Exception as e:
        print(f"❌ ОШИБКА Gemini: {e}")


# ─── ЗАПУСК ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Запуск тестов через прокси: {WORKER_URL}")
    print("=" * 50)

    test_openrouter()
    test_gemini()

    print("\n" + "=" * 50)
    print("Тестирование завершено.")

    # OPENROUTER_MODELS = [
    #     "xiaomi/mimo-v2-flash:free",
    #     "allenai/molmo-2-8b:free",
    #     "google/gemma-3-27b-it:free",
    #     "nousresearch/hermes-3-llama-3.1-405b:free",
    #     "nvidia/nemotron-3-nano-30b-a3b:free",
    #     "mistralai/devstral-2512:free",
    #     "tngtech/deepseek-r1t2-chimera:free",
    #     'liquid/lfm-2.5-1.2b-thinking:free',
    #     'qwen/qwen3-next-80b-a3b-instruct:free',
    # ]
