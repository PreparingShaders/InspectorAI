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


def test_gemini_models():
    print("\n[2/2] ТЕСТ: Список Gemini...")

    # Твой список для проверки
    GEMINI_MODELS = [
        "gemini-2.5-flash",
        "gemini-3.1-flash-image-preview",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash"
    ]

    for model_id in GEMINI_MODELS:
        print(f"--- Проверяю: {model_id} ---")
        try:
            # Важно: добавляем префикс models/, если его нет,
            # так как SDK иногда требует полный путь
            full_path = f"models/{model_id}" if not model_id.startswith("models/") else model_id

            response = gemini_client.models.generate_content(
                model=full_path,
                contents="Привет! Подтверди свою готовность фразой 'Модель активна'."
            )
            print(f"✅ УСПЕХ! Ответ: {response.text.strip()}")
        except Exception as e:
            # Если 404 — значит Google еще не выкатил это имя в общий доступ
            # Если 403 — проблема с ключом или регионом
            print(f"❌ ОШИБКА для {model_id}: {e}")

# ─── ЗАПУСК ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Запуск тестов через прокси: {WORKER_URL}")
    print("=" * 50)

    test_openrouter()
    test_gemini_models()

    print("\n" + "=" * 50)
    print("Тестирование завершено.")
