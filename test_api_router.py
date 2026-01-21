import os
import openai
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Получаем ключ
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("❌ ОШИБКА: Переменная OPEN_ROUTER_API_KEY не найдена в .env файле!")
    # exit()  # можно закомментировать, если хочешь продолжить без OpenRouter

# ── DeepSeek ─────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API")

if not DEEPSEEK_API_KEY:
    print("❌ ОШИБКА: Переменная DEEPSEEK_API не найдена в .env файле!")
    # exit()  # можно закомментировать, если хочешь продолжить без DeepSeek

# Клиент для OpenRouter
openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
) if OPENROUTER_API_KEY else None

# Клиент для DeepSeek
deepseek_client = openai.OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
) if DEEPSEEK_API_KEY else None


OPENROUTER_MODELS = [
    "xiaomi/mimo-v2-flash:free",
    "allenai/molmo-2-8b:free",
    "google/gemma-3-27b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "mistralai/devstral-2512:free",
    "tngtech/deepseek-r1t2-chimera:free",
    'liquid/lfm-2.5-1.2b-thinking:free',
    'qwen/qwen3-next-80b-a3b-instruct:free'
]

DEEPSEEK_MODELS = [
    "deepseek-chat",          # максимум бесплатных токенов + высокая скорость
    "deepseek-reasoner",      # почти такой же лимит, отличный reasoning
    "deepseek-coder",         # для задач с кодом
    "deepseek-coder-v2-lite", # лёгкая версия (если нужен минимальный расход)
]
ALL_MODELS = {
    "OpenRouter": (openrouter_client, OPENROUTER_MODELS),
    "DeepSeek":   (deepseek_client,   DEEPSEEK_MODELS),
}


def ping_model(client, model_name, provider_name=""):
    if not client:
        print(f"[{provider_name}] Клиент не инициализирован — пропускаем")
        return

    print(f"Проверяю: {model_name} ({provider_name}) ...", end=" ", flush=True)
    try:
        # Для OpenRouter нужны дополнительные заголовки
        extra_headers = None
        if provider_name == "OpenRouter":
            extra_headers = {
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "InspectorAI_Test",
            }

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Сколько будет 17 + 25? Ответь только числом."}
            ],
            max_tokens=10,
            temperature=0.0,
            extra_headers=extra_headers,
            timeout=15,
        )

        answer = response.choices[0].message.content.strip()
        print(f"✅ OK   →   {answer}")

    except Exception as e:
        print(f"❌ Ошибка: {str(e)[:120]}")


def ping_all():
    print("=== Проверка доступности моделей ===\n")

    for provider, (client, models) in ALL_MODELS.items():
        if not models or not client:
            print(f"→ {provider}: нет моделей или клиента — пропускаем\n")
            continue

        print(f"┌─ {provider} ({len(models)} моделей) ────────")
        for model in models:
            ping_model(client, model, provider)
        print("└───────────────────────────────────────\n")


if __name__ == "__main__":
    ping_all()