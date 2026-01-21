import os
import openai
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Получаем ключ
api_key = os.getenv("OPEN_ROUTER_API_KEY")

# Проверка, что ключ вообще прочитан
if not api_key:
    print("❌ ОШИБКА: Переменная OPENROUTER_API_KEY не найдена в .env файле!")
    exit()

# Настраиваем клиент
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

OPENROUTER_MODELS = [
    "xiaomi/mimo-v2-flash:free",
    "allenai/molmo-2-8b:free",
    "google/gemma-3-27b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "mistralai/devstral-2512:free",
    "tngtech/deepseek-r1t2-chimera:free",
    'liquid/lfm-2.5-1.2b-thinking:free',
    'qwen/qwen3-next-80b-a3b-instruct:free',
]


def ping_models():
    print(f"🔑 Ключ подгружен: {api_key[:8]}***")
    print(f"--- Начинаю проверку {len(OPENROUTER_MODELS)} моделей ---\n")

    for model in OPENROUTER_MODELS:
        print(f"Проверяю: {model}...", end=" ", flush=True)
        try:
            # Обязательные заголовки для OpenRouter
            extra_headers = {
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "InspectorAI_Test",
            }

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Какое сегодня число?"}],
                max_tokens=20,
                extra_headers=extra_headers,
                timeout=10  # Чтобы не висеть вечно, если модель перегружена
            )

            answer = response.choices[0].message.content.strip()
            print(f"✅ OK")
            print(f"   Ответ: {answer[:50]}...")

        except Exception as e:
            print(f"❌ Ошибка")
            print(f"   Причина: {e}")


if __name__ == "__main__":
    ping_models()