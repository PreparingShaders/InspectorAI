import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPEN_ROUTER_API_KEY")

if not api_key:
    print("❌ OPEN_ROUTER_API_KEY не найден в .env!")
    exit(1)

print("API-ключ найден, пробуем запрос...")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

try:
    response = client.chat.completions.create(
        model="nousresearch/hermes-3-llama-3.1-405b:free",   # бесплатная модель для теста
        messages=[
            {"role": "system", "content": "Ты полезный бот."},
            {"role": "user", "content": "Привет! Скажи мне шутку про программистов."}
        ],
        max_tokens=120,
        temperature=0.7,
    )

    print("\nУспех! Ответ модели:")
    print(response.choices[0].message.content.strip())

except Exception as e:
    print("\nОшибка:")
    print(str(e))