import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- НАСТРОЙКИ ---
load_dotenv()

WORKER_URL = os.getenv("WORKER_URL", "https://inspectorgpt.classname1984.workers.dev").rstrip('/')
OR_KEY = os.getenv("OPEN_ROUTER_API_KEY")

client = AsyncOpenAI(
    api_key=OR_KEY,
    base_url=f"{WORKER_URL}/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/Aleksey/InspectorAI",
        "X-Title": "InspectorAI_Debug"
    }
)


async def check_model(model_id, semaphore):
    """Проверяет модель с ограничением параллельных запросов."""
    async with semaphore:  # Ограничиваем, чтобы не забанили за спам
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                timeout=15.0
            )
            if response and response.choices:
                return model_id, True
        except Exception:
            pass
        return model_id, False


async def main():
    print(f"Запрос бесплатных моделей OpenRouter через: {WORKER_URL}\n")

    try:
        # 1. Получаем список моделей
        response = await client.models.list()

        if not response or not hasattr(response, 'data'):
            print("❌ Ошибка: API вернул пустой список или некорректный формат.")
            return

        # 2. Фильтруем ID моделей
        free_model_ids = [m.id for m in response.data if ":free" in m.id]

        if not free_model_ids:
            print("❌ Бесплатные модели не найдены.")
            return

        print(f"Найдено {len(free_model_ids)} моделей. Начинаю проверку...")

        # 3. Используем Semaphore, чтобы проверять по 5 моделей за раз (защита от 429)
        semaphore = asyncio.Semaphore(5)
        tasks = [check_model(m_id, semaphore) for m_id in free_model_ids]

        results = await asyncio.gather(*tasks)

        # 4. Итог
        active_models = [m_id for m_id, status in results if status]

        print("\n" + "=" * 60)
        print(f"✅ РАБОЧИЕ МОДЕЛИ ({len(active_models)} из {len(free_model_ids)}):")
        print("=" * 60)

        if active_models:
            for i, model in enumerate(sorted(active_models), 1):
                print(f"{i:2}. ⭐ {model}")
        else:
            print("Рабочих моделей не обнаружено.")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())