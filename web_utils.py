#web_utils
from ddgs import DDGS

TRUSTED_SITES = [
    "reuters.com", "apnews.com", "interfax.ru", "rbc.ru",
    "kommersant.ru", "tass.ru", "ria.ru", "provereno.media",
    "bbc.com/russian", "meduza.io", "vedomosti.ru", "ru.wikipedia.org"
]


async def get_web_context(query: str, period='w'):
    try:
        with DDGS() as ddgs:
            # 1. Сначала чистим запрос отдельно
            clean_q = query.replace('"', '').replace("'", "").strip()

            # 2. Формируем строку без сложной вложенности кавычек
            # Используем двойные кавычки снаружи, чтобы внутри спокойно писать текст
            refined_query = f'"{clean_q[:120]}" (фактчек OR проверка OR подробности OR разоблачение)'

            # 3. Поиск
            results = list(ddgs.text(refined_query, region='ru-ru', timelimit=period, max_results=8))

            if not results:
                results = list(ddgs.text(clean_q[:100], region='ru-ru', max_results=5))

            if not results:
                return None

            # --- Дальше твоя логика обработки результатов ---
            found_on_sites = set()
            context_parts = []

            for r in results:
                href = r.get('href', '').lower()
                # snippet может прийти в body или snippet
                snippet = r.get('body') or r.get('snippet') or ''
                title = r.get('title', '')

                for site in TRUSTED_SITES:
                    if site in href:
                        found_on_sites.add(site)

                context_parts.append(f"Заголовок: {title}\nСуть: {snippet}\nИсточник: {href}")

            trusted_report = ", ".join(found_on_sites) or "Доверенных источников не найдено"
            report = f"РЕЗУЛЬТАТЫ ПОИСКА ПО СМИ:\nУпомянуто в доверенных: {trusted_report}\n\nДАННЫЕ:\n" + "\n---\n".join(
                context_parts)
            return report

    except Exception as e:
        print(f"🌐 Ошибка в get_web_context: {e}")
        return None