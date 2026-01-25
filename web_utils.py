import html
import asyncio
from ddgs import DDGS

TRUSTED_SITES = [
    "reuters.com", "apnews.com", "interfax.ru", "rbc.ru",
    "kommersant.ru", "tass.ru", "ria.ru", "provereno.media",
    "bbc.com/russian", "meduza.io", "vedomosti.ru", "ru.wikipedia.org"
]

async def get_web_context(query: str, period='w'):
    try:
        with DDGS() as ddgs:
            clean_query = query.replace('"', '').strip()

            # Делаем один широкий поиск по всем доверенным сразу (срез до 12)
            sites_filter = " OR ".join([f"site:{s}" for s in TRUSTED_SITES[:12]])
            full_query = f"{clean_query[:80]} ({sites_filter})"

            results = ddgs.text(full_query, region='ru-ru', timelimit=period, max_results=10)

            if not results:
                # Если в белом списке пусто, ищем везде
                results = ddgs.text(clean_query[:150], region='ru-ru', max_results=5)

            if not results: return None

            # --- ЛОГИКА РЕЙТИНГА ---
            found_on_sites = set()
            context_parts = []

            for r in results:
                href = r.get('href', '').lower()
                for site in TRUSTED_SITES[:12]:
                    if site in href:
                        found_on_sites.add(site)

                context_parts.append(f"ИСТОЧНИК: {r.get('title')}\nURL: {href}")

            # Формируем отчет для нейронки
            report = "ОТЧЕТ ПО МОНИТОРИНГУ СМИ:\n"
            for site in TRUSTED_SITES[:12]:
                status = "✅ ЕСТЬ ПУБЛИКАЦИЯ" if site in found_on_sites else "❌ НЕТ ДАННЫХ"
                report += f"- {site}: {status}\n"

            report += "\nДЕТАЛИ:\n" + "\n".join(context_parts[:5])
            return report

    except Exception as e:
        print(f"🌐 Ошибка: {e}")
        return None