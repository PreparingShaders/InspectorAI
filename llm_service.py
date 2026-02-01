import re
import requests
from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Content
from collections import defaultdict

from config import (
    GEMINI_API_KEY, OPEN_ROUTER_API_KEY, WORKER_URL,
    SYSTEM_PROMPT_INSPECTOR, SYSTEM_PROMPT_CHAT, DEFAULT_OPENROUTER_MODELS, GEMINI_MODELS, API_TIMEOUT
)
from web_utils import get_web_context
from utils import format_to_html, get_model_short_name

# --- Инициализация клиентов ---
or_client = OpenAI(
    api_key=OPEN_ROUTER_API_KEY,
    base_url=f"{WORKER_URL}/v1",
    timeout=API_TIMEOUT
)

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(base_url=WORKER_URL, timeout=int(API_TIMEOUT * 1000)),
)

# --- Глобальные состояния моделей ---
current_free_or_models = []
GEMINI_MODEL_BY_ID = {str(i): m for i, m in enumerate(GEMINI_MODELS)}
OPENROUTER_MODEL_BY_ID = {}
chat_histories = defaultdict(list)
BLACKLISTED_MODELS = set()

def fetch_dynamic_models():
    url = "https://openrouter.ai/api/v1/models"
    try:
        headers = {"Authorization": f"Bearer {OPEN_ROUTER_API_KEY}"}
        response = requests.get(url, params={"order": "most-popular"}, headers=headers, timeout=10)
        if response.status_code == 200:
            all_models = response.json().get('data', [])
            processed = []
            for m in all_models:
                m_id = m.get('id', '')
                if ":free" in m_id and m_id not in BLACKLISTED_MODELS:
                    desc = m.get('description', '')
                    size_match = re.search(r'(\d+[.,]?\d*)\s*[Bb]', desc)
                    size_val = float(size_match.group(1).replace(',', '.')) if size_match else 0.1
                    processed.append({
                        'id': m_id,
                        'size': size_val,
                        'context': int(m.get('context_length', 0))
                    })
            sorted_models = sorted(processed, key=lambda x: (-x['size'], -x['context']))
            print(sorted_models)
            return [m['id'] for m in sorted_models[:15]]
    except Exception as e:
        print(f"⚠️ Ошибка обновления моделей: {e}")
    return None


# llm_service.py

current_free_or_models = []  # Изначально пустой или с дефолтами


def update_model_mappings():
    global current_free_or_models, OPENROUTER_MODEL_BY_ID

    new_models = fetch_dynamic_models()  # Твоя логика получения списка

    if new_models:
        current_free_or_models.clear()  # Очищаем старый список
        current_free_or_models.extend(new_models)  # Наполняем тот же самый объект

        # Обновляем маппинг ID
        OPENROUTER_MODEL_BY_ID.clear()
        for i, m in enumerate(current_free_or_models):
            OPENROUTER_MODEL_BY_ID[str(i + 100)] = m
# llm_service.py

# Добавляем mode="chat" в аргументы функции
async def process_llm(update, context, query, selected_model=None, selected_provider=None, thread_id=None, mode="chat"):
    chat_id = update.effective_chat.id

    system_instruction = SYSTEM_PROMPT_INSPECTOR if mode == "inspector" else SYSTEM_PROMPT_CHAT

    status_msg = await context.bot.send_message(chat_id, "⚡ Работаю...", message_thread_id=thread_id)

    # 2. Логика веб-поиска теперь срабатывает ТОЛЬКО если mode == "inspector"
    final_query = query
    if mode == "inspector":
        await context.bot.edit_message_text("🔍 Инспектор ищет доказательства...", chat_id, status_msg.message_id)
        from web_utils import get_web_context

        # Если в запросе есть технические метки — чистим их для поисковика
        search_term = query.replace("ОБЪЕКТ ПРОВЕРКИ:", "").split("\n\nВОПРОС:")[0].strip()

        web_data = await get_web_context(search_term)
        if web_data:
            final_query = f"КОНТЕКСТ ИЗ СЕТИ:\n{web_data}\n\nЗАДАЧА: Проведи фактчекинг: {search_term}"

    # 3. Работа с историей (оставляем как было, но используем final_query)
    history = chat_histories[chat_id]
    history.append(Content(role="user", parts=[types.Part(text=final_query)]))
    chat_histories[chat_id] = history[-4:]

    reply_text, used_model, used_prov = None, None, None

    models_to_try = []
    if selected_model:
        models_to_try.append((selected_provider, selected_model))

    for m in current_free_or_models:
        if ("openrouter", m) not in models_to_try:
            models_to_try.append(("openrouter", m))
    for m in GEMINI_MODELS:
        if ("gemini", m) not in models_to_try:
            models_to_try.append(("gemini", m))

    for prov, m_path in models_to_try:
        if m_path in BLACKLISTED_MODELS: continue
        try:
            name_short = get_model_short_name(m_path, prov)
            await context.bot.edit_message_text(f"🔄 Пробую {prov}: {name_short}...", chat_id, status_msg.message_id)

            if prov == "gemini":
                resp = gemini_client.models.generate_content(
                    model=m_path,
                    contents=[Content(role="model", parts=[types.Part(text=system_instruction)])] + history
                )
                reply_text = resp.text
            else:
                messages = [{"role": "system", "content": system_instruction}]
                for h in history:
                    messages.append({"role": "user" if h.role == "user" else "assistant", "content": h.parts[0].text})
                resp = or_client.chat.completions.create(model=m_path, messages=messages)
                reply_text = resp.choices[0].message.content

            if reply_text:
                used_model, used_prov = name_short, prov.capitalize()
                break
        except Exception as e:
            print(f"❌ Модель {m_path} упала: {e}")
            if prov == "openrouter": BLACKLISTED_MODELS.add(m_path)

    if reply_text:
        formatted = f"<b>{used_prov}: {used_model}</b>\n\n{format_to_html(reply_text)}"
        try:
            await context.bot.edit_message_text(formatted, chat_id, status_msg.message_id, parse_mode="HTML")
        except Exception:
            # Fallback на случай ошибок в HTML
            await context.bot.edit_message_text(reply_text[:4000], chat_id, status_msg.message_id)
        chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))
    else:
        await context.bot.edit_message_text("❌ Все модели недоступны.", chat_id, status_msg.message_id)