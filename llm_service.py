#llm_service
import re
import requests
from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Content
from collections import defaultdict
from telegram.helpers import escape_markdown

from config import (
    GEMINI_API_KEY, OPEN_ROUTER_API_KEY, WORKER_URL,
    SYSTEM_PROMPT_INSPECTOR, SYSTEM_PROMPT_CHAT, DEFAULT_OPENROUTER_MODELS, GEMINI_MODELS, API_TIMEOUT
)
from web_utils import get_web_context
from utils import safe_format_to_html, get_model_short_name

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
current_free_or_models = [DEFAULT_OPENROUTER_MODELS]
GEMINI_MODEL_BY_ID = {str(i): m for i, m in enumerate(GEMINI_MODELS)}
OPENROUTER_MODEL_BY_ID = {}
chat_histories = defaultdict(list)
BLACKLISTED_MODELS = set()


def fetch_dynamic_models():
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            all_models = response.json().get('data', [])

            # Начинаем с ручного списка
            result = [m for m in DEFAULT_OPENROUTER_MODELS]

            for m in all_models:
                m_id = m.get('id', '')
                pricing = m.get('pricing', {})
                is_free = pricing.get('prompt') == "0" and pricing.get('completion') == "0"

                # Добавляем новые бесплатные модели, которых нет в ручном списке
                if is_free and m_id not in result and m_id != "openrouter/free":
                    if m_id not in BLACKLISTED_MODELS:
                        result.append(m_id)

            return result[:15]
    except Exception as e:
        print(f"⚠️ Ошибка обновления: {e}")
    return DEFAULT_OPENROUTER_MODELS
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

    # Фактчекинг для инспектора
    final_query = query
    if mode == "inspector":
        search_term = query.replace("ОБЪЕКТ ПРОВЕРКИ:", "").split("\n\nВОПРОС:")[0].strip()
        web_data = await get_web_context(search_term)
        if web_data:
            final_query = f"КОНТЕКСТ ИЗ СЕТИ:\n{web_data}\n\nЗАДАЧА: Проведи фактчекинг: {search_term}"

    # История
    history = chat_histories[chat_id]
    history.append(Content(role="user", parts=[types.Part(text=final_query)]))
    chat_histories[chat_id] = history[-6:]

    # ОЧЕРЕДЬ МОДЕЛЕЙ: Сначала Trinity (или то что выбрал юзер), потом остальные
    models_to_try = []
    if selected_model and selected_provider:
        models_to_try.append((selected_provider, selected_model))

    # Добавляем Gemini в конец как надежный бэкап
    for m in GEMINI_MODELS:
        if ("gemini", m) not in models_to_try:
            models_to_try.append(("gemini", m))

    # Добавляем остальные OpenRouter
    for m in current_free_or_models:
        if ("openrouter", m) not in models_to_try:
            models_to_try.append(("openrouter", m))



    reply_text, used_model, used_prov = None, None, None

    for prov, m_path in models_to_try:
        if m_path in BLACKLISTED_MODELS:
            continue

        try:
            name_short = get_model_short_name(m_path, prov)
            await context.bot.edit_message_text(f"🔄 Пробую {prov}: {name_short}...", chat_id, status_msg.message_id)

            if prov == "gemini":
                # Убедитесь, что в config.py GEMINI_MODELS без префикса "models/"
                resp = gemini_client.models.generate_content(
                    model=m_path,
                    contents=[Content(role="model", parts=[types.Part(text=system_instruction)])] + history
                )
                reply_text = resp.text
            else:
                messages = [{"role": "system", "content": system_instruction}]
                for h in history:
                    messages.append({"role": "user" if h.role == "user" else "assistant", "content": h.parts[0].text})

                resp = or_client.chat.completions.create(
                    model=m_path,
                    messages=messages,
                    temperature=0.7,  # Уменьшаем "фантазии", делаем ответы конкретнее
                    top_p=0.9,  # Отсекаем совсем маловероятные слова
                    extra_body={
                        "repetition_penalty": 1.1  # Прямой запрет модели повторять одни и те же фразы
                    }
                )
                reply_text = resp.choices[0].message.content

            if reply_text:
                used_model, used_prov = name_short, prov.capitalize()
                break
        except Exception as e:
            print(f"❌ Ошибка {m_path}: {e}")
            # Временно баним только если это ошибка 401/403/404 (модели нет или лимит)
            if "404" in str(e) or "401" in str(e):
                BLACKLISTED_MODELS.add(m_path)

    # Отправка результата
    if reply_text:
        # 1. Экранируем техническую информацию (имя провайдера и модели)
        # Это важно, чтобы символы типа '-' в названии модели не ломали MarkdownV2
        header_text = escape_markdown(f"{used_prov}: {used_model}", version=2)

        # 2. Формируем финальное сообщение.
        # Используем жирный шрифт (*) для заголовка в стиле MarkdownV2
        formatted = f"*{header_text}*\n\n{safe_format_to_html(reply_text)}"

        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text=formatted,
                parse_mode="MarkdownV2"  # МЕНЯЕМ НА MarkdownV2
            )
        except Exception as e:
            print(f"⚠️ Ошибка рендеринга MarkdownV2: {e}")
            # Если Telegram все равно ругается на разметку, отправляем чистый текст
            clean_text = escape_markdown(reply_text, version=2)
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text=f"*{header_text}*\n\n{clean_text}",
                parse_mode="MarkdownV2"
            )

        chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))
    else:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_msg.message_id,
            text=r"❌ *Все модели временно недоступны\. Попробуйте позже\.*",
            parse_mode="MarkdownV2"
        )