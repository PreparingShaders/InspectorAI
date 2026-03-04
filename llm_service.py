#llm_service
import re
import requests, telegram
from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Content
from collections import defaultdict
from telegram.helpers import escape_markdown
from PIL import Image
import io
import base64

from config import (
    GEMINI_API_KEY, OPEN_ROUTER_API_KEY, WORKER_URL, NUTRITION_MODELS, NUTRITION_TRIGGERS,
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


async def process_llm(update, context, query, selected_model=None, selected_provider=None, thread_id=None, mode="chat",
                      image_data=None):
    chat_id = update.effective_chat.id
    query_text = query if query else ""
    query_lower = query_text.lower().strip()

    # --- ЖЕСТКИЙ ФИЛЬТР: ЕСЛИ ЕСТЬ ФОТО ---
    is_nutrition_visual = False
    if image_data:
        # 1. Если текста к фото вообще нет — ИГНОРИРУЕМ
        if not query_lower:
            return None

        # 2. Если текст есть, ищем триггеры
        has_trigger = any(trigger in query_lower for trigger in NUTRITION_TRIGGERS)

        if has_trigger:
            is_nutrition_visual = True
        else:
            # 3. Если текст есть (например, "привет"), но в нем НЕТ триггера — ИГНОРИРУЕМ
            # Это именно то, что ты просил: фото + обычный текст = игнор
            return None

    if mode == "nutrition" or is_nutrition_visual:
        system_instruction = "Ты — ИИ-нутрициолог. Твоя задача — анализировать фото еды, определять КБЖУ и давать советы на основе лимитов пользователя."
    else:
        system_instruction = SYSTEM_PROMPT_INSPECTOR if mode == "inspector" else SYSTEM_PROMPT_CHAT

    # ОТПРАВЛЯЕМ СТАТУС БЕЗ parse_mode
    status_text = "⚡ Работаю..." if mode != "inspector" else "🔍 Вхожу в режим инспектора: анализирую..."
    if is_nutrition_visual: status_text = "🥗 Считаю калории..."

    status_msg = await context.bot.send_message(
        chat_id=chat_id,
        text=status_text,
        message_thread_id=thread_id
    )

    final_query = query_text
    if mode == "inspector":
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_msg.message_id,
            text="🌐 Ищу информацию в интернете..."
        )

        search_term = query_text.replace("ОБЪЕКТ ПРОВЕРКИ:", "").split("\n\nВОПРОС:")[0].strip()
        web_data = await get_web_context(search_term)

        if web_data:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text="🧠 Данные получены. Формирую ответ..."
            )
            final_query = f"ДАННЫЕ ИЗ СЕТИ:\n{web_data}\n\nЗАПРОС:\n{query_text}"

    # История
    history = chat_histories[chat_id]
    history.append(Content(role="user", parts=[types.Part(text=query_text or "[Фото]")]))
    chat_histories[chat_id] = history[-6:]

    # ОЧЕРЕДЬ МОДЕЛЕЙ
    models_to_try = []
    if selected_model and selected_provider:
        models_to_try.append((selected_provider, selected_model))

    if is_nutrition_visual:
        # Для еды используем наш спец-список с авто-провайдером
        for m in NUTRITION_MODELS:
            prov = "openrouter" if "/" in m else "gemini"
            if (prov, m) not in models_to_try:
                models_to_try.append((prov, m))
    else:
        # Обычный список
        for m in current_free_or_models:
            prov = "openrouter" if "/" in m else "gemini"
            if (prov, m) not in models_to_try:
                models_to_try.append((prov, m))

    reply_text, used_model, used_prov = None, None, None
    for prov, m_path in models_to_try:
        if m_path in BLACKLISTED_MODELS: continue

        try:
            name_short = get_model_short_name(m_path, prov)
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=status_msg.message_id,
                text=f"🔄 Пробую {prov}: {name_short}..."
            )

            if prov == "gemini":
                user_parts = [types.Part(text=final_query)]
                if image_data:
                    user_parts.append(types.Part.from_bytes(
                        data=bytes(image_data),
                        mime_type="image/jpeg"
                    ))

                current_contents = [Content(role="model", parts=[types.Part(text=system_instruction)])]
                current_contents += chat_histories[chat_id][:-1]
                current_contents.append(Content(role="user", parts=user_parts))

                resp = gemini_client.models.generate_content(model=m_path, contents=current_contents)
                reply_text = resp.text
            else:
                # OPENROUTER С ПОДДЕРЖКОЙ ФОТО
                messages = [{"role": "system", "content": system_instruction}]
                for h in chat_histories[chat_id][:-1]:
                    h_text = h.parts[0].text if h.parts else ""
                    messages.append({"role": "user" if h.role == "user" else "assistant", "content": h_text})

                user_content = [{"type": "text", "text": final_query}]
                if image_data:
                    img_b64 = base64.b64encode(image_data).decode('utf-8')
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })

                messages.append({"role": "user", "content": user_content})
                resp = or_client.chat.completions.create(model=m_path, messages=messages, temperature=0.7)
                reply_text = resp.choices[0].message.content

            if reply_text:
                used_model, used_prov = name_short, prov.capitalize()
                chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))
                break

        except Exception as e:
            print(f"❌ Ошибка {m_path}: {e}")
            if any(code in str(e) for code in ["404", "401", "model_not_found"]):
                BLACKLISTED_MODELS.add(m_path)

    # Отправка результата (ТВОЙ БЛОК БЕЗ ИЗМЕНЕНИЙ)
    if reply_text:
        clean_header = escape_markdown(f"{used_prov}: {used_model}", version=2)
        formatted_body = safe_format_to_html(reply_text)
        final_text = f"*{clean_header}*\n\n{formatted_body}"

        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=status_msg.message_id,
                text=final_text, parse_mode="MarkdownV2"
            )
        except Exception as e:
            print(f"❌ Критическая ошибка разметки: {e}")
            try:
                fallback_text = escape_markdown(f"{used_prov}: {used_model}\n\n{reply_text}", version=2)
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=status_msg.message_id,
                    text=fallback_text[:4090],
                    parse_mode="MarkdownV2"
                )
            except:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=status_msg.message_id,
                    text=f"📋 Ответ (без разметки):\n\n{reply_text}"
                )

        return reply_text
    return None