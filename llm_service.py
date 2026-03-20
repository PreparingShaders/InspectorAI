#llm_service
import re
import requests
import base64
from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import Content
from collections import defaultdict
from telegram.error import BadRequest
import html
import logging # Import the logging module
from telegram.constants import MessageLimit # Импортируем лимит сообщений Telegram

from config import (
    GEMINI_API_KEY, OPEN_ROUTER_API_KEY, WORKER_URL, NUTRITION_MODELS,
    SYSTEM_PROMPT_INSPECTOR, SYSTEM_PROMPT_CHAT, SYSTEM_PROMPT_NUTRITION,
    DEFAULT_OPENROUTER_MODELS, GEMINI_MODELS, API_TIMEOUT
)
from web_utils import get_web_context
from utils import to_html, get_model_short_name

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
current_free_or_models = DEFAULT_OPENROUTER_MODELS.copy()
GEMINI_MODEL_BY_ID = {}
OPENROUTER_MODEL_BY_ID = {}
NUTRITION_MODEL_BY_ID = {}
chat_histories = defaultdict(list)
BLACKLISTED_MODELS = set()

def fetch_dynamic_models():
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            all_models = response.json().get('data', [])
            result = DEFAULT_OPENROUTER_MODELS.copy()
            for m in all_models:
                m_id = m.get('id', '')
                pricing = m.get('pricing', {})
                is_free = pricing.get('prompt') == "0" and pricing.get('completion') == "0"
                if is_free and m_id not in result and m_id != "openrouter/free":
                    if m_id not in BLACKLISTED_MODELS:
                        result.append(m_id)
            return result[:15]
    except requests.exceptions.RequestException as e:
        logging.warning(f"Ошибка обновления моделей: {e}")
    return DEFAULT_OPENROUTER_MODELS

def update_model_mappings():
    global current_free_or_models, OPENROUTER_MODEL_BY_ID, NUTRITION_MODEL_BY_ID, GEMINI_MODEL_BY_ID
    new_models = fetch_dynamic_models()
    if new_models:
        current_free_or_models = new_models
        OPENROUTER_MODEL_BY_ID.clear()
        for i, m in enumerate(current_free_or_models):
            OPENROUTER_MODEL_BY_ID[str(i + 100)] = m
    
    GEMINI_MODEL_BY_ID.clear()
    for i, m in enumerate(GEMINI_MODELS):
        GEMINI_MODEL_BY_ID[str(i)] = m

    NUTRITION_MODEL_BY_ID.clear()
    for i, m in enumerate(NUTRITION_MODELS):
        NUTRITION_MODEL_BY_ID[str(i + 200)] = m

# Вызываем один раз при старте
update_model_mappings()

async def process_llm(update, context, query, selected_model=None, selected_provider=None, thread_id=None, mode="chat", image_data=None, suppress_direct_reply=False, system_prompt_override: str | None = None):
    chat_id = update.effective_chat.id

    if system_prompt_override:
        system_instruction = system_prompt_override
    elif mode == "inspector": system_instruction = SYSTEM_PROMPT_INSPECTOR
    elif mode == "nutrition": system_instruction = SYSTEM_PROMPT_NUTRITION
    else: system_instruction = SYSTEM_PROMPT_CHAT

    status_msg = None
    if not suppress_direct_reply:
        status_text = "⚡️ Думаю..."
        if mode == "inspector": status_text = "🔍 Проверяю факты..."
        elif mode == "nutrition": status_text = "🥗 Считаю КБЖУ..."
        status_msg = await context.bot.send_message(chat_id=chat_id, text=status_text, message_thread_id=thread_id)

    final_query = query
    if mode == "inspector" and status_msg:
        await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text="🌐 Ищу информацию в сети...")
        search_term = query.replace("ОБЪЕКТ ПРОВЕРКИ:", "").split("\n\nВОПРОС:")[0].strip()
        web_data = await get_web_context(search_term)
        if web_data:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text="🧠 Анализирую данные...")
            final_query = f"ДАННЫЕ ИЗ СЕТИ:\n{web_data}\n\nЗАПРОС:\n{query}"

    history = chat_histories[chat_id]
    history.append(Content(role="user", parts=[types.Part(text=query or "[Фото]")]))
    chat_histories[chat_id] = history[-6:]

    models_to_try = []
    if selected_model:
        prov = "openrouter" if "/" in selected_model else "gemini"
        models_to_try.append((prov, selected_model))

    if mode == "nutrition":
        for m in NUTRITION_MODELS:
            prov = "openrouter" if "/" in m else "gemini"
            if (prov, m) not in models_to_try: models_to_try.append((prov, m))
    else:
        for m in current_free_or_models:
            if ("openrouter", m) not in models_to_try: models_to_try.append(("openrouter", m))
        for m in GEMINI_MODELS:
            if ("gemini", m) not in models_to_try: models_to_try.append(("gemini", m))

    reply_text, used_model_path, used_prov = None, None, None
    for prov, m_path in models_to_try:
        if m_path in BLACKLISTED_MODELS: continue
        try:
            if status_msg:
                name_short = get_model_short_name(m_path, prov)
                await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text=f"🔄 Пробую {prov}: {name_short}...")

            if prov == "gemini":
                user_parts = [types.Part(text=final_query)]
                if image_data: user_parts.insert(0, types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_data)))
                contents = [Content(role="model", parts=[types.Part(text=system_instruction)])] + chat_histories[chat_id][:-1]
                contents.append(Content(role="user", parts=user_parts))
                resp = gemini_client.models.generate_content(model=m_path, contents=contents)
                reply_text = resp.text
            else: # openrouter
                messages = [{"role": "system", "content": system_instruction}]
                for h in chat_histories[chat_id][:-1]: messages.append({"role": "user" if h.role == "user" else "assistant", "content": h.parts[0].text})
                user_content = [{"type": "text", "text": final_query}]
                if image_data:
                    img_b64 = base64.b64encode(image_data).decode('utf-8')
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
                messages.append({"role": "user", "content": user_content})
                resp = or_client.chat.completions.create(model=m_path, messages=messages, temperature=0.7)
                reply_text = resp.choices[0].message.content

            if reply_text:
                used_model_path, used_prov = m_path, prov
                chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))
                break
        except Exception as e:
            logging.error(f"Ошибка {m_path}: {e}")
            if any(code in str(e) for code in ["404", "401", "model_not_found", "billing", "insufficient_quota"]):
                BLACKLISTED_MODELS.add(m_path)

    if suppress_direct_reply:
        if status_msg: await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
        return reply_text, used_model_path

    if reply_text:
        used_model_name = get_model_short_name(used_model_path, used_prov)
        header = f"<b>{html.escape(used_prov.capitalize())}: {html.escape(used_model_name)}</b>"
        formatted_body = to_html(reply_text)

        # Telegram's message limit is 4096 characters.
        # Резервируем место для заголовка и текста продолжения.
        MAX_PART_LENGTH = MessageLimit.MAX_TEXT_LENGTH - len(header) - 100 

        message_parts = []
        current_part_lines = []
        current_part_length = 0

        lines = formatted_body.split('\n')

        for line in lines:
            # Если добавление текущей строки превышает лимит, и у нас уже есть контент в current_part_lines
            if current_part_length + len(line) + 1 > MAX_PART_LENGTH and current_part_lines: # +1 для переноса строки
                message_parts.append('\n'.join(current_part_lines))
                current_part_lines = [line]
                current_part_length = len(line) + 1
            else:
                current_part_lines.append(line)
                current_part_length += len(line) + 1

        if current_part_lines:
            message_parts.append('\n'.join(current_part_lines))

        # Обработка случаев, когда одна часть все еще слишком длинная (например, очень длинная строка без переносов)
        final_message_parts = []
        for part in message_parts:
            if len(part) > MAX_PART_LENGTH:
                # Если часть все еще слишком длинная, обрезаем ее.
                # Это может нарушить HTML-теги, но это крайняя мера.
                truncated_part = part[:MAX_PART_LENGTH - 50] + "<i>... (обрезано)</i>"
                final_message_parts.append(truncated_part)
            else:
                final_message_parts.append(part)

        # Отправляем первую часть, редактируя статусное сообщение
        if final_message_parts:
            first_part_content = final_message_parts[0]
            if len(final_message_parts) > 1:
                first_part_content += "\n\n<i>(Продолжение следует...)</i>"
            
            try:
                await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text=f"{header}\n\n{first_part_content}", parse_mode="HTML")
            except BadRequest as e:
                logging.error(f"Не удалось отредактировать сообщение с первой частью (HTML): {e}")
                # Fallback на обычный текст, если HTML не сработал для первой части
                plain_text_fallback = f"{header}\n\n{reply_text[:MAX_PART_LENGTH - len(header) - 100]}..."
                await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text=plain_text_fallback)


            # Отправляем последующие части как новые сообщения
            for i in range(1, len(final_message_parts)):
                part_content = final_message_parts[i]
                if i < len(final_message_parts) - 1:
                    part_content += "\n\n<i>(Продолжение...)</i>"
                
                try:
                    await context.bot.send_message(chat_id=chat_id, text=part_content, parse_mode="HTML", message_thread_id=thread_id)
                except BadRequest as e:
                    logging.error(f"Не удалось отправить последующую часть сообщения (HTML): {e}")
                    # Fallback для последующих частей
                    # Здесь мы берем часть из исходного reply_text, чтобы избежать проблем с HTML-парсингом
                    start_idx = i * (MAX_PART_LENGTH - 50) # Приблизительное смещение
                    end_idx = (i + 1) * (MAX_PART_LENGTH - 50)
                    plain_text_fallback = f"<i>(Часть {i+1} ответа)</i>\n\n{reply_text[start_idx:end_idx]}..."
                    await context.bot.send_message(chat_id=chat_id, text=plain_text_fallback, message_thread_id=thread_id)

        else:
            # Контент не сгенерирован
            if status_msg:
                await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text="😔 Не удалось получить ответ от моделей. Попробуйте позже.")

    else:
        if status_msg:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text="😔 Не удалось получить ответ от моделей. Попробуйте позже.")

    return reply_text or ""
