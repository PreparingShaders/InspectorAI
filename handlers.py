import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from llm_service import process_llm
from utils import handle_voice_transcription

# Статика из конфига
from config import (
    CORRECT_PASSWORD, AUTH_QUESTION, TRIGGERS, CHECK_WORDS, GEMINI_MODELS
)

# Инструменты из utils
from utils import get_model_short_name

# Состояния пользователей
authorized_users = set()
user_selected_model = {}  # {user_id: model_path}
user_selected_provider = {}  # {user_id: "gemini" или "openrouter"}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in authorized_users:
        model = user_selected_model.get(user_id)
        text = "Ты уже авторизован!\n\n"
        if model:
            prov = user_selected_provider.get(user_id, "").upper()
            name = get_model_short_name(model, prov.lower())
            text += f"Текущая модель: {prov} → {name}\n\n"
        else:
            text += "Режим: 🤖 Автоматический выбор\n\n"
        text += "Сменить модель → /model"
        await update.message.reply_text(text)
    else:
        await update.message.reply_text(AUTH_QUESTION)


async def show_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ЛОКАЛЬНЫЙ ИМПОРТ для предотвращения циклической зависимости
    from llm_service import update_model_mappings, current_free_or_models

    update_model_mappings()
    user_id = update.effective_user.id
    keyboard = []

    # --- Секция OpenRouter ---
    keyboard.append([InlineKeyboardButton("🎁 OpenRouter (Most Popular Free):", callback_data="dummy")])
    or_buttons = []
    for i, model in enumerate(current_free_or_models):
        name = get_model_short_name(model, "openrouter")
        prefix = "✅ " if user_selected_model.get(user_id) == model else ""
        or_buttons.append(InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:o:{i + 100}"))
        if len(or_buttons) == 2:
            keyboard.append(or_buttons)
            or_buttons = []
    if or_buttons: keyboard.append(or_buttons)

    # --- Секция Gemini ---
    keyboard.append([InlineKeyboardButton("──────────────", callback_data="dummy")])
    keyboard.append([InlineKeyboardButton("✨ Gemini (Резерв):", callback_data="dummy")])
    gem_buttons = []
    for i, model in enumerate(GEMINI_MODELS):
        name = get_model_short_name(model, "gemini")
        prefix = "✅ " if user_selected_model.get(user_id) == model else ""
        gem_buttons.append(InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:g:{i}"))
        if len(gem_buttons) == 2:
            keyboard.append(gem_buttons)
            gem_buttons = []
    if gem_buttons: keyboard.append(gem_buttons)

    keyboard.append([InlineKeyboardButton("🤖 Автовыбор (OR -> Gem)", callback_data="sel:auto")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    text = "<b>Выбор модели ИИ</b>\nСортировка по весу знаний (B) и популярности."
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.effective_message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")


# handlers.py
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.voice:
        return

    # 1. Распознаем
    text = await handle_voice_transcription(message)

    if text:
        # 2. СРАЗУ отправляем транскрипцию (чтобы видеть результат в группе)
        await message.reply_text(f"🎤 <b>Распознано:</b>\n<i>{text}</i>", parse_mode="HTML")

        # 3. Пробрасываем в логику команд
        if update.effective_chat.type in ["group", "supergroup"]:
            await handle_group(update, context, voice_text=text)
        else:
            await handle_private(update, context, voice_text=text)

async def handle_private(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    user_id = update.effective_user.id
    message = update.message
    if not message: return

    # 1. Определяем, с каким текстом работаем
    raw_text = voice_text or message.text
    if not raw_text: return

    # 2. Проверка авторизации
    if user_id not in authorized_users:
        # Проверяем пароль именно в raw_text (чтобы можно было сказать пароль голосом)
        if raw_text.strip().lower() == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await message.reply_text("Доступ разрешен! Используй /model для выбора.")
        else:
            await message.reply_text(AUTH_QUESTION)
        return

    # 3. Основная логика (уже для авторизованных)
    text_lower = raw_text.lower()

    # Ищем ссылки именно в сообщении (Entities)
    has_url = any(e.type in ['url', 'text_link'] for e in (message.entities or []))

    # Ищем проверочные слова в распознанном или присланном тексте
    has_check = any(word in text_lower for word in CHECK_WORDS)

    # Решаем: Инспектор или Чат
    mode = "inspector" if (has_url or has_check) else "chat"

    from llm_service import process_llm
    # Передаем raw_text (распознанный голос или текст) в LLM
    await process_llm(
        update, context, raw_text,
        user_selected_model.get(user_id),
        user_selected_provider.get(user_id),
        mode=mode
    )

# handlers.py

async def handle_group(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    message = update.message
    if not message or not (message.text or message.caption or voice_text):
        return

    # 1. Сбор текста: приоритет у голоса, затем текст сообщения или подпись к фото
    raw_text = voice_text or message.text or message.caption or ""
    text_lower = raw_text.lower()
    user_id = update.effective_user.id

    # 2. Проверка триггеров обращения (регулярка теперь смотрит в raw_text)
    trigger_pattern = rf"^({'|'.join(map(re.escape, TRIGGERS))})\b"
    match = re.search(trigger_pattern, text_lower)

    # 3. Проверка: ответ ли это на сообщение бота
    is_reply_to_bot = False
    if message.reply_to_message:
        is_reply_to_bot = message.reply_to_message.from_user.id == context.bot.id

    # Если нет ни триггера, ни реплая боту — игнорируем
    if not (match or is_reply_to_bot):
        return

    # 4. Режим Инспектора (проверка фактов)
    is_factcheck = any(word in text_lower for word in CHECK_WORDS)
    mode = "inspector" if is_factcheck else "chat"

    # 5. Чистим текст запроса от триггера (если он был в начале)
    user_query = raw_text
    if match:
        # Убираем "Андрюха, " из начала, сохраняя регистр остальной части
        user_query = re.sub(trigger_pattern, '', raw_text, flags=re.IGNORECASE).strip()

    # 6. Логика работы с контекстом (реплаи)
    if message.reply_to_message:
        reply = message.reply_to_message
        reply_text = reply.text or reply.caption or ""

        if is_factcheck:
            # Если сказали "проверь" на чье-то сообщение
            final_prompt = f"ОБЪЕКТ ПРОВЕРКИ: {reply_text}\n\nВОПРОС: {user_query}"
        else:
            # Если просто болтаем в контексте реплая
            final_prompt = f"Контекст сообщения: {reply_text}\nВопрос: {user_query}"
    else:
        final_prompt = user_query

    # 7. Отправка в LLM
    from llm_service import process_llm
    await process_llm(
        update, context, final_prompt,
        user_selected_model.get(user_id),
        user_selected_provider.get(user_id),
        thread_id=message.message_thread_id,
        mode=mode
    )

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from llm_service import GEMINI_MODEL_BY_ID, OPENROUTER_MODEL_BY_ID

    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    if data == "open_menu":
        await show_model_selection(update, context)
        return

    if data == "sel:auto":
        user_selected_model[user_id] = None
        user_selected_provider[user_id] = None
        await query.edit_message_text("🤖 Режим автовыбора включен (сначала лучшие бесплатные OR).")
        return

    if not data.startswith("sel:"): return

    _, prov_code, idx = data.split(":")
    model_path = None
    provider = None

    if prov_code == "g":
        model_path = GEMINI_MODEL_BY_ID.get(idx)
        provider = "gemini"
    elif prov_code == "o":
        model_path = OPENROUTER_MODEL_BY_ID.get(idx)
        provider = "openrouter"

    if model_path:
        user_selected_model[user_id] = model_path
        user_selected_provider[user_id] = provider
        name = get_model_short_name(model_path, provider)
        await query.edit_message_text(f"🎯 Выбрана модель:\n<b>{provider.upper()}</b> → <code>{name}</code>",
                                      parse_mode="HTML")