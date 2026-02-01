import re
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

# Финансы
from finance import (
    register_user, apply_expense,
    get_detailed_report, get_all_users_except
)

# LLM и Сервисы
from llm_service import (
    update_model_mappings, current_free_or_models,
    GEMINI_MODEL_BY_ID, OPENROUTER_MODEL_BY_ID, process_llm
)

# Инструменты и Конфиг
from utils import handle_voice_transcription, get_model_short_name
from config import (
    CORRECT_PASSWORD, AUTH_QUESTION, TRIGGERS, CHECK_WORDS,
    GEMINI_MODELS, FINANCE_WORDS
)

# Состояния пользователей
authorized_users = set()
user_selected_model = {}  # {user_id: model_path}
user_selected_provider = {}  # {user_id: "gemini" или "openrouter"}


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

async def send_participant_selector(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отрисовка меню выбора участников для разделения счета"""
    payer_id = update.effective_user.id
    amount = context.user_data.get('tmp_amount')
    selected = context.user_data.get('tmp_participants', [])

    users = get_all_users_except(payer_id)
    keyboard = []

    # Кнопки участников
    for uid, name in users.items():
        label = f"✅ {name}" if uid in selected else name
        keyboard.append([InlineKeyboardButton(label, callback_data=f"f_toggle:{uid}")])

    # Кнопки управления
    keyboard.append([
        InlineKeyboardButton("🚀 РАССЧИТАТЬ", callback_data="f_confirm"),
        InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")
    ])

    text = f"💰 <b>Счет на сумму: {amount} руб.</b>\nКто участвовал в этом расходе?"
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")


# --- ОСНОВНЫЕ ХЕНДЛЕРЫ ---

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
    update_model_mappings()
    user_id = update.effective_user.id
    keyboard = []

    # OpenRouter
    keyboard.append([InlineKeyboardButton("🎁 OpenRouter (Free):", callback_data="dummy")])
    or_buttons = []
    for i, model in enumerate(current_free_or_models):
        name = get_model_short_name(model, "openrouter")
        prefix = "✅ " if user_selected_model.get(user_id) == model else ""
        or_buttons.append(InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:o:{i + 100}"))
        if len(or_buttons) == 2:
            keyboard.append(or_buttons)
            or_buttons = []
    if or_buttons: keyboard.append(or_buttons)

    # Gemini
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

    keyboard.append([InlineKeyboardButton("🤖 Автовыбор", callback_data="sel:auto")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    text = "<b>Выбор модели ИИ</b>"
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.effective_message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.voice: return
    text = await handle_voice_transcription(message)
    if text:
        await message.reply_text(f"🎤 <b>Распознано:</b>\n<i>{text}</i>", parse_mode="HTML")
        if update.effective_chat.type in ["group", "supergroup"]:
            await handle_group(update, context, voice_text=text)
        else:
            await handle_private(update, context, voice_text=text)


async def handle_private(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    user_id = update.effective_user.id
    message = update.message
    if not message: return
    raw_text = voice_text or message.text or message.caption or ""
    if user_id not in authorized_users:
        if raw_text.strip().lower() == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await message.reply_text("✅ Доступ разрешен!")
            return
        await message.reply_text(AUTH_QUESTION)
        return

    # Логика инспектора/чата (упрощено для ревью)
    is_forwarded = bool(message.forward_origin)
    reply_text = message.reply_to_message.text if message.reply_to_message else ""

    mode = "chat"
    final_prompt = raw_text

    for word in CHECK_WORDS:
        if word in raw_text.lower() or is_forwarded:
            mode = "inspector"
            break

    await process_llm(update, context, final_prompt, user_selected_model.get(user_id),
                      user_selected_provider.get(user_id), mode=mode)


async def handle_group(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    message = update.message
    if not message: return

    # Регистрация участника
    register_user(update.effective_user.id, update.effective_user.first_name)

    raw_text = voice_text or message.text or message.caption or ""
    text_lower = raw_text.lower()

    # Проверка состояния "Ожидание суммы"
    if context.user_data.get('finance_state') == 'WAITING_AMOUNT':
        amount_match = re.search(r"(\d+(?:[.,]\d+)?)", raw_text)
        if amount_match:
            amount = float(amount_match.group(1).replace(',', '.'))
            context.user_data.update(
                {'tmp_amount': amount, 'tmp_participants': [], 'finance_state': 'SELECT_PARTICIPANTS'})
            await send_participant_selector(update, context)
            return
        elif "отмена" in text_lower:
            context.user_data.clear()
            await message.reply_text("Отменено.")
            return

    # Проверка триггера (имя бота)
    trigger_pattern = rf"^({'|'.join(map(re.escape, TRIGGERS))})\b"
    if not re.search(trigger_pattern, text_lower): return

    user_query = re.sub(trigger_pattern, '', raw_text, flags=re.IGNORECASE).strip().lstrip(',. ')
    query_lower = user_query.lower()

    # Финансовый блок
    if any(word in query_lower for word in FINANCE_WORDS):
        if any(w in query_lower for w in ["баланс", "долг"]):
            await message.reply_text(get_detailed_report(), parse_mode="HTML")
            return
        if "счет" in query_lower:
            context.user_data['finance_state'] = 'WAITING_AMOUNT'
            await message.reply_text("💵 <b>Введите сумму расхода:</b>", parse_mode="HTML")
            return

    # Режим LLM
    is_factcheck = any(word in query_lower for word in CHECK_WORDS)
    mode = "inspector" if is_factcheck else "chat"

    reply_text = message.reply_to_message.text if message.reply_to_message else ""
    final_prompt = f"Контекст: {reply_text}\nВопрос: {user_query}" if reply_text else user_query

    await process_llm(update, context, final_prompt, user_selected_model.get(update.effective_user.id),
                      user_selected_provider.get(update.effective_user.id), thread_id=message.message_thread_id,
                      mode=mode)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    # --- ФИНАНСЫ В CALLBACK ---
    if data.startswith("f_toggle:"):
        uid = data.split(":")[1]
        participants = context.user_data.get('tmp_participants', [])
        if uid in participants:
            participants.remove(uid)
        else:
            participants.append(uid)
        context.user_data['tmp_participants'] = participants
        await send_participant_selector(update, context)
        await query.answer()
        return

    if data == "f_confirm":
        participants = context.user_data.get('tmp_participants')
        if not participants:
            await query.answer("Выбери хотя бы одного!", show_alert=True)
            return
        share = apply_expense(user_id, participants, context.user_data.get('tmp_amount'))
        await query.edit_message_text(f"✅ Записано!\nКаждый (включая тебя) должен по {share} р.")
        context.user_data.clear()
        return

    if data == "f_cancel":
        context.user_data.clear()
        await query.edit_message_text("❌ Расчет отменен.")
        return

    # --- МОДЕЛИ В CALLBACK ---
    await query.answer()
    if data == "open_menu":
        await show_model_selection(update, context)
    elif data == "sel:auto":
        user_selected_model[user_id] = user_selected_provider[user_id] = None
        await query.edit_message_text("🤖 Автовыбор включен.")
    elif data.startswith("sel:"):
        _, prov_code, idx = data.split(":")
        provider = "gemini" if prov_code == "g" else "openrouter"
        model_path = GEMINI_MODEL_BY_ID.get(idx) if prov_code == "g" else OPENROUTER_MODEL_BY_ID.get(idx)

        if model_path:
            user_selected_model[user_id], user_selected_provider[user_id] = model_path, provider
            name = get_model_short_name(model_path, provider)
            await query.edit_message_text(f"🎯 Выбрана модель: <b>{name}</b>", parse_mode="HTML")