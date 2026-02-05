#handlers
import re
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

# Финансы
from finance import (
    register_user, apply_expense,
    get_detailed_report, get_all_users_except, settle_debt
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

    payer_id = update.effective_user.id
    chat_id = update.effective_chat.id
    amount = context.user_data.get('tmp_amount')
    selected = context.user_data.get('tmp_participants', [])

    # 1. Получаем ВСЕХ из базы
    all_known_users = get_all_users_except(payer_id)
    keyboard = []

    # 2. Фильтруем только тех, кто в этом чате
    for uid, name in all_known_users.items():
        try:
            # Проверяем статус участника
            member = await context.bot.get_chat_member(chat_id, int(uid))
            # Если юзер не вышел и не забанен
            if member.status not in ['left', 'kicked']:
                label = f"✅ {name}" if uid in selected else name
                keyboard.append([InlineKeyboardButton(label, callback_data=f"f_toggle:{uid}")])
        except Exception as e:
            # Если ошибка (юзер не найден в чате), просто идем дальше
            logging.error(f"Ошибка проверки юзера {uid}: {e}")
            continue

    # 3. Если вдруг в списке никого нет (кроме плательщика)
    if not keyboard:
        text = f"💰 <b>Сумма: {amount} р.</b>\n\n❌ В базе нет других участников этой группы.\nПусть они напишут что-нибудь в чат, чтобы я их запомнил!"
        keyboard.append([InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")])
    else:
        text = f"💰 <b>Сумма: {amount} р.</b>\n\nВыбери тех, кто скидывается (кроме тебя):"
        keyboard.append([
            InlineKeyboardButton("🚀 РАССЧИТАТЬ", callback_data="f_confirm"),
            InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")
        ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        if update.callback_query:
            await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
        else:
            await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")
    except Exception as e:
        logging.error(f"Ошибка отправки кнопок: {e}")

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

    # 1. Проверяем, является ли сообщение пересланным (форвардом)
    is_forwarded = bool(message.forward_origin)

    # 2. Собираем текст из реплая (если есть), учитывая подписи к медиа
    if message.reply_to_message:
        reply_text = message.reply_to_message.text or message.reply_to_message.caption or ""
    else:
        reply_text = ""

    # 3. Определяем режим: если есть триггер-слово ИЛИ это форвард
    is_factcheck = any(word in raw_text.lower() for word in CHECK_WORDS)

    if is_factcheck or is_forwarded:
        mode = "inspector"
    else:
        mode = "chat"

    # 4. Формируем финальный промпт
    # Если есть реплей (контекст), склеиваем его с вопросом/командой юзера
    if reply_text:
        final_prompt = f"Контекст: {reply_text}\nВопрос: {raw_text}"
    else:
        # Если реплея нет, но это форвард с подписью — текст уже в raw_text
        final_prompt = raw_text

    await process_llm(update, context, final_prompt, user_selected_model.get(user_id),
                      user_selected_provider.get(user_id), mode=mode)


async def handle_group(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    message = update.message
    if not message: return

    # Регистрация участника
    register_user(update.effective_user.id, update.effective_user.first_name)

    raw_text = voice_text or message.text or message.caption or ""
    text_lower = raw_text.lower()

    # Проверка состояний финансов
    state = context.user_data.get('finance_state')

    if state in ['WAITING_AMOUNT', 'WAITING_PAYBACK_AMOUNT']:
        amount_match = re.search(r"(\d+(?:[.,]\d+)?)", raw_text)
        if amount_match:
            amount = float(amount_match.group(1).replace(',', '.'))

            if state == 'WAITING_AMOUNT':
                context.user_data.update(
                    {'tmp_amount': amount, 'tmp_participants': [], 'finance_state': 'SELECT_PARTICIPANTS'})
                await send_participant_selector(update, context)

            else:  # WAITING_PAYBACK_AMOUNT
                from finance import settle_debt
                creditor_id = context.user_data.get('tmp_creditor_id')
                success, text = settle_debt(update.effective_user.id, creditor_id, amount)
                if success:
                    context.user_data.clear()
                    await message.reply_text(text, parse_mode="HTML")
                else:
                    await message.reply_text(f"❌ {text}\nПопробуй еще раз или напиши 'отмена'.", parse_mode="HTML")
            return

    # Проверка триггера (имя бота)
    trigger_pattern = rf"^({'|'.join(map(re.escape, TRIGGERS))})\b"
    if not re.search(trigger_pattern, text_lower): return

    user_query = re.sub(trigger_pattern, '', raw_text, flags=re.IGNORECASE).strip().lstrip(',. ')
    query_lower = user_query.lower()

    # Финансовый блок
    # 1. Сначала проверяем жесткие команды баланса (независимо от FINANCE_WORDS)
    if any(w in query_lower for w in ["баланс", "задолжность", "кто кому"]):
        await message.reply_text(get_detailed_report(), parse_mode="HTML")
        return

    # 2. Потом проверяем FINANCE_WORDS для начала записи чека
    if any(word in query_lower for word in FINANCE_WORDS):
        context.user_data['finance_state'] = 'WAITING_AMOUNT'
        await message.reply_text("💵 <b>Введите сумму расхода:</b>", parse_mode="HTML")
        return

    # 3. Час расплаты (списание долга)
    if any(w in query_lower for w in ["час расплаты", "вернуть долг", "отдать долг", 'ланистеры платят долги']):
        from finance import load_db
        db = load_db()
        my_debts = db.get(str(update.effective_user.id), {}).get("debts", {})

        # Фильтруем только реальные долги (> 0)
        active_debts = {k: v for k, v in my_debts.items() if v > 0}

        if not active_debts:
            await message.reply_text("✨ Ты никому ничего не должен. Спи спокойно, Ланистер!")
            return

        if len(active_debts) == 1:
            # Если должен только одному человеку
            creditor_id = list(active_debts.keys())[0]
            context.user_data.update({
                'finance_state': 'WAITING_PAYBACK_AMOUNT',
                'tmp_creditor_id': creditor_id
            })
            creditor_name = db.get(creditor_id, {}).get("name", "Друг")
            await message.reply_text(f"💰 Сколько возвращаем для <b>{creditor_name}</b>?", parse_mode="HTML")
        else:
            # Если должен нескольким — строим клавиатуру
            keyboard = []
            for c_id, amt in active_debts.items():
                c_name = db.get(c_id, {}).get("name", "Unknown")
                keyboard.append(
                    [InlineKeyboardButton(f"{c_name} (долг: {amt} р.)", callback_data=f"pay_select:{c_id}")])

            keyboard.append([InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")])
            await message.reply_text("Кому именно ты возвращаешь долг?", reply_markup=InlineKeyboardMarkup(keyboard))
        return


    # Режим LLM
    is_factcheck = any(word in query_lower for word in CHECK_WORDS)
    mode = "inspector" if is_factcheck else "chat"

    # Исправленная логика получения текста из реплая
    if message.reply_to_message:
        # Проверяем и текст, и подпись под фото/видео
        reply_text = message.reply_to_message.text or message.reply_to_message.caption or ""
        final_prompt = f"Контекст: {reply_text}\nВопрос: {user_query}"
    else:
        final_prompt = user_query

    await process_llm(
        update, context, final_prompt,
        user_selected_model.get(update.effective_user.id),
        user_selected_provider.get(update.effective_user.id),
        thread_id=message.message_thread_id,
        mode=mode
    )


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

    if data.startswith("pay_select:"):
        creditor_id = data.split(":")[1]
        from finance import load_db
        db = load_db()
        creditor_name = db.get(creditor_id, {}).get("name", "Друг")

        context.user_data.update({
            'finance_state': 'WAITING_PAYBACK_AMOUNT',
            'tmp_creditor_id': creditor_id
        })
        await query.edit_message_text(f"💰 Сколько возвращаем для <b>{creditor_name}</b>?", parse_mode="HTML")
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