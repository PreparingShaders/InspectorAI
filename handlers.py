#handlers
import re
import logging
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler, MessageHandler, filters, CommandHandler, CallbackQueryHandler

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

# Нутрициолог
from nutrition import (
    calculate_nutrition_plan,
    update_user_profile,
    get_user_profile,
    get_daily_summary,
    add_food_log
)

# Инструменты и Конфиг
from utils import handle_voice_transcription, get_model_short_name
from config import (
    CORRECT_PASSWORD, AUTH_QUESTION, TRIGGERS, CHECK_WORDS,
    GEMINI_MODELS, FINANCE_WORDS, NUTRITION_TRIGGERS
)

# Состояния пользователей
authorized_users = set()
user_selected_model = {}  # {user_id: model_path}
user_selected_provider = {}  # {user_id: "gemini" или "openrouter"}

# Состояния для диалога
(PROFILE_GENDER, PROFILE_AGE, PROFILE_HEIGHT, PROFILE_WEIGHT, 
 PROFILE_ACTIVITY, PROFILE_GOAL) = range(6)

# --- Хендлеры для Нутрициолога ---

async def profile_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начинает диалог по настройке профиля."""
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.message.reply_text(AUTH_QUESTION)
        return ConversationHandler.END

    keyboard = [
        [InlineKeyboardButton("Мужской", callback_data="male"), InlineKeyboardButton("Женский", callback_data="female")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Давай настроим твой профиль. Укажи свой пол:", reply_markup=reply_markup)
    return PROFILE_GENDER

async def profile_gender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получает пол и запрашивает возраст."""
    query = update.callback_query
    await query.answer()
    context.user_data['profile_gender'] = query.data
    await query.edit_message_text(text=f"Пол: {query.data}. Теперь введи свой возраст (полных лет):")
    return PROFILE_AGE

async def profile_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получает возраст и запрашивает рост."""
    age = int(update.message.text)
    context.user_data['profile_age'] = age
    await update.message.reply_text(f"Возраст: {age}. Теперь введи свой рост (в см):")
    return PROFILE_HEIGHT

async def profile_height(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получает рост и запрашивает вес."""
    height = float(update.message.text)
    context.user_data['profile_height'] = height
    await update.message.reply_text(f"Рост: {height} см. Теперь введи свой вес (в кг):")
    return PROFILE_WEIGHT

async def profile_weight(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получает вес и запрашивает уровень активности."""
    weight = float(update.message.text)
    context.user_data['profile_weight'] = weight
    keyboard = [
        [InlineKeyboardButton("Сидячий образ жизни", callback_data="1.2")],
        [InlineKeyboardButton("Легкая активность (1-3 тренировки/нед)", callback_data="1.375")],
        [InlineKeyboardButton("Средняя активность (3-5 тренировок/нед)", callback_data="1.55")],
        [InlineKeyboardButton("Высокая активность (6-7 тренировок/нед)", callback_data="1.725")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"Вес: {weight} кг. Теперь выбери свой уровень активности:", reply_markup=reply_markup)
    return PROFILE_ACTIVITY

async def profile_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получает активность и запрашивает цель."""
    query = update.callback_query
    await query.answer()
    context.user_data['profile_activity'] = float(query.data)
    keyboard = [
        [InlineKeyboardButton("Похудение", callback_data="weight_loss")],
        [InlineKeyboardButton("Рекомпозиция", callback_data="recomposition")],
        [InlineKeyboardButton("Набор массы", callback_data="mass_gain")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(text="Отлично! И последнee, выбери свою главную цель:", reply_markup=reply_markup)
    return PROFILE_GOAL

async def profile_goal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получает цель, рассчитывает план и завершает диалог."""
    query = update.callback_query
    await query.answer()
    context.user_data['profile_goal'] = query.data

    # Собираем все данные
    profile_data = {
        'user_id': query.from_user.id,
        'gender': context.user_data['profile_gender'],
        'age': context.user_data['profile_age'],
        'height': context.user_data['profile_height'],
        'weight': context.user_data['profile_weight'],
        'activity_level': context.user_data['profile_activity'],
        'goal': context.user_data['profile_goal'],
    }

    # Рассчитываем план
    nutrition_plan = calculate_nutrition_plan(profile_data)
    profile_data.update(nutrition_plan)

    # Сохраняем в БД
    update_user_profile(query.from_user.id, profile_data)

    await query.edit_message_text(
        text=f"✅ Профиль настроен!\n\n"
             f"Твоя цель: {profile_data['goal']}\n"
             f"Твоя норма калорий: {profile_data['target_calories']} ккал\n"
             f"Белки: {profile_data['target_proteins']} г\n"
             f"Жиры: {profile_data['target_fats']} г\n"
             f"Углеводы: {profile_data['target_carbs']} г"
    )
    # Очищаем временные данные
    context.user_data.clear()
    return ConversationHandler.END

async def profile_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отменяет диалог настройки профиля."""
    await update.message.reply_text("Настройка профиля отменена.")
    context.user_data.clear()
    return ConversationHandler.END

async def show_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает дневной статус КБЖУ."""
    user_id = update.effective_user.id
    profile = get_user_profile(user_id)
    if not profile:
        await update.message.reply_text("Сначала нужно настроить профиль с помощью /profile")
        return

    summary = get_daily_summary(user_id)
    
    rem_c = profile['target_calories'] - summary['total_calories']
    rem_p = profile['target_proteins'] - summary['total_proteins']
    rem_f = profile['target_fats'] - summary['total_fats']
    rem_carb = profile['target_carbs'] - summary['total_carbs']

    def fmt(val):
        return f"{val}" if val >= 0 else f"⚠️ {val} (Перебор!)"

    await update.message.reply_text(
        f"📊 *Статус на сегодня*\n\n"
        f"🔥 *Калории:*\n`{summary['total_calories']}` из `{profile['target_calories']}`\nОсталось: `{fmt(rem_c)}`\n\n"
        f"🥩 *Белки:*\n`{summary['total_proteins']}` из `{profile['target_proteins']}`\nОсталось: `{fmt(rem_p)}`\n\n"
        f"🥑 *Жиры:*\n`{summary['total_fats']}` из `{profile['target_fats']}`\nОсталось: `{fmt(rem_f)}`\n\n"
        f"🍞 *Углеводы:*\n`{summary['total_carbs']}` из `{profile['target_carbs']}`\nОсталось: `{fmt(rem_carb)}`",
        parse_mode="MarkdownV2"
    )

profile_setup_handler = ConversationHandler(
    entry_points=[CommandHandler('profile', profile_start)],
    states={
        PROFILE_GENDER: [CallbackQueryHandler(profile_gender, pattern='^(male|female)$')],
        PROFILE_AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_age)],
        PROFILE_HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_height)],
        PROFILE_WEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_weight)],
        PROFILE_ACTIVITY: [CallbackQueryHandler(profile_activity, pattern=r'^1\.')],
        PROFILE_GOAL: [CallbackQueryHandler(profile_goal, pattern='^(weight_loss|recomposition|mass_gain)$')],
    },
    fallbacks=[CommandHandler('cancel', profile_cancel)],
)

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


import time


async def handle_private(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    user_id = update.effective_user.id
    message = update.message
    if not message: return

    raw_text = voice_text or message.text or message.caption or ""
    photo = message.photo[-1] if message.photo else None

    # --- Авторизация ---
    if user_id not in authorized_users:
        if raw_text.strip().lower() == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await message.reply_text("✅ Доступ разрешен!")
        else:
            await message.reply_text(AUTH_QUESTION)
        return

    # --- Обработка подтверждения КБЖУ ---
    if context.user_data.get('confirm_meal') and raw_text.lower() in ['да', 'нет']:
        if raw_text.lower() == 'да':
            meal_data = context.user_data['confirm_meal']
            add_food_log(user_id, meal_data)
            await message.reply_text("✅ Прием пищи сохранен!")
            await show_status(update, context)
        else:
            await message.reply_text("❌ Операция отменена.")
        context.user_data.pop('confirm_meal', None)
        return

    # --- Режим Нутрициолога ---
    is_nutrition_mode = photo and any(word in (raw_text or "").lower() for word in NUTRITION_TRIGGERS)
    if is_nutrition_mode:
        profile = get_user_profile(user_id)
        if not profile:
            await message.reply_text("Сначала нужно настроить профиль с помощью /profile")
            return

        photo_file = await photo.get_file()
        image_data = await photo_file.download_as_bytearray()
        
        prompt = f"Цели пользователя: {profile['target_calories']} ккал, {profile['target_proteins']} г белка. {raw_text}"
        
        llm_response = await process_llm(update, context, prompt, mode="nutrition", image_data=image_data)
        
        # Парсим JSON из ответа
        try:
            json_part = re.search(r'```json\n(.*)\n```', llm_response, re.DOTALL).group(1)
            meal_data = json.loads(json_part)
            
            context.user_data['confirm_meal'] = meal_data
            
            await message.reply_text(
                f"🤖 Я распознал следующее:\n"
                f"Калории: {meal_data['calories']}\n"
                f"Белки: {meal_data['proteins']}\n"
                f"Жиры: {meal_data['fats']}\n"
                f"Углеводы: {meal_data['carbs']}\n\n"
                f"Сохранить эти данные? (да/нет)"
            )
        except (AttributeError, json.JSONDecodeError):
            await message.reply_text("Не удалось распознать КБЖУ в ответе. Попробуйте еще раз.")
        return

    # --- Остальные режимы ---
    is_forwarded = bool(message.forward_origin)
    is_factcheck_trigger = any(word in raw_text.lower() for word in CHECK_WORDS)
    mode = "inspector" if (is_forwarded or is_factcheck_trigger) else "chat"

    context_text = ""
    if message.reply_to_message:
        context_text = message.reply_to_message.text or message.reply_to_message.caption or ""

    final_prompt = f"КОНТЕКСТ ПРЕДЫДУЩЕГО СООБЩЕНИЯ:\n{context_text}\n\nТЕКУЩИЙ ЗАПРОС:\n{raw_text}" if context_text else raw_text

    if final_prompt:
        await process_llm(
            update, context, final_prompt,
            selected_model=user_selected_model.get(user_id),
            selected_provider=user_selected_provider.get(user_id),
            mode=mode
        )


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