#handlers
import re
import logging
import json
from datetime import date, timedelta
from collections import defaultdict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes, ConversationHandler, MessageHandler, filters, CommandHandler, CallbackQueryHandler
from telegram.helpers import escape_markdown

# ... (импорты)
from finance import (
    register_user, apply_expense,
    get_detailed_report, get_all_users_except, settle_debt
)
from llm_service import (
    update_model_mappings, current_free_or_models,
    GEMINI_MODEL_BY_ID, OPENROUTER_MODEL_BY_ID, process_llm
)
from nutrition import (
    calculate_nutrition_plan,
    update_user_profile,
    get_user_profile,
    get_daily_summary,
    add_food_log,
    get_remaining_macros,
    apply_cheat_meal_plan,
    get_adjusted_target,
    get_historical_summary
)
from utils import handle_voice_transcription, get_model_short_name
from config import (
    CORRECT_PASSWORD, AUTH_QUESTION, TRIGGERS, CHECK_WORDS,
    GEMINI_MODELS, FINANCE_WORDS, NUTRITION_TRIGGERS
)

# ... (состояния)
authorized_users = set()
user_selected_model = {}
user_selected_provider = {}
(PROFILE_GENDER, PROFILE_AGE, PROFILE_HEIGHT, PROFILE_WEIGHT, 
 PROFILE_ACTIVITY, PROFILE_GOAL) = range(6)
(CHEAT_MEAL_INPUT, CHEAT_MEAL_CONFIRM) = range(6, 8)


def get_main_keyboard() -> ReplyKeyboardMarkup:
    keyboard = [
        [KeyboardButton("📊 Статус"), KeyboardButton("📈 Статистика")],
        [KeyboardButton("❓ Что съесть?"), KeyboardButton("🍩 Читмил")],
        [KeyboardButton("⚙️ Профиль"), KeyboardButton("🤖 Сменить модель")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def parse_llm_json(response_text: str) -> dict | None:
    """
    NEW: Гибко парсит JSON из ответа LLM.
    Сначала ищет блок ```json, если не находит - пытается парсить весь текст.
    """
    if not response_text:
        return None
    
    # Попытка 1: Найти экранированный блок JSON
    match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logging.warning("Найден блок JSON, но не удалось его распарсить.")
            pass # Переходим к следующей попытке

    # Попытка 2: Считать весь текст как JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        logging.error(f"Не удалось распарсить как JSON весь текст: {response_text}")
        return None

async def show_nutrition_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.message.reply_text(AUTH_QUESTION)
        return
    profile = get_user_profile(user_id)
    if not profile:
        await update.message.reply_text("Сначала нужно настроить профиль: /profile")
        return
    historical_data = get_historical_summary(user_id, days=7)
    if not historical_data:
        await update.message.reply_text("Пока нет данных для статистики. Начни записывать приемы пищи.")
        return
    message = ["*📈 Статистика за последнюю неделю:*\n"]
    total_calories, total_proteins, total_fats, total_carbs, day_count = 0, 0, 0, 0, 0
    for i in range(7):
        day = date.today() - timedelta(days=i)
        day_str, day_name = day.strftime("%Y-%m-%d"), day.strftime("%a")
        day_data = historical_data.get(day_str)
        escaped_date = escape_markdown(f"{day_name} {day.day:02d}.{day.month:02d}", version=2)
        if day_data:
            day_count += 1
            total_calories += day_data['calories']
            total_proteins += day_data['proteins']
            total_fats += day_data['fats']
            total_carbs += day_data['carbs']
            norm_c = profile['target_calories']
            sign = "✅" if day_data['calories'] <= norm_c else "❗️"
            message.append(f"`{escaped_date}`: {sign} {int(day_data['calories'])} / {norm_c} ккал")
        else:
            message.append(f"`{escaped_date}`: 😴 Нет данных")
    if day_count > 0:
        avg_c, avg_p, avg_f, avg_carb = total_calories / day_count, total_proteins / day_count, total_fats / day_count, total_carbs / day_count
        message.append("\n*📊 Среднее / Норма:*")
        message.append(f"🔥 `{int(avg_c)} / {profile['target_calories']}`")
        message.append(f"🥩 `{int(avg_p)} / {profile['target_proteins']}`")
        message.append(f"🥑 `{int(avg_f)} / {profile['target_fats']}`")
        message.append(f"🍞 `{int(avg_carb)} / {profile['target_carbs']}`")
    await update.message.reply_text("\n".join(message), parse_mode="MarkdownV2")

async def get_recipe_suggestion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.message.reply_text(AUTH_QUESTION)
        return
    remaining = get_remaining_macros(user_id)
    if not remaining:
        await update.message.reply_text("Сначала настрой профиль: /profile")
        return
    priority = max(remaining, key=lambda k: remaining[k] if 'calories' not in k else -1)
    prompt = (
        f"Ты — саркастичный фитнес-тренер. У твоего подопечного остались лимиты: "
        f"КБЖУ {remaining['remaining_calories']:.0f}/{remaining['remaining_proteins']:.0f}/{remaining['remaining_fats']:.0f}/{remaining['remaining_carbs']:.0f}. "
        f"Ему особенно не хватает {priority.replace('remaining_', '')}. "
        f"Предложи ОДИН максимально простой перекус или снэк. "
        f"Никаких сложных рецептов. Буквально: 'возьми то, смешай с этим'. "
        f"Говори кратко, по делу и с легкой иронией. Ответ только на русском."
    )
    await process_llm(update, context, prompt, mode="chat")

async def cheat_meal_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.message.reply_text(AUTH_QUESTION)
        return ConversationHandler.END
    await update.message.reply_text("Пришли фото или подробное описание того, что ты съел, и я оценю масштаб трагедии.")
    return CHEAT_MEAL_INPUT

async def cheat_meal_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """UPDATED: Использует новую функцию parse_llm_json."""
    user_id = update.effective_user.id
    message = update.message
    photo = message.photo[-1] if message.photo else None
    text_input = message.text or message.caption or ""
    if not photo and not text_input:
        await message.reply_text("Нужно что-то прислать: фото или текст. Попробуй еще раз или нажми /cancel.")
        return CHEAT_MEAL_INPUT
    image_data = None
    if photo:
        photo_file = await photo.get_file()
        image_data = await photo_file.download_as_bytearray()
    prompt = f"Оцени КБЖУ для этого блюда. Описание: '{text_input}'. Ответь только JSON-объектом вида: {{'calories': X, 'proteins': Y, 'fats': Z, 'carbs': W}}"
    llm_response = await process_llm(update, context, prompt, mode="nutrition", image_data=image_data)
    
    meal_data = parse_llm_json(llm_response) # Используем новый парсер

    if meal_data:
        context.user_data['cheat_meal_data'] = meal_data
        keyboard = [[InlineKeyboardButton("Да, я это съел", callback_data="confirm_cheat"), InlineKeyboardButton("Нет, отмена", callback_data="cancel_cheat")]]
        await message.reply_text(
            f"Я думаю, это было примерно на {meal_data.get('calories', 0)} ккал. "
            f"Записать этот прием пищи и скорректировать остаток на сегодня?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return CHEAT_MEAL_CONFIRM
    else:
        await message.reply_text("Не смог распознать КБЖУ. Попробуй описать блюдо подробнее или пришли другое фото. Для отмены нажми /cancel.")
        return CHEAT_MEAL_INPUT

async def cheat_meal_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    cheat_data = context.user_data.get('cheat_meal_data')
    if query.data == 'confirm_cheat':
        apply_cheat_meal_plan(user_id, cheat_data)
        new_remaining = get_remaining_macros(user_id)
        prompt = (
            f"Ты — поддерживающий ИИ-нутрициолог. Пользователь только что съел лишнего. Его остаток на день теперь: "
            f"{new_remaining['remaining_calories']:.0f} ккал, {new_remaining['remaining_proteins']:.0f} г белка, "
            f"{new_remaining['remaining_fats']:.0f} г жиров, {new_remaining['remaining_carbs']:.0f} г углеводов. "
            f"Не ругай его. Скажи, что все в порядке и читмил уже учтен в сегодняшней норме. "
            f"Чтобы не ложиться спать голодным, предложи ему на выбор что-то ОЧЕНЬ легкое и низкокалорийное. "
            f"Также мягко предложи завтра чуть больше подвигаться. Ответ должен быть коротким, позитивным и только на русском."
        )
        await query.edit_message_text("Понял. Читмил учтен в сегодняшнем дне. Сейчас дам совет, как жить дальше.")
        await process_llm(update, context, prompt, mode="chat")
    else:
        await query.edit_message_text("Хорошо, отменил. Живи спокойно.")
    context.user_data.pop('cheat_meal_data', None)
    return ConversationHandler.END

async def cheat_meal_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    await update.message.reply_text("Диалог 'Читмил' отменен.")
    context.user_data.pop('cheat_meal_data', None)
    return ConversationHandler.END

cheat_meal_handler = ConversationHandler(
    entry_points=[
        CommandHandler('cheatmeal', cheat_meal_start),
        MessageHandler(filters.Regex('^🍩 Читмил$'), cheat_meal_start)
    ],
    states={
        CHEAT_MEAL_INPUT: [MessageHandler(filters.TEXT | filters.PHOTO | filters.CAPTION, cheat_meal_input)],
        CHEAT_MEAL_CONFIRM: [CallbackQueryHandler(cheat_meal_confirm, pattern='^(confirm_cheat|cancel_cheat)$')]
    },
    fallbacks=[CommandHandler('cancel', cheat_meal_cancel)],
)

async def profile_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.message.reply_text(AUTH_QUESTION)
        return ConversationHandler.END
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.reply_text("Укажи свой пол:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Мужской", callback_data="male"), InlineKeyboardButton("Женский", callback_data="female")]]))
    else:
        await update.message.reply_text("Укажи свой пол:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Мужской", callback_data="male"), InlineKeyboardButton("Женский", callback_data="female")]]))
    return PROFILE_GENDER

async def profile_gender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    query = update.callback_query
    await query.answer()
    context.user_data['profile_gender'] = query.data
    await query.edit_message_text(text=f"Пол: {query.data}. Введи возраст:")
    return PROFILE_AGE

async def profile_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    context.user_data['profile_age'] = int(update.message.text)
    await update.message.reply_text("Введи рост (см):")
    return PROFILE_HEIGHT

async def profile_height(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    context.user_data['profile_height'] = float(update.message.text)
    await update.message.reply_text("Введи вес (кг):")
    return PROFILE_WEIGHT

async def profile_weight(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    context.user_data['profile_weight'] = float(update.message.text)
    keyboard = [
        [InlineKeyboardButton("Сидячий", callback_data="1.2")],
        [InlineKeyboardButton("Легкая активность", callback_data="1.375")],
        [InlineKeyboardButton("Средняя активность", callback_data="1.55")],
        [InlineKeyboardButton("Высокая активность", callback_data="1.725")],
    ]
    await update.message.reply_text("Уровень активности:", reply_markup=InlineKeyboardMarkup(keyboard))
    return PROFILE_ACTIVITY

async def profile_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    query = update.callback_query
    await query.answer()
    context.user_data['profile_activity'] = float(query.data)
    keyboard = [
        [InlineKeyboardButton("Похудение", callback_data="weight_loss")],
        [InlineKeyboardButton("Рекомпозиция", callback_data="recomposition")],
        [InlineKeyboardButton("Набор массы", callback_data="mass_gain")],
    ]
    await query.edit_message_text(text="Выбери цель:", reply_markup=InlineKeyboardMarkup(keyboard))
    return PROFILE_GOAL

async def profile_goal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    query = update.callback_query
    await query.answer()
    context.user_data['profile_goal'] = query.data
    profile_data = {
        'user_id': query.from_user.id,
        'gender': context.user_data['profile_gender'], 'age': context.user_data['profile_age'],
        'height': context.user_data['profile_height'], 'weight': context.user_data['profile_weight'],
        'activity_level': context.user_data['profile_activity'], 'goal': context.user_data['profile_goal'],
    }
    nutrition_plan = calculate_nutrition_plan(profile_data)
    profile_data.update(nutrition_plan)
    update_user_profile(query.from_user.id, profile_data)
    await query.edit_message_text(
        f"✅ Профиль настроен!\n"
        f"Цель: {profile_data['goal']}\n"
        f"Калории: {profile_data['target_calories']} ккал\n"
        f"Б: {profile_data['target_proteins']} г, Ж: {profile_data['target_fats']} г, У: {profile_data['target_carbs']} г"
    )
    context.user_data.clear()
    return ConversationHandler.END

async def profile_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (код без изменений)
    await update.message.reply_text("Настройка профиля отменена.")
    context.user_data.clear()
    return ConversationHandler.END

async def show_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)
    user_id = update.effective_user.id
    target = get_adjusted_target(user_id)
    if not target:
        await update.message.reply_text("Сначала нужно настроить профиль с помощью /profile")
        return
    summary = get_daily_summary(user_id)
    rem_c = target['target_calories'] - summary['total_calories']
    rem_p = target['target_proteins'] - summary['total_proteins']
    rem_f = target['target_fats'] - summary['total_fats']
    rem_carb = target['target_carbs'] - summary['total_carbs']
    def fmt(val): return f"{val}" if val >= 0 else f"⚠️ {val}"
    await update.message.reply_text(
        f"📊 *Статус на сегодня*\n\n"
        f"🔥 *Калории:*\n`{summary['total_calories']}` из `{target['target_calories']}`\nОсталось: `{fmt(rem_c)}`\n\n"
        f"🥩 *Белки:*\n`{summary['total_proteins']}` из `{target['target_proteins']}`\nОсталось: `{fmt(rem_p)}`\n\n"
        f"🥑 *Жиры:*\n`{summary['total_fats']}` из `{target['target_fats']}`\nОсталось: `{fmt(rem_f)}`\n\n"
        f"🍞 *Углеводы:*\n`{summary['total_carbs']}` из `{target['target_carbs']}`\nОсталось: `{fmt(rem_carb)}`",
        parse_mode="MarkdownV2"
    )

profile_setup_handler = ConversationHandler(
    entry_points=[
        CommandHandler('profile', profile_start),
        MessageHandler(filters.Regex('^⚙️ Профиль$'), profile_start)
    ],
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)
    user_id = update.effective_user.id
    if user_id in authorized_users:
        await update.message.reply_text(
            "Клавиатура на месте! Чем займемся?",
            reply_markup=get_main_keyboard()
        )
    else:
        await update.message.reply_text(AUTH_QUESTION)

async def show_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)
    update_model_mappings()
    user_id = update.effective_user.id
    keyboard = []
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
    # ... (код без изменений)
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
    """UPDATED: Использует новую функцию parse_llm_json."""
    user_id = update.effective_user.id
    message = update.message
    if not message: return
    raw_text = voice_text or message.text or message.caption or ""
    photo = message.photo[-1] if message.photo else None
    if user_id not in authorized_users:
        if raw_text.strip().lower() == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await message.reply_text(
                "✅ Доступ разрешен! Теперь у тебя есть кнопки для быстрого доступа.",
                reply_markup=get_main_keyboard()
            )
        else:
            await message.reply_text(AUTH_QUESTION)
        return
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
    
    # --- Mode Determination ---
    final_prompt = raw_text
    mode = "chat"
    is_forwarded = bool(message.forward_origin)
    is_factcheck_trigger = any(word in raw_text.lower() for word in CHECK_WORDS)
    is_nutrition_trigger = photo and any(word in (raw_text or "").lower() for word in NUTRITION_TRIGGERS)

    if is_forwarded or is_factcheck_trigger:
        mode = "inspector"
        if message.reply_to_message:
            reply_text = message.reply_to_message.text or message.reply_to_message.caption or ""
            final_prompt = f"Контекст: {reply_text}\nВопрос: {raw_text}"
    elif is_nutrition_trigger:
        profile = get_user_profile(user_id)
        if not profile:
            await message.reply_text("Сначала нужно настроить профиль с помощью /profile")
            return
        photo_file = await photo.get_file()
        image_data = await photo_file.download_as_bytearray()
        prompt = f"Цели пользователя: {profile['target_calories']} ккал, {profile['target_proteins']} г белка. {raw_text}"
        llm_response = await process_llm(update, context, prompt, mode="nutrition", image_data=image_data)
        
        meal_data = parse_llm_json(llm_response) # Используем новый парсер

        if meal_data:
            context.user_data['confirm_meal'] = meal_data
            await message.reply_text(
                f"🤖 Я распознал:\n"
                f"КБЖУ: {meal_data['calories']}/{meal_data['proteins']}/{meal_data['fats']}/{meal_data['carbs']}\n"
                f"Сохранить? (да/нет)"
            )
        else:
            await message.reply_text("Не удалось распознать КБЖУ в ответе. Попробуйте еще раз.")
        return
    else:
        if message.reply_to_message:
            context_text = message.reply_to_message.text or message.reply_to_message.caption or ""
            final_prompt = f"КОНТЕКСТ ПРЕДЫДУЩЕГО СООБЩЕНИЯ:\n{context_text}\n\nТЕКУЩИЙ ЗАПРОС:\n{raw_text}"

    if final_prompt:
        await process_llm(
            update, context, final_prompt,
            selected_model=user_selected_model.get(user_id),
            selected_provider=user_selected_provider.get(user_id),
            mode=mode
        )

async def handle_group(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    # ... (код без изменений)
    message = update.message
    if not message: return
    register_user(update.effective_user.id, update.effective_user.first_name)
    raw_text = voice_text or message.text or message.caption or ""
    text_lower = raw_text.lower()
    state = context.user_data.get('finance_state')
    if state in ['WAITING_AMOUNT', 'WAITING_PAYBACK_AMOUNT']:
        amount_match = re.search(r"(\d+(?:[.,]\d+)?)", raw_text)
        if amount_match:
            amount = float(amount_match.group(1).replace(',', '.'))
            if state == 'WAITING_AMOUNT':
                context.user_data.update(
                    {'tmp_amount': amount, 'tmp_participants': [], 'finance_state': 'SELECT_PARTICIPANTS'})
                await send_participant_selector(update, context)
            else:
                from finance import settle_debt
                creditor_id = context.user_data.get('tmp_creditor_id')
                success, text = settle_debt(update.effective_user.id, creditor_id, amount)
                if success:
                    context.user_data.clear()
                    await message.reply_text(text, parse_mode="HTML")
                else:
                    await message.reply_text(f"❌ {text}\nПопробуй еще раз или напиши 'отмена'.", parse_mode="HTML")
            return
    trigger_pattern = rf"^({'|'.join(map(re.escape, TRIGGERS))})\b"
    if not re.search(trigger_pattern, text_lower): return
    user_query = re.sub(trigger_pattern, '', raw_text, flags=re.IGNORECASE).strip().lstrip(',. ')
    query_lower = user_query.lower()
    if any(w in query_lower for w in ["баланс", "задолжность", "кто кому"]):
        await message.reply_text(get_detailed_report(), parse_mode="HTML")
        return
    if any(word in query_lower for word in FINANCE_WORDS):
        context.user_data['finance_state'] = 'WAITING_AMOUNT'
        await message.reply_text("💵 <b>Введите сумму расхода:</b>", parse_mode="HTML")
        return
    if any(w in query_lower for w in ["час расплаты", "вернуть долг", "отдать долг", 'ланистеры платят долги']):
        from finance import load_db
        db = load_db()
        my_debts = db.get(str(update.effective_user.id), {}).get("debts", {})
        active_debts = {k: v for k, v in my_debts.items() if v > 0}
        if not active_debts:
            await message.reply_text("✨ Ты никому ничего не должен.")
            return
        if len(active_debts) == 1:
            creditor_id = list(active_debts.keys())[0]
            context.user_data.update({
                'finance_state': 'WAITING_PAYBACK_AMOUNT',
                'tmp_creditor_id': creditor_id
            })
            creditor_name = db.get(creditor_id, {}).get("name", "Друг")
            await message.reply_text(f"💰 Сколько возвращаем для <b>{creditor_name}</b>?", parse_mode="HTML")
        else:
            keyboard = []
            for c_id, amt in active_debts.items():
                c_name = db.get(c_id, {}).get("name", "Unknown")
                keyboard.append(
                    [InlineKeyboardButton(f"{c_name} (долг: {amt} р.)", callback_data=f"pay_select:{c_id}")])
            keyboard.append([InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")])
            await message.reply_text("Кому именно ты возвращаешь долг?", reply_markup=InlineKeyboardMarkup(keyboard))
        return
    is_factcheck = any(word in query_lower for word in CHECK_WORDS)
    mode = "inspector" if is_factcheck else "chat"
    if message.reply_to_message:
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
    # ... (код без изменений)
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id
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

async def send_participant_selector(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (код без изменений)
    payer_id = update.effective_user.id
    chat_id = update.effective_chat.id
    amount = context.user_data.get('tmp_amount')
    selected = context.user_data.get('tmp_participants', [])
    all_known_users = get_all_users_except(payer_id)
    keyboard = []
    for uid, name in all_known_users.items():
        try:
            member = await context.bot.get_chat_member(chat_id, int(uid))
            if member.status not in ['left', 'kicked']:
                label = f"✅ {name}" if uid in selected else name
                keyboard.append([InlineKeyboardButton(label, callback_data=f"f_toggle:{uid}")])
        except Exception as e:
            logging.error(f"Ошибка проверки юзера {uid}: {e}")
            continue
    if not keyboard:
        text = f"💰 <b>Сумма: {amount} р.</b>\n\n❌ В базе нет других участников этой группы."
        keyboard.append([InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")])
    else:
        text = f"💰 <b>Сумма: {amount} р.</b>\n\nВыбери тех, кто скидывается:"
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