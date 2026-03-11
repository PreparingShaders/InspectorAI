import html
import logging
from datetime import date, timedelta

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
    CommandHandler,
    CallbackQueryHandler,
)

from llm_service import process_llm
from nutrition import (
    calculate_nutrition_plan,
    update_user_profile,
    get_user_profile,
    get_daily_summary,
    get_remaining_macros,
    get_adjusted_target,
    get_historical_summary,
    add_food_log,
)
from config import AUTH_QUESTION
from handlers.state import (
    authorized_users, user_selected_model
)
from handlers.base import (
    cancel_conversation, get_main_keyboard
)

# Conversation states
(PROFILE_GENDER, PROFILE_AGE, PROFILE_HEIGHT, PROFILE_WEIGHT,
 PROFILE_ACTIVITY, PROFILE_GOAL) = range(6)


def get_nutrition_inline_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("📊 Статус", callback_data="nutrition_status"),
         InlineKeyboardButton("📈 Статистика", callback_data="nutrition_stats")],
        [InlineKeyboardButton("❓ Что съесть?", callback_data="nutrition_recipe"),
         InlineKeyboardButton("⚙️ Профиль", callback_data="nutrition_profile")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)


async def show_nutrition_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return
    
    text = "<b>🥗 Меню Нутрициолога:</b>"
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=get_nutrition_inline_keyboard(), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(text, reply_markup=get_nutrition_inline_keyboard(), parse_mode="HTML")


async def show_nutrition_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return
    profile = get_user_profile(user_id)
    if not profile:
        await update.effective_message.reply_text("Сначала нужно настроить профиль: /profile")
        return
    historical_data = get_historical_summary(user_id, days=7)
    if not historical_data:
        await update.effective_message.reply_text("Пока нет данных для статистики. Начни записывать приемы пищи.")
        return
    message = ["<b>📈 Статистика за последнюю неделю:</b>\n"]
    total_calories, total_proteins, total_fats, total_carbs, day_count = 0, 0, 0, 0, 0
    for i in range(7):
        day = date.today() - timedelta(days=i)
        day_str, day_name = day.strftime("%Y-%m-%d"), day.strftime("%a")
        day_data = historical_data.get(day_str)
        escaped_date = f"{html.escape(day_name)} {day.day:02d}.{day.month:02d}"
        if day_data:
            day_count += 1
            total_calories += day_data['calories']
            total_proteins += day_data['proteins']
            total_fats += day_data['fats']
            total_carbs += day_data['carbs']
            norm_c = profile['target_calories']
            sign = "✅" if day_data['calories'] <= norm_c else "❗️"
            message.append(f"<code>{escaped_date}</code>: {sign} {int(day_data['calories'])} / {norm_c} ккал")
        else:
            message.append(f"<code>{escaped_date}</code>: 😴 Нет данных")
    if day_count > 0:
        avg_c, avg_p, avg_f, avg_carb = total_calories / day_count, total_proteins / day_count, total_fats / day_count, total_carbs / day_count
        message.append("\n<b>📊 Среднее / Норма:</b>")
        message.append(f"🔥 <code>{int(avg_c)} / {profile['target_calories']}</code>")
        message.append(f"🥩 <code>{int(avg_p)} / {profile['target_proteins']}</code>")
        message.append(f"🥑 <code>{int(avg_f)} / {profile['target_fats']}</code>")
        message.append(f"🍞 <code>{int(avg_carb)} / {profile['target_carbs']}</code>")
    await update.effective_message.reply_text("\n".join(message), parse_mode="HTML")


async def get_recipe_suggestion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return

    profile = get_user_profile(user_id)
    if not profile:
        await update.effective_message.reply_text("Сначала нужно настроить профиль: /profile")
        return

    today_summary = get_daily_summary(user_id)
    remaining = get_remaining_macros(user_id)
    historical_data = get_historical_summary(user_id, days=7)

    avg_c, avg_p, avg_f, avg_carb, day_count = 0, 0, 0, 0, 0
    if historical_data:
        for day_data in historical_data.values():
            if day_data:
                day_count += 1
                avg_c += day_data['calories']
                avg_p += day_data['proteins']
                avg_f += day_data['fats']
                avg_carb += day_data['carbs']
        if day_count > 0:
            avg_c /= day_count
            avg_p /= day_count
            avg_f /= day_count
            avg_carb /= day_count

    prompt = f"""
Ты — опытный и вдумчивый ИИ-нутрициолог. Твоя задача — помочь пользователю сбалансировать свой рацион на остаток дня.

Вот полная картина по пользователю:

1.  **Дневной план (цель):**
    *   Калории: {profile['target_calories']}
    *   Белки: {profile['target_proteins']} г
    *   Жиры: {profile['target_fats']} г
    *   Углеводы: {profile['target_carbs']} г

2.  **Уже съедено за сегодня:**
    *   Калории: {int(today_summary['total_calories'])}
    *   Белки: {int(today_summary['total_proteins'])} г
    *   Жиры: {int(today_summary['total_fats'])} г
    *   Углеводы: {int(today_summary['total_carbs'])} г

3.  **Осталось на сегодня:**
    *   Калории: {int(remaining['remaining_calories'])}
    *   Белки: {int(remaining['remaining_proteins'])} г
    *   Жиры: {int(remaining['remaining_fats'])} г
    *   Углеводы: {int(remaining['remaining_carbs'])} г

4.  **Контекст (средние показатели за неделю):**
    *   Средние калории: {int(avg_c)}
    *   Средние белки: {int(avg_p)} г
    *   Средние жиры: {int(avg_f)} г
    *   Средние углеводы: {int(avg_carb)} г

**Твоя задача:**
Проанализируй ситуацию и дай пользователю четкий, поддерживающий и выполнимый совет.

*   **Если калории уже превышены:** Прямо скажи об этом. Посоветуй больше сегодня не есть и, возможно, добавить легкую активность (например, прогулку). Успокой, что один день — не катастрофа.
*   **Если есть дефицит:** Предложи 1-2 **конкретных и простых** варианта приема пищи, чтобы закрыть оставшиеся потребности. Сделай акцент на тех нутриентах, которых не хватает больше всего.
*   **Если ситуация смешанная** (например, жиры превышены, а белки в дефиците): Объясни это. Предложи очень легкие, богатые белком и почти безжировые варианты.

Говори на русском языке, кратко и по делу. Твой тон — заботливый профессионал.
"""
    await update.effective_message.reply_text("🤔 Анализирую ваш рацион и думаю, что предложить... один момент.")
    model_to_use = user_selected_model.get(user_id)
    await process_llm(update, context, prompt, mode="chat", selected_model=model_to_use)


async def profile_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return ConversationHandler.END
    if update.callback_query:
        await update.callback_query.answer()
        # Удаляем предыдущее сообщение с инлайн-клавиатурой, чтобы не было дублирования
        await update.callback_query.message.delete()
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Укажи свой пол:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Мужской", callback_data="male"), InlineKeyboardButton("Женский", callback_data="female")]]))
    else:
        await update.effective_message.reply_text("Укажи свой пол:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Мужской", callback_data="male"), InlineKeyboardButton("Женский", callback_data="female")]]))
    return PROFILE_GENDER


async def profile_gender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.user_data['profile_gender'] = query.data
    await query.edit_message_text(text=f"Пол: {query.data}. Введи возраст:")
    return PROFILE_AGE


async def profile_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        age = int(update.message.text)
        if not 10 < age < 100:
            raise ValueError("Некорректный возраст")
        context.user_data['profile_age'] = age
        await update.message.reply_text("Введи рост (см):")
        return PROFILE_HEIGHT
    except (ValueError, TypeError):
        await update.message.reply_text("Это не похоже на возраст. Пожалуйста, введи возраст числом. Для отмены введи /cancel.")
        return PROFILE_AGE


async def profile_height(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        height = float(update.message.text.replace(',', '.'))
        if not 100 < height < 250:
            raise ValueError("Некорректный рост")
        context.user_data['profile_height'] = height
        await update.message.reply_text("Введи вес (кг):")
        return PROFILE_WEIGHT
    except (ValueError, TypeError):
        await update.message.reply_text("Это не похоже на рост. Пожалуйста, введи рост числом (в см). Для отмены введи /cancel.")
        return PROFILE_HEIGHT


async def profile_weight(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        weight = float(update.message.text.replace(',', '.'))
        if not 30 < weight < 200:
            raise ValueError("Некорректный вес")
        context.user_data['profile_weight'] = weight
        keyboard = [
            [InlineKeyboardButton("Сидячий", callback_data="1.2")],
            [InlineKeyboardButton("Легкая активность", callback_data="1.375")],
            [InlineKeyboardButton("Средняя активность", callback_data="1.55")],
            [InlineKeyboardButton("Высокая активность", callback_data="1.725")],
        ]
        await update.message.reply_text("Уровень активности:", reply_markup=InlineKeyboardMarkup(keyboard))
        return PROFILE_ACTIVITY
    except (ValueError, TypeError):
        await update.message.reply_text("Это не похоже на вес. Пожалуйста, введи вес числом (в кг). Для отмены введи /cancel.")
        return PROFILE_WEIGHT


async def profile_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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
        f"✅ <b>Профиль настроен!</b>\n"
        f"Цель: {html.escape(profile_data['goal'])}\n"
        f"Калории: <code>{profile_data['target_calories']}</code> ккал\n"
        f"Б: <code>{profile_data['target_proteins']}</code> г, Ж: <code>{profile_data['target_fats']}</code> г, У: <code>{profile_data['target_carbs']}</code> г",
        parse_mode="HTML"
    )
    context.user_data.clear()
    return ConversationHandler.END


profile_setup_handler = ConversationHandler(
    entry_points=[
        CommandHandler('profile', profile_start),
        CallbackQueryHandler(profile_start, pattern='^nutrition_profile$') # Добавляем этот entry_point
    ],
    states={
        PROFILE_GENDER: [CallbackQueryHandler(profile_gender, pattern='^(male|female)$')],
        PROFILE_AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_age)],
        PROFILE_HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_height)],
        PROFILE_WEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile_weight)],
        PROFILE_ACTIVITY: [CallbackQueryHandler(profile_activity, pattern=r'^1\.')],
        PROFILE_GOAL: [CallbackQueryHandler(profile_goal, pattern='^(weight_loss|recomposition|mass_gain)$')],
    },
    fallbacks=[
        CommandHandler('cancel', cancel_conversation),
    ],
)


def create_progress_bar(percentage: int, length: int = 10) -> str:
    """Генерирует текстовый прогресс-бар."""
    display_percentage = min(percentage, 100)
    filled_length = int(length * display_percentage / 100)
    bar = '▇' * filled_length + '░' * (length - filled_length)
    return f"[{bar}] {percentage}%"


def get_dynamic_status_text(consumed: int, target: int, nutrient_type: str) -> str:
    """Генерирует динамический текстовый статус для макронутриента."""
    if target <= 0:
        return ""

    percentage = int((consumed / target) * 100)
    remaining = target - consumed
    diff_text = f"{abs(int(remaining))}"

    if nutrient_type == "calories":
        if percentage < 30:
            return f"Осталось: {diff_text} ккал | Начало положено! 🏁"
        elif 30 <= percentage < 60:
            return f"Осталось: {diff_text} ккал | Разгон взят, полёт нормальный ✈️"
        elif 60 <= percentage < 90:
            return f"Осталось: {diff_text} ккал | Почти у цели, аккуратнее с перекусами 🧐"
        elif 90 <= percentage <= 105:
            prefix = "Осталось" if remaining >= 0 else "Перебор"
            return f"{prefix}: {diff_text} ккал | Идеальное попадание! 🎯 Ювелирная точность."
        elif 105 < percentage <= 115:
            return f"Перебор: {diff_text} ккал | Чуть выше нормы, не критично 🤏"
        else: # Свыше 115%
            return f"Перебор: {diff_text} ккал | Финита ля комедия...Пухляшь 💀"

    if nutrient_type == "proteins":
        if consumed < target:
            return f"Осталось: {remaining} | Нужно поднажать 🥩"
        else:
            return f"Сверх нормы:  +{abs(remaining)} | Мощно! Ля ты машина 💪"

    if nutrient_type == "fats":
        if percentage > 110:
            return f"Перебор: {abs(remaining)} | Хорош жрать! Стапэ 🔴"
        elif percentage > 100:
            return f"Перебор: {abs(remaining)} | Тормози 🟡"
        elif percentage < 80:
             return f"Осталось: {remaining} | Баки пусты!"
        else:
            return f"Осталось: {remaining} | В норме"

    if nutrient_type == "carbs":
        if percentage > 110:
            return f"Перебор: {abs(remaining)} | Хорош жрать! Стапэ 🔴"
        elif percentage > 100:
            return f"Перебор: {abs(remaining)} | Тормози 🟡"
        else:
            return f"Осталось: {remaining} | Замори червячка"

    return ""


async def show_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return

    target = get_adjusted_target(user_id)
    if not target:
        await update.effective_message.reply_text("Сначала нужно настроить профиль с помощью /profile")
        return

    summary = get_daily_summary(user_id)
    
    consumed_c = int(summary['total_calories'])
    target_c = int(target['target_calories'])
    percent_c = int((consumed_c / target_c) * 100) if target_c > 0 else 0
    
    consumed_p = int(summary['total_proteins'])
    target_p = int(target['target_proteins'])
    percent_p = int((consumed_p / target_p) * 100) if target_p > 0 else 0

    consumed_f = int(summary['total_fats'])
    target_f = int(target['target_fats'])
    percent_f = int((consumed_f / target_f) * 100) if target_f > 0 else 0

    consumed_carb = int(summary['total_carbs'])
    target_carb = int(target['target_carbs'])
    percent_carb = int((consumed_carb / target_carb) * 100) if target_carb > 0 else 0

    text = f"""<b>📊 Статус на сегодня</b>

🔥 <b>Калории:</b> <code>{consumed_c} / {target_c}</code>
<code>{html.escape(create_progress_bar(percent_c))}</code>
{html.escape(get_dynamic_status_text(consumed_c, target_c, "calories"))}

🥩 <b>Белки:</b> <code>{consumed_p} / {target_p}</code>
<code>{html.escape(create_progress_bar(percent_p))}</code>
{html.escape(get_dynamic_status_text(consumed_p, target_p, "proteins"))}

🥑 <b>Жиры:</b> <code>{consumed_f} / {target_f}</code>
<code>{html.escape(create_progress_bar(percent_f))}</code>
{html.escape(get_dynamic_status_text(consumed_f, target_f, "fats"))}

🍞 <b>Углеводы:</b> <code>{consumed_carb} / {target_carb}</code>
<code>{html.escape(create_progress_bar(percent_carb))}</code>
{html.escape(get_dynamic_status_text(consumed_carb, target_carb, "carbs"))}
"""
    # Отправляем новое сообщение, а не редактируем текущее
    await update.effective_message.reply_text(text, reply_markup=get_nutrition_inline_keyboard(), parse_mode="HTML")


async def handle_nutrition_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles photo with nutrition trigger words."""
    user_id = update.effective_user.id
    message = update.message
    raw_text = message.caption or ""
    photo = message.photo[-1]

    profile = get_user_profile(user_id)
    if not profile:
        await message.reply_text("Сначала нужно настроить профиль с помощью /profile")
        return
    
    status_msg = await message.reply_text("🥗 Анализирую фото и считаю КБЖУ...")

    photo_file = await photo.get_file()
    image_data = await photo_file.download_as_bytearray()
    
    from InspectorAI.handlers.state import user_selected_nutrition_model
    from InspectorAI.handlers.base import parse_llm_json, format_meal_data_for_display
    
    model_to_use = user_selected_nutrition_model.get(user_id)
    llm_response, used_model_path = await process_llm(
        update, context, raw_text, mode="nutrition", image_data=image_data,
        selected_model=model_to_use,
        suppress_direct_reply=True
    )

    meal_data = parse_llm_json(llm_response)

    if meal_data:
        context.user_data['confirm_meal'] = meal_data
        formatted_text = format_meal_data_for_display(meal_data, model_name=used_model_path)
        keyboard = [
            [
                InlineKeyboardButton("✅ Сохранить", callback_data="confirm_meal_save"),
                InlineKeyboardButton("❌ Отмена", callback_data="confirm_meal_cancel")
            ]
        ]
        await status_msg.edit_text(f"{formatted_text}\n\nСохранить этот прием пищи?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
    else:
        await status_msg.edit_text("Не удалось распознать КБЖУ в ответе. Попробуйте еще раз.")


async def confirm_meal_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the 'Save' or 'Cancel' button for a meal."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data

    if data == "confirm_meal_save":
        meal_data = context.user_data.pop('confirm_meal', None)
        if meal_data:
            add_food_log(user_id, meal_data)
            await query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_message(chat_id=query.message.chat_id, text="✅ Прием пищи сохранен!")
            # Show updated status
            await show_status(update, context)
        else:
            await query.edit_message_text("❌ Не удалось найти данные о еде для сохранения.")
        return

    if data == "confirm_meal_cancel":
        context.user_data.pop('confirm_meal', None)
        await query.edit_message_text("❌ Операция отменена.")
        return


async def handle_nutrition_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Handles callbacks related to nutrition menu."""
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    if not data.startswith("nutrition_") and data != "main_menu":
        return False

    await query.answer()
    
    # Удаляем сообщение с инлайн-клавиатурой, чтобы очистить чат
    # Это нужно делать только если мы не начинаем ConversationHandler, который сам отправит новое сообщение
    if data != "nutrition_profile":
        await query.message.delete()

    if data == "nutrition_status":
        await show_status(update, context)
    elif data == "nutrition_stats":
        await show_nutrition_stats(update, context)
    elif data == "nutrition_recipe":
        await get_recipe_suggestion(update, context)
    elif data == "nutrition_profile":
        # ConversationHandler будет запущен через entry_points
        pass
    elif data == "main_menu":
        # Отправляем новое сообщение с основной ReplyKeyboardMarkup
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Возвращаемся в главное меню!",
            reply_markup=get_main_keyboard()
        )
    
    return True