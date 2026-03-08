#handlers
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='telegram.ext')

import re
import logging
import json
import html
from datetime import date, timedelta
from collections import defaultdict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton, error
from telegram.ext import ContextTypes, ConversationHandler, MessageHandler, filters, CommandHandler, CallbackQueryHandler

# ... (импорты)
from finance import (
    register_user, apply_expense,
    get_detailed_report, get_all_users_except, settle_debt
)
from llm_service import (
    update_model_mappings, current_free_or_models,
    GEMINI_MODEL_BY_ID, OPENROUTER_MODEL_BY_ID, process_llm, NUTRITION_MODEL_BY_ID
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
from utils import handle_voice_transcription, get_model_short_name, to_html
from config import (
    ALLOWED_USER_IDS, AUTH_QUESTION, TRIGGERS, CHECK_WORDS,
    GEMINI_MODELS, FINANCE_WORDS, NUTRITION_TRIGGERS, NUTRITION_MODELS
)

# ... (состояния)
authorized_users = set(ALLOWED_USER_IDS)
user_selected_model = {}
user_selected_nutrition_model = {}
user_selected_provider = {}
(PROFILE_GENDER, PROFILE_AGE, PROFILE_HEIGHT, PROFILE_WEIGHT,
 PROFILE_ACTIVITY, PROFILE_GOAL) = range(6)
(CHEAT_MEAL_INPUT, CHEAT_MEAL_CONFIRM) = range(6, 8)


def get_main_keyboard() -> ReplyKeyboardMarkup:
    keyboard = [
        [KeyboardButton("📊 Статус"), KeyboardButton("📈 Статистика")],
        [KeyboardButton("❓ Что съесть?"), KeyboardButton("🍩 Читмил")],
        [KeyboardButton("⚙️ Профиль"), KeyboardButton("🤖 Сменить модель")],
        [KeyboardButton("❌ Отмена")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def parse_llm_json(response_text: str) -> dict | None:
    """
    Гибко парсит JSON из ответа LLM, захватывая текст перед JSON как комментарий.
    """
    if not response_text:
        return None

    match = re.search(r'(.*?)```json\n(.*?)\n```', response_text, re.DOTALL)

    if match:
        comment_text = match.group(1).strip()
        json_text = match.group(2)
        try:
            data = json.loads(json_text)
            if comment_text:
                # Убираем to_html, чтобы передать текст "как есть"
                data['comment'] = comment_text
            return data
        except json.JSONDecodeError:
            logging.warning("Найден блок JSON, но не удалось его распарсить.")
            pass

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        logging.error(f"Не удалось распарсить как JSON весь текст: {response_text}")
        return None

def format_meal_data_for_display(meal_data: dict, model_name: str | None) -> str:
    """Форматирует данные о еде из JSON в красивый HTML-текст."""
    if not meal_data:
        return "Не удалось распознать данные о еде."

    comment = meal_data.get('comment', '')

    text = ""
    if model_name:
        short_model_name = get_model_short_name(model_name, "gemini" if "gemini" in model_name else "openrouter")
        text += f"🤖 <b>Модель:</b> <code>{html.escape(short_model_name)}</code>\n\n"

    if comment:
        # Убираем теги <i> и используем html.escape для безопасного вывода
        text += f"{html.escape(comment)}\n\n"

    text += (
        f"<b>📊 Оценка КБЖУ:</b>\n"
        f"🔥 Калории: <code>{meal_data.get('calories', 0)}</code>\n"
        f"🥩 Белки: <code>{meal_data.get('proteins', 0)} г</code>\n"
        f"🥑 Жиры: <code>{meal_data.get('fats', 0)} г</code>\n"
        f"🍞 Углеводы: <code>{meal_data.get('carbs', 0)} г</code>"
    )
    return text

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


async def cheat_meal_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return ConversationHandler.END
    await update.effective_message.reply_text("Пришли фото или подробное описание того, что ты съел, и я оценю масштаб трагедии.")
    return CHEAT_MEAL_INPUT

async def cheat_meal_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    message = update.message
    photo = message.photo[-1] if message.photo else None
    text_input = message.text or message.caption or ""
    if not photo and not text_input:
        await message.reply_text("Нужно что-то прислать: фото или текст. Попробуй еще раз или нажми /cancel.")
        return CHEAT_MEAL_INPUT

    status_msg = await message.reply_text("🍩 Оцениваю масштаб трагедии...")

    image_data = None
    if photo:
        photo_file = await photo.get_file()
        image_data = await photo_file.download_as_bytearray()
    
    model_to_use = user_selected_nutrition_model.get(user_id)
    llm_response, used_model_path = await process_llm(
        update, context, text_input, mode="nutrition", image_data=image_data,
        selected_model=model_to_use,
        suppress_direct_reply=True
    )

    meal_data = parse_llm_json(llm_response)

    if meal_data:
        context.user_data['cheat_meal_data'] = meal_data
        formatted_text = format_meal_data_for_display(meal_data, model_name=used_model_path)
        keyboard = [[InlineKeyboardButton("Да, я это съел", callback_data="confirm_cheat"), InlineKeyboardButton("Нет, отмена", callback_data="cancel_cheat")]]
        await status_msg.edit_text(
            f"{formatted_text}\n\nЗаписать этот прием пищи и скорректировать остаток на сегодня?",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
        return CHEAT_MEAL_CONFIRM
    else:
        await status_msg.edit_text("Не смог распознать КБЖУ. Попробуй описать блюдо подробнее или пришли другое фото. Для отмены нажми /cancel.")
        return CHEAT_MEAL_INPUT

async def cheat_meal_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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

async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    await update.effective_message.reply_text(
        'Действие отменено.', reply_markup=get_main_keyboard()
    )
    context.user_data.clear()
    return ConversationHandler.END

cheat_meal_handler = ConversationHandler(
    entry_points=[
        CommandHandler('cheatmeal', cheat_meal_start),
        MessageHandler(filters.Regex('^🍩 Читмил$'), cheat_meal_start)
    ],
    states={
        CHEAT_MEAL_INPUT: [MessageHandler((filters.TEXT & ~filters.COMMAND & ~filters.Regex('^❌ Отмена$')) | filters.PHOTO | filters.CAPTION, cheat_meal_input)],
        CHEAT_MEAL_CONFIRM: [CallbackQueryHandler(cheat_meal_confirm, pattern='^(confirm_cheat|cancel_cheat)$')]
    },
    fallbacks=[
        CommandHandler('cancel', cancel_conversation),
        MessageHandler(filters.Regex('^❌ Отмена$'), cancel_conversation)
    ],
)

async def profile_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return ConversationHandler.END
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.reply_text("Укажи свой пол:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Мужской", callback_data="male"), InlineKeyboardButton("Женский", callback_data="female")]]))
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
        await update.message.reply_text("Это не похоже на возраст. Пожалуйста, введи возраст числом. Для отмены введи /cancel или нажми кнопку '❌ Отмена'.")
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
        await update.message.reply_text("Это не похоже на рост. Пожалуйста, введи рост числом (в см). Для отмены введи /cancel или нажми кнопку '❌ Отмена'.")
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
        await update.message.reply_text("Это не похоже на вес. Пожалуйста, введи вес числом (в кг). Для отмены введи /cancel или нажми кнопку '❌ Отмена'.")
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
    await update.effective_message.reply_text(text, parse_mode="HTML")


profile_setup_handler = ConversationHandler(
    entry_points=[
        CommandHandler('profile', profile_start),
        MessageHandler(filters.Regex('^⚙️ Профиль$'), profile_start)
    ],
    states={
        PROFILE_GENDER: [CallbackQueryHandler(profile_gender, pattern='^(male|female)$')],
        PROFILE_AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex('^❌ Отмена$'), profile_age)],
        PROFILE_HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex('^❌ Отмена$'), profile_height)],
        PROFILE_WEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex('^❌ Отмена$'), profile_weight)],
        PROFILE_ACTIVITY: [CallbackQueryHandler(profile_activity, pattern=r'^1\.')],
        PROFILE_GOAL: [CallbackQueryHandler(profile_goal, pattern='^(weight_loss|recomposition|mass_gain)$')],
    },
    fallbacks=[
        CommandHandler('cancel', cancel_conversation),
        MessageHandler(filters.Regex('^❌ Отмена$'), cancel_conversation)
    ],
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in authorized_users:
        await update.effective_message.reply_text(
            "Клавиатура на месте! Чем займемся?",
            reply_markup=get_main_keyboard()
        )
    else:
        await update.effective_message.reply_text(AUTH_QUESTION)

async def show_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return

    update_model_mappings()
    keyboard = []

    def create_buttons(models, model_map, prefix, start_idx, selected_model_dict):
        buttons = []
        for i, model in enumerate(models):
            name = get_model_short_name(model, "gemini" if "gemini" in model else "openrouter")
            current_selection = selected_model_dict.get(user_id)
            is_selected = current_selection == model
            button_prefix = "✅ " if is_selected else ""
            callback_data = f"sel:{prefix}:{i + start_idx}"
            buttons.append(InlineKeyboardButton(f"{button_prefix}{name}", callback_data=callback_data))
        return [buttons[i:i + 2] for i in range(0, len(buttons), 2)]

    keyboard.append([InlineKeyboardButton("💬 Модели для общения:", callback_data="dummy")])
    keyboard.extend(create_buttons(current_free_or_models, OPENROUTER_MODEL_BY_ID, "o", 100, user_selected_model))
    keyboard.extend(create_buttons(GEMINI_MODELS, GEMINI_MODEL_BY_ID, "g", 0, user_selected_model))
    
    keyboard.append([InlineKeyboardButton("──────────────", callback_data="dummy")])
    
    keyboard.append([InlineKeyboardButton("🥗 Модели для анализа еды:", callback_data="dummy")])
    keyboard.extend(create_buttons(NUTRITION_MODELS, NUTRITION_MODEL_BY_ID, "n", 200, user_selected_nutrition_model))

    keyboard.append([InlineKeyboardButton("🤖 Автовыбор для чата", callback_data="sel:auto_chat")])
    keyboard.append([InlineKeyboardButton("🥗 Автовыбор для еды", callback_data="sel:auto_nutrition")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    text = "<b>Выбор модели ИИ</b>"
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.effective_message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.voice: return
    
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await message.reply_text(AUTH_QUESTION)
        return

    text = await handle_voice_transcription(message)
    if text:
        await message.reply_text(f"🎤 <b>Распознано:</b>\n<i>{html.escape(text)}</i>", parse_mode="HTML")
        if update.effective_chat.type in ["group", "supergroup"]:
            await handle_group(update, context, voice_text=text)
        else:
            await handle_private(update, context, voice_text=text)

async def handle_private(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    user_id = update.effective_user.id
    message = update.message
    if not message: return
    
    if user_id not in authorized_users:
        await message.reply_text(AUTH_QUESTION)
        return

    raw_text = voice_text or message.text or message.caption or ""
    photo = message.photo[-1] if message.photo else None
    
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
        
        status_msg = await message.reply_text("🥗 Анализирую фото и считаю КБЖУ...")

        photo_file = await photo.get_file()
        image_data = await photo_file.download_as_bytearray()
        
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
        return
    else:
        if message.reply_to_message:
            context_text = message.reply_to_message.text or message.reply_to_message.caption or ""
            final_prompt = f"КОНТЕКСТ ПРЕДЫДУЩЕГО СООБЩЕНИЯ:\n{context_text}\n\nТЕКУЩИЙ ЗАПРОС:\n{raw_text}"

    if final_prompt:
        await process_llm(
            update, context, final_prompt,
            selected_model=user_selected_model.get(user_id),
            mode=mode
        )

async def handle_group(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
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
            await message.reply_text(f"💰 Сколько возвращаем для <b>{html.escape(creditor_name)}</b>?", parse_mode="HTML")
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
        selected_model=user_selected_model.get(update.effective_user.id),
        mode=mode
    )

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    if user_id not in authorized_users:
        await query.answer(text=AUTH_QUESTION, show_alert=True)
        return
    
    await query.answer()

    if data == "confirm_meal_save":
        meal_data = context.user_data.pop('confirm_meal', None)
        if meal_data:
            add_food_log(user_id, meal_data)
            # Убираем кнопки с исходного сообщения, оставляя его в чате
            await query.edit_message_reply_markup(reply_markup=None)
            # Отправляем новое сообщение с подтверждением
            await context.bot.send_message(chat_id=query.message.chat_id, text="✅ Прием пищи сохранен!")
            # Показываем обновленный статус отдельным сообщением
            await show_status(update, context)
        else:
            await query.edit_message_text("❌ Не удалось найти данные о еде для сохранения. Возможно, вы уже сохранили его.")
        return

    if data == "confirm_meal_cancel":
        context.user_data.pop('confirm_meal', None)
        await query.edit_message_text("❌ Операция отменена.")
        return

    if data.startswith("sel:"):
        parts = data.split(":")
        action = parts[1]

        if action == "auto_chat":
            user_selected_model.pop(user_id, None)
            user_selected_provider.pop(user_id, None)
            await query.edit_message_text("🤖 Автовыбор для чата включен.")
            return
        if action == "auto_nutrition":
            user_selected_nutrition_model.pop(user_id, None)
            await query.edit_message_text("🥗 Автовыбор для еды включен.")
            return

        prov_code, idx = parts[1], parts[2]
        
        model_path, provider, model_dict, name_prefix = None, None, None, ""

        if prov_code == 'g':
            model_path = GEMINI_MODEL_BY_ID.get(idx)
            provider = "gemini"
            model_dict = user_selected_model
            name_prefix = "💬"
        elif prov_code == 'o':
            model_path = OPENROUTER_MODEL_BY_ID.get(idx)
            provider = "openrouter"
            model_dict = user_selected_model
            name_prefix = "💬"
        elif prov_code == 'n':
            model_path = NUTRITION_MODEL_BY_ID.get(idx)
            provider = "gemini" if "gemini" in model_path else "openrouter"
            model_dict = user_selected_nutrition_model
            name_prefix = "🥗"

        if model_path and model_dict is not None:
            model_dict[user_id] = model_path
            if prov_code in ['g', 'o']:
                 user_selected_provider[user_id] = provider
            name = get_model_short_name(model_path, provider)
            await query.edit_message_text(f"{name_prefix} Выбрана модель: <b>{html.escape(name)}</b>", parse_mode="HTML")
        
    elif data.startswith("f_toggle:"):
        uid = data.split(":")[1]
        participants = context.user_data.get('tmp_participants', [])
        if uid in participants:
            participants.remove(uid)
        else:
            participants.append(uid)
        context.user_data['tmp_participants'] = participants
        await send_participant_selector(update, context)
    elif data.startswith("pay_select:"):
        creditor_id = data.split(":")[1]
        from finance import load_db
        db = load_db()
        creditor_name = db.get(creditor_id, {}).get("name", "Друг")
        context.user_data.update({
            'finance_state': 'WAITING_PAYBACK_AMOUNT',
            'tmp_creditor_id': creditor_id
        })
        await query.edit_message_text(f"💰 Сколько возвращаем для <b>{html.escape(creditor_name)}</b>?", parse_mode="HTML")
    elif data == "f_confirm":
        participants = context.user_data.get('tmp_participants')
        if not participants:
            await query.answer("Выбери хотя бы одного!", show_alert=True)
            return
        share = apply_expense(user_id, participants, context.user_data.get('tmp_amount'))
        await query.edit_message_text(f"✅ Записано!\nКаждый (включая тебя) должен по {share:.2f} р.")
        context.user_data.clear()
    elif data == "f_cancel":
        context.user_data.clear()
        await query.edit_message_text("❌ Расчет отменен.")
    elif data == "open_menu":
        await show_model_selection(update, context)

async def send_participant_selector(update: Update, context: ContextTypes.DEFAULT_TYPE):
    payer_id = update.effective_user.id
    chat_id = update.effective_chat.id
    amount = context.user_data.get('tmp_amount', 0)
    
    selected = context.user_data.get('tmp_participants', [])
    all_known_users = get_all_users_except(payer_id)
    keyboard = []
    
    for uid, name in all_known_users.items():
        try:
            member = await context.bot.get_chat_member(chat_id, int(uid))
            if member.status not in ['left', 'kicked']:
                label = f"✅ {name}" if uid in selected else name
                keyboard.append([InlineKeyboardButton(label, callback_data=f"f_toggle:{uid}")])
        except error.BadRequest as e:
            if "participant_id_invalid" in str(e).lower():
                continue
            logging.error(f"Ошибка проверки юзера {uid} в чате {chat_id}: {e}")
        except Exception as e:
            logging.error(f"Неожиданная ошибка проверки юзера {uid}: {e}")
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
        logging.error(f"Ошибка отправки кнопок выбора участников: {e}")