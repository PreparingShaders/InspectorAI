import logging
import json
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler,
    MessageHandler,
    CommandHandler,
    filters,
)

from full_stat_analyzer import get_data_for_full_analysis, _calculate_bmr
from llm_service import process_llm
from config import SYSTEM_PROMPT_FULL_ANALYSIS
from handlers.state import authorized_users, user_selected_model
from handlers.base import get_main_keyboard
from nutrition import get_user_profile, update_user_profile, calculate_nutrition_plan

# Состояния для диалога
(
    ASK_PROFILE_UPDATE,
    HANDLE_PROFILE_INPUT,
    ASK_SLEEP,
    ASK_ACTIVITY,
    ASK_METABOLISM,
    PROCESS_ANALYSIS,
) = range(6)

# --- Вспомогательные функции ---

def _validate_and_get_value(text: str, min_val: float, max_val: float, error_msg: str) -> (float | None, str | None):
    try:
        value = float(text.replace(",", "."))
        if not (min_val <= value <= max_val):
            return None, f"Значение должно быть в диапазоне от {min_val} до {max_val}."
        return value, None
    except (ValueError, TypeError):
        return None, error_msg

# --- Функции диалога ---

async def start_full_stat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text("Эта функция доступна только авторизованным пользователям.")
        return ConversationHandler.END

    context.user_data["full_analysis_extra"] = {}
    profile = get_user_profile(user_id)

    if not profile or not all(k in profile for k in ['age', 'height', 'weight']):
        await update.effective_message.reply_text("Сначала нужно создать или полностью заполнить профиль (возраст, рост, вес). Перейдите в меню 'Нутрициолог' -> 'Профиль'.")
        return ConversationHandler.END

    context.user_data["profile_data"] = profile
    profile_text = (
        f"Текущие данные профиля:\n"
        f"<b>Возраст:</b> {profile['age']}\n"
        f"<b>Рост:</b> {profile['height']} см\n"
        f"<b>Вес:</b> {profile['weight']} кг\n\n"
        "Данные верны?"
    )
    keyboard = [[InlineKeyboardButton("✅ Да", callback_data="profile_correct"), InlineKeyboardButton("✏️ Нет, обновить", callback_data="profile_update")]]
    await update.effective_message.reply_text(text=profile_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
    return ASK_PROFILE_UPDATE

async def handle_profile_update_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "profile_correct":
        await query.edit_message_text("Отлично! Теперь несколько необязательных вопросов.")
        return await ask_next_question(update, context, "sleep")
    elif query.data == "profile_update":
        await query.edit_message_text("Введите новые данные в формате: <b>Возраст, Рост, Вес</b>\nНапример: 30, 180, 75", parse_mode="HTML")
        return HANDLE_PROFILE_INPUT

async def handle_profile_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    text = update.effective_message.text
    match = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+(\.\d+)?)\s*$", text)
    if not match:
        await update.effective_message.reply_text("Неверный формат. Пожалуйста, введите данные как в примере: <b>30, 180, 75</b>", parse_mode="HTML")
        return HANDLE_PROFILE_INPUT

    age, height, weight = int(match.group(1)), int(match.group(2)), float(match.group(3))
    
    # --- ИСПРАВЛЕНИЕ ОШИБКИ ---
    profile_data = get_user_profile(user_id)
    profile_data['age'] = age
    profile_data['height'] = height
    profile_data['weight'] = weight
    
    # Пересчитываем КБЖУ на основе новых данных
    new_targets = calculate_nutrition_plan(profile_data)
    profile_data.update(new_targets)
    
    update_user_profile(user_id, profile_data)
    # -------------------------

    context.user_data["profile_data"] = get_user_profile(user_id) # Обновляем данные в контексте
    await update.effective_message.reply_text("Профиль успешно обновлен!")
    return await ask_next_question(update, context, "sleep")

async def ask_next_question(update: Update, context: ContextTypes.DEFAULT_TYPE, question_key: str) -> int:
    questions = {
        "sleep": ("😴 Введите среднее время сна (в часах)", ASK_SLEEP, "Рассчитать среднее (8ч)"),
        "activity": ("🏃‍♂️ Введите средний калораж активности за день", ASK_ACTIVITY, "Рассчитать среднее (400)"),
        "metabolism": ("🔥 Введите ваш обмен веществ в покое (BMR)", ASK_METABOLISM, "Рассчитать по формуле"),
    }
    if question_key not in questions:
        return await process_analysis(update, context)

    text, state, btn_text = questions[question_key]
    keyboard = [[InlineKeyboardButton(btn_text, callback_data=f"calc_{question_key}")]]
    
    msg_sender = update.callback_query.edit_message_text if update.callback_query else update.effective_message.reply_text
    await msg_sender(text=text, reply_markup=InlineKeyboardMarkup(keyboard))
    return state

async def handle_sleep(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    value, error = _validate_and_get_value(update.effective_message.text, 1, 16, "Неверный формат. Введите число, например: 7.5")
    if error:
        await update.effective_message.reply_text(error)
        return ASK_SLEEP
    context.user_data["full_analysis_extra"]["avg_sleep_hours"] = value
    # Сообщение больше не удаляется
    return await ask_next_question(update, context, "activity")

async def handle_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    value, error = _validate_and_get_value(update.effective_message.text, 0, 3000, "Неверный формат. Введите целое число, например: 300")
    if error:
        await update.effective_message.reply_text(error)
        return ASK_ACTIVITY
    context.user_data["full_analysis_extra"]["avg_activity_kcal"] = value
    # Сообщение больше не удаляется
    return await ask_next_question(update, context, "metabolism")

async def handle_metabolism(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    value, error = _validate_and_get_value(update.effective_message.text, 1000, 4000, "Неверный формат. Введите целое число, например: 1800")
    if error:
        await update.effective_message.reply_text(error)
        return ASK_METABOLISM
    context.user_data["full_analysis_extra"]["bmr_kcal"] = value
    # Сообщение больше не удаляется
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Спасибо! Начинаю анализ...")
    return await process_analysis(update, context)

async def handle_calculation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    question_key = query.data.split("_")[1]
    
    extra_data = context.user_data["full_analysis_extra"]
    profile = context.user_data["profile_data"]
    
    if question_key == "sleep":
        extra_data["avg_sleep_hours"] = 8.0
        return await ask_next_question(update, context, "activity")
    elif question_key == "activity":
        extra_data["avg_activity_kcal"] = 400
        return await ask_next_question(update, context, "metabolism")
    elif question_key == "metabolism":
        extra_data["bmr_kcal"] = round(_calculate_bmr(profile['weight'], profile['height'], profile['age']))
        await query.edit_message_text("Спасибо! Начинаю анализ...")
        return await process_analysis(update, context)
    return ConversationHandler.END

async def process_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if update.callback_query: pass
    else: await update.effective_message.reply_text("⏳ Собираю и анализирую ваши данные... Это может занять до минуты.")

    try:
        analysis_data = await get_data_for_full_analysis(user_id)
        analysis_data["user_additional_data"] = context.user_data.get("full_analysis_extra", {})

        if not analysis_data.get('nutrition_summary') and not analysis_data.get('workout_summary', {}).get('progression_analysis'):
            await context.bot.send_message(user_id, "Недостаточно данных для анализа.", reply_markup=get_main_keyboard())
            return ConversationHandler.END

        prompt = f"""
Проанализируй данные моего клиента.

**ПРАВИЛА КОММУНИКАЦИИ:**
- Говори на человеческом языке, используя метафоры ('запас прочности', 'фундамент для рекорда').
- Будь эмпатичным и поддерживающим. Цель - мотивировать.
- Структурируй ответ по пунктам, как указано в 'Твоя задача'.

**СПЕЦИАЛЬНЫЕ ИНСТРУКЦИИ:**
- Если 'overtrain_risk' == true, твоя первая и главная задача — выразить обеспокоенность и мягко спросить о восстановлении (сон, стресс). НЕ СОВЕТУЙ 'работать усерднее'.

**ДАННЫЕ КЛИЕНТА:**
```json
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}
```

**Твоя задача:**
Дай комплексный анализ и рекомендации по этой структуре:
1.  **Общий вердикт:** Краткое резюме (2-3 предложения) о прогрессе к цели.
2.  **Анализ питания:** Оценка КБЖУ, возможные проблемы.
3.  **Анализ тренировок:**
    - Оцени общий прогресс по 'overall_strength_trend'.
    - Для каждого упражнения из 'progression_analysis' проанализируй 'pr_status', 'trend_analysis', 'efficiency_analysis', объясняя их на простом языке.
    - Сделай выводы по 'overtrain_risk', если он активен.
4.  **Анализ доп. данных:** Кратко проанализируй 'user_additional_data' (сон, активность, BMR).
5.  **Ключевые рекомендации:** 3-5 самых важных шагов, сформулированных эмпатично.
"""
        await process_llm(
            update, context, prompt,
            selected_model=user_selected_model.get(user_id),
            mode="chat",
            system_prompt_override=SYSTEM_PROMPT_FULL_ANALYSIS
        )
    except Exception as e:
        logging.error(f"Ошибка при создании полного анализа для user_id {user_id}: {e}", exc_info=True)
        await context.bot.send_message(user_id, "Произошла ошибка при подготовке анализа.", reply_markup=get_main_keyboard())
    finally:
        for key in ["full_analysis_extra", "profile_data"]:
            if key in context.user_data: del context.user_data[key]
    return ConversationHandler.END

async def cancel_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_message.reply_text("Анализ отменен.", reply_markup=get_main_keyboard())
    for key in ["full_analysis_extra", "profile_data"]:
        if key in context.user_data: del context.user_data[key]
    return ConversationHandler.END

full_stat_conversation_handler = ConversationHandler(
    entry_points=[MessageHandler(filters.Regex('^📊 Полный анализ$'), start_full_stat)],
    states={
        ASK_PROFILE_UPDATE: [CallbackQueryHandler(handle_profile_update_choice)],
        HANDLE_PROFILE_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_profile_input)],
        ASK_SLEEP: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_sleep), CallbackQueryHandler(handle_calculation, pattern="^calc_sleep$")],
        ASK_ACTIVITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_activity), CallbackQueryHandler(handle_calculation, pattern="^calc_activity$")],
        ASK_METABOLISM: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_metabolism), CallbackQueryHandler(handle_calculation, pattern="^calc_metabolism$")],
    },
    fallbacks=[CommandHandler("cancel", cancel_analysis)],
    per_user=True,
    per_chat=True,
)
