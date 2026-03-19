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

from full_stat_analyzer import get_data_for_full_analysis
from llm_service import process_llm
from config import SYSTEM_PROMPT_FULL_ANALYSIS
from handlers.state import authorized_users, user_selected_model
from handlers.base import get_main_keyboard
from nutrition import get_user_profile, update_user_profile

# Состояния для диалога
(
    ASK_PROFILE_UPDATE,
    HANDLE_PROFILE_INPUT,
    ASK_SLEEP,
    ASK_ACTIVITY,
    ASK_METABOLISM,
    PROCESS_ANALYSIS,
) = range(6)

# --- Функции диалога ---

async def start_full_stat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало диалога полного анализа."""
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text("Эта функция доступна только авторизованным пользователям.")
        return ConversationHandler.END

    context.user_data["full_analysis_extra"] = {}
    profile = get_user_profile(user_id)

    if not profile:
        await update.effective_message.reply_text("Сначала нужно создать профиль. Перейдите в меню 'Нутрициолог' -> 'Профиль'.")
        return ConversationHandler.END

    context.user_data["profile_data"] = profile
    profile_text = (
        f"Текущие данные профиля:\n"
        f"<b>Возраст:</b> {profile.get('age', 'не указан')}\n"
        f"<b>Рост:</b> {profile.get('height', 'не указан')} см\n"
        f"<b>Вес:</b> {profile.get('weight', 'не указан')} кг\n\n"
        "Данные верны?"
    )
    keyboard = [
        [
            InlineKeyboardButton("✅ Да, все верно", callback_data="profile_correct"),
            InlineKeyboardButton("✏️ Нет, обновить", callback_data="profile_update"),
        ]
    ]
    await update.effective_message.reply_text(
        text=profile_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML",
    )
    return ASK_PROFILE_UPDATE


async def handle_profile_update_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора пользователя по обновлению профиля."""
    query = update.callback_query
    await query.answer()

    if query.data == "profile_correct":
        await query.edit_message_text("Отлично! Теперь несколько необязательных вопросов для более точного анализа.")
        return await ask_next_question(update, context, "sleep")

    elif query.data == "profile_update":
        await query.edit_message_text(
            "Введите новые данные в формате: <b>Возраст, Рост, Вес</b>\n"
            "Например: 30, 180, 75",
            parse_mode="HTML",
        )
        return HANDLE_PROFILE_INPUT


async def handle_profile_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получение и обновление данных профиля."""
    user_id = update.effective_user.id
    text = update.effective_message.text
    match = re.match(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+(\.\d+)?)\s*$", text)

    if not match:
        await update.effective_message.reply_text(
            "Неверный формат. Пожалуйста, введите данные как в примере: <b>30, 180, 75</b>",
            parse_mode="HTML",
        )
        return HANDLE_PROFILE_INPUT

    age, height, weight = int(match.group(1)), int(match.group(2)), float(match.group(3))
    update_user_profile(user_id, age=age, height=height, weight=weight)
    
    await update.effective_message.reply_text("Профиль успешно обновлен!")
    return await ask_next_question(update, context, "sleep")


async def ask_next_question(update: Update, context: ContextTypes.DEFAULT_TYPE, question_key: str) -> int:
    """Задает следующий вопрос из серии."""
    questions = {
        "sleep": ("😴 Введите ваше среднее время сна за последний месяц (в часах, например: 7.5)", ASK_SLEEP),
        "activity": ("🏃‍♂️ Введите средний калораж активности за день (например: 300)", ASK_ACTIVITY),
        "metabolism": ("🔥 Введите ваш обмен веществ в состоянии покоя (BMR) в ккал (например: 1800)", ASK_METABOLISM),
    }

    if question_key not in questions:
        return await process_analysis(update, context)

    text, state = questions[question_key]
    keyboard = [[InlineKeyboardButton("Пропустить", callback_data=f"skip_{question_key}")]]
    
    # Если это первый вопрос, отправляем новое сообщение. Иначе - редактируем.
    if update.callback_query:
        await update.callback_query.edit_message_text(
            text=text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
         await update.effective_message.reply_text(
            text=text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    return state


async def handle_sleep(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ответа про сон."""
    text = update.effective_message.text
    try:
        sleep_hours = float(text.replace(",", "."))
        context.user_data["full_analysis_extra"]["avg_sleep_hours"] = sleep_hours
        await update.effective_message.delete()
    except (ValueError, TypeError):
        await update.effective_message.reply_text("Неверный формат. Введите число, например: 7.5")
        return ASK_SLEEP
    
    return await ask_next_question(update, context, "activity")


async def handle_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ответа про активность."""
    text = update.effective_message.text
    try:
        activity_kcal = int(text)
        context.user_data["full_analysis_extra"]["avg_activity_kcal"] = activity_kcal
        await update.effective_message.delete()
    except (ValueError, TypeError):
        await update.effective_message.reply_text("Неверный формат. Введите целое число, например: 300")
        return ASK_ACTIVITY

    return await ask_next_question(update, context, "metabolism")


async def handle_metabolism(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ответа про метаболизм."""
    text = update.effective_message.text
    try:
        bmr_kcal = int(text)
        context.user_data["full_analysis_extra"]["bmr_kcal"] = bmr_kcal
        await update.effective_message.delete()
    except (ValueError, TypeError):
        await update.effective_message.reply_text("Неверный формат. Введите целое число, например: 1800")
        return ASK_METABOLISM

    # Это был последний вопрос, переходим к анализу
    # Нужно использовать effective_chat, так как последнее сообщение было удалено
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Спасибо! Начинаю анализ...")
    return await process_analysis(update, context)


async def handle_skip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка пропуска вопроса."""
    query = update.callback_query
    await query.answer()
    
    question_key = query.data.split("_")[1]
    next_questions = {"sleep": "activity", "activity": "metabolism", "metabolism": None}
    
    next_key = next_questions.get(question_key)
    if next_key:
        return await ask_next_question(update, context, next_key)
    else:
        # Пропущен последний вопрос, запускаем анализ
        await query.edit_message_text("Спасибо! Начинаю анализ...")
        return await process_analysis(update, context)


async def process_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Финальный шаг: сбор всех данных и запуск LLM."""
    user_id = update.effective_user.id
    
    # Сообщение о начале анализа
    if update.callback_query:
        # Если последний был колбэк, то сообщение уже отредактировано
        pass
    else:
        # Если был текстовый ввод, сообщение новое
        await update.effective_message.reply_text("⏳ Собираю и анализирую ваши данные... Это может занять до минуты.")

    try:
        analysis_data = await get_data_for_full_analysis(user_id)
        
        # Добавляем опциональные данные, собранные в диалоге
        if context.user_data.get("full_analysis_extra"):
            analysis_data["user_additional_data"] = context.user_data["full_analysis_extra"]

        if not analysis_data.get('nutrition_summary') and not analysis_data.get('workout_summary', {}).get('progression_analysis'):
            await context.bot.send_message(
                chat_id=user_id,
                text="Недостаточно данных для анализа. Пожалуйста, занесите в дневник хотя бы несколько приемов пищи и тренировок.",
                reply_markup=get_main_keyboard()
            )
            return ConversationHandler.END

        # --- Формирование промпта для ИИ ---
        prompt_parts = []

        # 1. Основные данные
        prompt_parts.append("Проанализируй следующие данные моего клиента:")
        prompt_parts.append(f"```json\n{json.dumps(analysis_data, indent=2, ensure_ascii=False)}\n```")

        # 2. Правила коммуникации
        prompt_parts.append("\n**ПРАВИЛА КОММУНИКАЦИИ:**")
        prompt_parts.append("- Говори на человеческом языке, избегай сухих цифр и терминов, если их можно объяснить простыми словами.")
        prompt_parts.append("- Используй метафоры и аналогии, чтобы объяснить сложные концепции (например, 'запас прочности', 'фундамент для рекорда').")
        prompt_parts.append("- Будь эмпатичным и поддерживающим. Цель - мотивировать, а не критиковать.")
        prompt_parts.append("- Если данные отсутствуют или неполны, указывай на это и давай рекомендации с осторожностью.")
        prompt_parts.append("- Структурируй ответ по пунктам, как указано в 'Твоя задача'.")

        # 3. Уровень уверенности (на основе введенных пользователем данных)
        confidence_notes = []
        if not analysis_data.get("user_additional_data", {}).get("avg_sleep_hours"):
            confidence_notes.append("- Данные о среднем времени сна отсутствуют.")
        if not analysis_data.get("user_additional_data", {}).get("avg_activity_kcal"):
            confidence_notes.append("- Данные о среднем калораже активности отсутствуют.")
        if not analysis_data.get("user_additional_data", {}).get("bmr_kcal"):
            confidence_notes.append("- Данные о базальном метаболизме (BMR) введены пользователем, но не проверены.")
        
        if confidence_notes:
            prompt_parts.append("\n**УРОВЕНЬ УВЕРЕННОСТИ В АНАЛИЗЕ ДОПОЛНИТЕЛЬНЫХ ДАННЫХ:**")
            prompt_parts.extend(confidence_notes)
            prompt_parts.append("Инструкция: Давай рекомендации в областях с отсутствующими/непроверенными данными с осторожностью. Например: 'Поскольку у меня нет точных данных о твоем сне, я не могу сделать однозначных выводов, но убедись, что ты спишь достаточно...'")

        # 4. Специальные инструкции для ИИ
        prompt_parts.append("\n**СПЕЦИАЛЬНЫЕ ИНСТРУКЦИИ:**")
        if analysis_data.get("workout_summary", {}).get("overtrain_risk"):
            prompt_parts.append("- **ВЫСШИЙ ПРИОРИТЕТ:** Обнаружен риск перетренированности. Твоя первая и главная задача — выразить обеспокоенность и мягко спросить о восстановлении (сон, стресс, питание). НЕ СОВЕТУЙ 'работать усерднее'. Предположи, что может потребоваться неделя отдыха или снижение нагрузки. Сфокусируйся на заботе о здоровье.")
        
        if analysis_data.get("user_profile", {}).get("calculated_bmr") and analysis_data.get("user_additional_data", {}).get("bmr_kcal"):
            prompt_parts.append(f"- Сравни введенный пользователем BMR ({analysis_data['user_additional_data']['bmr_kcal']} ккал) с расчетным ({analysis_data['user_profile']['calculated_bmr']} ккал). Укажи на возможные несоответствия и их влияние на цели.")

        # 5. Структура ответа
        prompt_parts.append("\n**Твоя задача:**")
        prompt_parts.append("Дай комплексный анализ и рекомендации, следуя этой структуре:")
        prompt_parts.append("1.  **Общий вердикт:** Краткое резюме (2-3 предложения), соответствует ли текущий прогресс цели пользователя.")
        prompt_parts.append("2.  **Анализ питания:** Оцени, соответствует ли КБЖУ цели и весу. Выяви возможные проблемы (например, недостаток белка).")
        prompt_parts.append("3.  **Анализ тренировок:**")
        prompt_parts.append("    - Оцени общий прогресс в ключевых упражнениях, используя 'overall_strength_trend'.")
        prompt_parts.append("    - Для каждого упражнения из 'progression_analysis' проанализируй:")
        prompt_parts.append("        - 'pr_status' (Индекс 'Свежего Рекорда'): переведи в понятный язык (например, 'давно не было рекордов').")
        prompt_parts.append("        - 'trend_analysis' (Скользящее среднее): объясни динамику (растет, падает, плато).")
        prompt_parts.append("        - 'efficiency_analysis' (Матрица 'Объем / Интенсивность'): объясни, что означает 'Efficiency Score' и как он соотносится с 1ПМ (например, 'запас прочности растет').")
        prompt_parts.append("    - Сделай выводы по 'overtrain_risk', если он активен, следуя специальным инструкциям.")
        prompt_parts.append("4.  **Анализ доп. данных:** Если есть 'user_additional_data', кратко проанализируй их (сон, активность, BMR), учитывая 'УРОВЕНЬ УВЕРЕННОСТИ'.")
        prompt_parts.append("5.  **Ключевые рекомендации:** 3-5 самых важных шагов для пользователя, сформулированных эмпатично и мотивирующе.")

        final_prompt = "\n".join(prompt_parts)

        selected_model = user_selected_model.get(user_id)
        await process_llm(
            update, context, final_prompt,
            selected_model=selected_model,
            mode="chat",
            system_prompt_override=SYSTEM_PROMPT_FULL_ANALYSIS
        )

    except Exception as e:
        logging.error(f"Ошибка при создании полного анализа для user_id {user_id}: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=user_id,
            text="Произошла ошибка при подготовке анализа. Попробуйте позже.",
            reply_markup=get_main_keyboard()
        )
    finally:
        # Очистка временных данных
        if "full_analysis_extra" in context.user_data:
            del context.user_data["full_analysis_extra"]
        if "profile_data" in context.user_data:
            del context.user_data["profile_data"]
            
    return ConversationHandler.END


async def cancel_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена диалога."""
    await update.effective_message.reply_text("Анализ отменен.", reply_markup=get_main_keyboard())
    # Очистка временных данных
    if "full_analysis_extra" in context.user_data:
        del context.user_data["full_analysis_extra"]
    if "profile_data" in context.user_data:
        del context.user_data["profile_data"]
    return ConversationHandler.END


# Создание ConversationHandler
full_stat_conversation_handler = ConversationHandler(
    entry_points=[MessageHandler(filters.Regex('^📊 Полный анализ$'), start_full_stat)],
    states={
        ASK_PROFILE_UPDATE: [CallbackQueryHandler(handle_profile_update_choice)],
        HANDLE_PROFILE_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_profile_input)],
        ASK_SLEEP: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_sleep),
            CallbackQueryHandler(handle_skip, pattern="^skip_sleep$"),
        ],
        ASK_ACTIVITY: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_activity),
            CallbackQueryHandler(handle_skip, pattern="^skip_activity$"),
        ],
        ASK_METABOLISM: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_metabolism),
            CallbackQueryHandler(handle_skip, pattern="^skip_metabolism$"),
        ],
    },
    fallbacks=[CommandHandler("cancel", cancel_analysis)],
    per_user=True,
    per_chat=True,
)
