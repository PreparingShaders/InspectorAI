import logging
import json
from telegram import Update
from telegram.ext import ContextTypes

from full_stat_analyzer import get_data_for_full_analysis
from llm_service import process_llm
from config import SYSTEM_PROMPT_FULL_ANALYSIS
from handlers.state import authorized_users
# Исправленный импорт
from handlers.base import get_main_keyboard

async def handle_full_stat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик для команды "Полный анализ".
    Собирает данные, отправляет их в ИИ и возвращает пользователю анализ.
    """
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text("Эта функция доступна только авторизованным пользователям.")
        return

    await update.effective_message.reply_text("⏳ Собираю и анализирую ваши данные за последние 2 месяца... Это может занять до минуты.")

    try:
        # 1. Сбор и структурирование данных
        analysis_data = await get_data_for_full_analysis(user_id)

        # Проверяем, есть ли хоть какие-то данные для анализа
        if not analysis_data.get('nutrition_summary') and not analysis_data.get('workout_summary', {}).get('progression_analysis'):
            await update.effective_message.reply_text(
                "Недостаточно данных для анализа. Пожалуйста, занесите в дневник хотя бы несколько приемов пищи и тренировок.",
                reply_markup=get_main_keyboard()
            )
            return

        # 2. Формирование промпта для ИИ
        prompt = f"""
Проанализируй следующие данные моего клиента.

Вот данные в формате JSON:
```json
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}
```

**Твоя задача:**
Дай комплексный анализ и рекомендации, следуя этой структуре:
1.  **Общий вердикт:** Краткое резюме (2-3 предложения), соответствует ли текущий прогресс цели пользователя.
2.  **Анализ питания:** Оцени, соответствует ли КБЖУ цели и весу. Выяви возможные проблемы (например, недостаток белка).
3.  **Анализ тренировок:** Оцени прогресс в ключевых упражнениях. Есть ли стагнация (плато)?
4.  **Ключевые рекомендации:** 3-5 самых важных шагов для пользователя.
"""

        # 3. Отправка в LLM и получение ответа
        await process_llm(
            update, 
            context, 
            prompt, 
            mode="chat", 
            system_prompt_override=SYSTEM_PROMPT_FULL_ANALYSIS
        )

    except Exception as e:
        logging.error(f"Ошибка при создании полного анализа для user_id {user_id}: {e}", exc_info=True)
        await update.effective_message.reply_text(
            "Произошла ошибка при подготовке анализа. Попробуйте позже.",
            reply_markup=get_main_keyboard()
            )
