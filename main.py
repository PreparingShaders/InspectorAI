#main
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='telegram.ext')

import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)

from config import BOT_TOKEN
from handlers.base import start # Import start from base.py
from handlers.common_handlers import (
    show_model_selection, handle_private,
    handle_group, callback_handler, handle_voice,
)
from handlers.nutrition_handlers import (
    profile_setup_handler,
    show_nutrition_menu, # Импортируем новую функцию
)
from handlers.workouts_handlers import (
    show_workouts_menu,
    add_workout_conversation_handler,
    edit_workout_conversation_handler, # Новый импорт
    run_workout_conversation_handler # Новый импорт
)
from llm_service import update_model_mappings
from utils import handle_voice_transcription, link_fixer_logic
from nutrition import init_db as init_nutrition_db
from workouts import init_db as init_workouts_db # Новый импорт

# Настройка логгирования
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.WARNING
)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)

async def update_models_job(context: ContextTypes.DEFAULT_TYPE):
    import asyncio
    await asyncio.to_thread(update_model_mappings)
    logging.info("🔄 Динамический список моделей обновлен.")

def main():
    if not BOT_TOKEN:
        logging.error("❌ BOT_TOKEN не найден!")
        return
    
    init_nutrition_db()
    init_workouts_db() # Инициализация базы данных тренировок

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    jq = app.job_queue
    if jq:
        jq.run_once(update_models_job, when=0)
        jq.run_repeating(update_models_job, interval=1800)

    # --- РЕГИСТРАЦИЯ ХЕНДЛЕРОВ ---

    # 1. Диалоги
    app.add_handler(profile_setup_handler)
    app.add_handler(add_workout_conversation_handler)
    app.add_handler(edit_workout_conversation_handler) # Регистрация ConversationHandler для редактирования тренировок
    app.add_handler(run_workout_conversation_handler) # Регистрация ConversationHandler для запуска тренировок

    # 2. Основные команды и кнопки
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.Regex('^🥗 Нутрициолог$'), show_nutrition_menu)) # Новый обработчик
    app.add_handler(MessageHandler(filters.Regex('^🏋️ Тренировки$'), show_workouts_menu)) # Новый обработчик
    app.add_handler(MessageHandler(filters.Regex('^🤖 Сменить модель$'), show_model_selection))
    
    # Старые команды для обратной совместимости (удаляем нутрициологические)
    app.add_handler(CommandHandler("model", show_model_selection))

    # 3. Обработчики сообщений
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(
        (filters.Entity("url") | filters.Entity("text_link")) & ~filters.COMMAND,
        link_fixer_logic
    ), group=-1)
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & (filters.TEXT | filters.FORWARDED | filters.CAPTION) & ~filters.COMMAND,
        handle_private
    ))
    app.add_handler(MessageHandler(
        filters.ChatType.GROUPS & filters.TEXT & ~filters.COMMAND,
        handle_group
    ))

    logging.info("🚀 Бот запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()