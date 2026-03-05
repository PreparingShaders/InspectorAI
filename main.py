#main
import logging
import warnings
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters
)

# Удаляем UserWarning от библиотеки, если JobQueue еще не прогрузился
warnings.filterwarnings("ignore", category=UserWarning, module='telegram.ext')

from config import BOT_TOKEN
from handlers import (
    start, show_model_selection, handle_private,
    handle_group, callback_handler, handle_voice,
    profile_setup_handler, show_status # Импортируем новый хендлер
)
from llm_service import update_model_mappings
from utils import handle_voice_transcription, link_fixer_logic
from nutrition import init_db as init_nutrition_db # Импортируем функцию инициализации БД

# Выключаем логи от библиотек (httpx, apscheduler и т.д.)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

# Настраиваем основной лог только на важные ошибки
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.WARNING  # Было INFO, стало WARNING
)

async def voice_handler(update, context):
    """Прослойка для обработки голоса через utils"""
    if not update.message or not update.message.voice:
        return
    text = await handle_voice_transcription(update.message)
    if text:
        await update.message.reply_text(f"<b>Транскрипция:</b>\n\n{text}", parse_mode="HTML")


async def update_models_job(context):
    """Фоновая задача для динамического обновления списка моделей"""
    # Вызываем в отдельном потоке, так как там есть requests (синхронный)
    import asyncio
    await asyncio.to_thread(update_model_mappings)
    logging.info("🔄 Динамический список моделей обновлен.")


def main():
    if not BOT_TOKEN:
        logging.error("❌ BOT_TOKEN не найден!")
        return
    
    # Инициализируем базу данных для нутрициолога
    init_nutrition_db()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # 1. Задачи (JobQueue)
    jq = app.job_queue
    if jq:
        jq.run_once(update_models_job, when=0)
        jq.run_repeating(update_models_job, interval=1800)

    # 2. РЕГИСТРАЦИЯ ХЕНДЛЕРОВ (Порядок важен!)

    # Добавляем диалог для профиля
    app.add_handler(profile_setup_handler)

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", show_status))
    app.add_handler(CommandHandler("model", show_model_selection))
    app.add_handler(CallbackQueryHandler(callback_handler))

    # ГОЛОСОВЫЕ (Один хендлер для всего!)
    # Он должен стоять ВЫШЕ текстовых хендлеров
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Ссылки (Инспектор на автопилоте)
    app.add_handler(MessageHandler(
        (filters.Entity("url") | filters.Entity("text_link")) & ~filters.COMMAND,
        link_fixer_logic
    ), group=-1)

    # ЛИЧКА (Текст)
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & (filters.TEXT | filters.FORWARDED | filters.CAPTION) & ~filters.COMMAND,
        handle_private
    ))

    # ГРУППЫ (Текст)
    app.add_handler(MessageHandler(
        filters.ChatType.GROUPS & filters.TEXT & ~filters.COMMAND,
        handle_group
    ))

    logging.info("🚀 Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()