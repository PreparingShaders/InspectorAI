import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext

from google import genai
from google.genai.types import GenerateContentConfig, Content
from collections import defaultdict

load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
BOT_USERNAME = os.getenv('BOT_USERNAME')  # Должно быть "@inspector" (с @)

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"  # ← переключаю на стабильную, preview иногда капризничает

SYSTEM_PROMPT = """
Ты — InspectorAI, остроумный ИИ-помощник для анализа и комментариев в чате.
Текущая дата — январь 2026 года. Всегда отвечай с учётом 2026 года.
Отвечай кратко, информативно, с юмором где уместно.
Отвечай только на русском языке.
"""

chat_histories = defaultdict(list)
authorized_users = set()

AUTH_QUESTION = "Укажите разгон нивы до 100 км/ч"
CORRECT_PASSWORD = "4 секунды"

async def start(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id
    if user_id in authorized_users:
        await update.message.reply_text('Ты уже авторизован! Пиши в группе с упоминанием меня.')
    else:
        await update.message.reply_text(AUTH_QUESTION)

async def handle_auth(update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id
    if user_id not in authorized_users and update.effective_chat.type == 'private':
        user_answer = update.message.text.strip().lower()
        if user_answer == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await update.message.reply_text('Авторизация пройдена! Теперь используй меня в группах.')
        else:
            await update.message.reply_text('Неверно. Попробуй снова после /start.')

async def handle_possible_mention(update: Update, context: CallbackContext) -> None:
    if not update.message or not update.message.text:
        return

    message_text = update.message.text
    user_id = update.effective_user.id
    chat_type = update.effective_chat.type

    # Проверяем наличие упоминания (с учётом возможных пробелов/регистра)
    if BOT_USERNAME.lower() not in message_text.lower():
        return

    # Авторизация только для групп (в привате — всегда отвечаем после /start)
    if chat_type != 'private' and user_id not in authorized_users:
        await update.message.reply_text('Сначала авторизуйся в личке со мной через /start.')
        return

    # Убираем упоминание
    query = message_text.replace(BOT_USERNAME, '').strip()

    if not query:  # просто @inspector без текста — игнорируем
        return

    chat_id = update.effective_chat.id

    # Обработка реплая
    if update.message.reply_to_message:
        replied_text = update.message.reply_to_message.text or ""
        if "прокомментируй" in query.lower():
            query = f"Прокомментируй это сообщение: '{replied_text}'. {query}"
        else:
            query = f"Контекст: '{replied_text}'. {query}"

    try:
        history = chat_histories[chat_id]
        history.append(Content(role="user", parts=[{"text": query}]))
        if len(history) > 12:  # чуть увеличил историю
            history = history[-12:]

        contents = [Content(role="model", parts=[{"text": SYSTEM_PROMPT}])] + history

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=GenerateContentConfig(temperature=0.8, max_output_tokens=1024)
        )

        gemini_reply = response.text or "Gemini задумался..."
        history.append(Content(role="model", parts=[{"text": gemini_reply}]))

    except Exception as e:
        gemini_reply = f"Ошибка Gemini: {str(e)[:200]}"  # обрезаем длинные ошибки

    if len(gemini_reply) > 4096:
        gemini_reply = gemini_reply[:4093] + "..."

    await update.message.reply_text(gemini_reply)


def main() -> None:
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.PRIVATE, handle_auth))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_possible_mention))

    print("Бот запущен. Privacy mode должен быть выключен для групп!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)  # ← добавили ALL_TYPES на всякий


if __name__ == '__main__':
    main()