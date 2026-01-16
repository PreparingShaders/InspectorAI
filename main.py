import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Content

from collections import defaultdict

load_dotenv()

InspectorGPT = os.getenv('InspectorGPT')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
BOT_USERNAME = os.getenv('BOT_USERNAME').lstrip("@").lower()  # без @
CORRECT_PASSWORD = os.getenv('Password')
OPEN_ROUTER_API_KEY = os.getenv('OPEN_ROUTER_API_KEY')

# ─── Инициализация клиента с прокси через Cloudflare Worker ───
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        base_url="https://inspectorgpt.classname1984.workers.dev"
    )
)

SYSTEM_PROMPT = '''
Ты — ИИ помощник.  
Точная, понятная информация + фактчекинг.  
Простой язык. Кратко (≤300 зн).  
Редкий тонкий юмор ок.  
Только русский. Январь 2026.
'''

chat_histories = defaultdict(list)
authorized_users = set()

AUTH_QUESTION = "Тут у нас пароль. Нужно отгадать загадку. Скажи, за какое время разгоняется нива до 100 км/ч"

# ─── Вспомогательная функция для упоминания бота ───
def is_bot_mentioned(message, bot_username: str) -> bool:
    if not message.entities:
        return False

    for entity in message.entities:
        if entity.type == "mention":
            mention_text = message.text[entity.offset: entity.offset + entity.length]
            if mention_text.lower() == f"@{bot_username.lower()}":
                return True
    return False


# Используем полные имена из твоего списка
MODELS_PRIORITY = [
    'models/gemini-2.5-flash-lite',  # Приоритет №1 (Лимит 1000)
    'models/gemini-2.0-flash-lite',
    'models/gemini-2.5-flash',
    'models/gemini-3-flash-preview',
    'models/gemini-1.5-flash-lite-latest'
]


async def process_llm(update: Update, final_query: str):
    if not final_query or not final_query.strip():
        return

    chat_id = update.effective_chat.id
    history = chat_histories.get(chat_id, [])

    # Добавляем сообщение пользователя
    history.append(Content(role="user", parts=[types.Part(text=final_query)]))
    chat_histories[chat_id] = history[-6:]  # оставляем последние 14

    reply_text = "…я задумался, попробуй иначе"
    used_provider = None  # Будем хранить, кто ответил: "gemini" или "openrouter"

    # ─── Попытка через Gemini ───
    for current_model in MODELS_PRIORITY:
        try:
            print(f"Запрос к Gemini: {current_model}")

            response = client.models.generate_content(
                model=current_model,
                contents=[Content(role="model", parts=[types.Part(text=SYSTEM_PROMPT)])] + history,
                config=GenerateContentConfig(
                    temperature=0.75,
                    max_output_tokens=512,
                    top_p=0.92
                )
            )

            if response and response.text:
                reply_text = response.text.strip()
                chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))
                used_provider = "Gemini"
                break

        except Exception as e:
            err_str = str(e).lower()
            print(f"Gemini {current_model} → ошибка: {err_str[:100]}...")
            if any(x in err_str for x in ["429", "quota", "rate limit", "503", "unavailable", "404"]):
                continue
            else:
                break

    # ─── Fallback на OpenRouter ───
    if used_provider is None:
        print("Gemini недоступен → пробуем OpenRouter")
        try:
            or_client = OpenAI(
                api_key=OPEN_ROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in history:
                role = "user" if msg.role == "user" else "assistant"
                messages.append({"role": role, "content": msg.parts[0].text})

            response = or_client.chat.completions.create(
                model="nousresearch/hermes-3-llama-3.1-405b:free",
                messages=messages,
                temperature=0.75,
                max_tokens=512,
                top_p=0.92,
            )

            if response.choices and response.choices[0].message.content:
                reply_text = response.choices[0].message.content.strip()
                chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))
                used_provider = "OR"

        except Exception as or_error:
            print(f"OpenRouter ошибка: {str(or_error)[:150]}")
            reply_text = "Сейчас оба ИИ недоступны. Попробуй через 5–10 минут."

    # ─── Добавляем маркер источника ───
    if used_provider == "Gemini":
        final_reply = f"(Gemini) {reply_text}"
    elif used_provider == "OR":
        final_reply = f"(OR) {reply_text}"
    else:
        final_reply = reply_text  # сообщение об ошибке без префикса

    # ─── Отправка ───
    if update.message:
        try:
            await update.message.reply_text(final_reply[:4096])
        except Exception as e:
            print(f"Ошибка отправки: {e}")

# ─── Обработка /start в личке ───
async def start(update: Update, context) -> None:
    user_id = update.effective_user.id
    if user_id in authorized_users:
        await update.message.reply_text("Ты уже авторизован! Пиши в группе с упоминанием меня.")
    else:
        await update.message.reply_text(AUTH_QUESTION)


# ─── Обработка личного чата: авторизация + диалог ───
async def handle_private(update: Update, context) -> None:
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if user_id not in authorized_users:
        # проверка пароля
        if text.lower() == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await update.message.reply_text("Авторизация пройдена! Андрей Генадьевич готов общаться.")
        else:
            await update.message.reply_text("Неверно. Попробуй снова после /start.")
        return

    # После авторизации → обычный диалог
    await process_llm(update, text)


# ─── Обработка сообщений в группе ───
async def handle_group(update: Update, context) -> None:
    message = update.message
    if not message or not message.text:
        return

    text = message.text.strip()

    if not is_bot_mentioned(message, BOT_USERNAME):
        return  # реагируем только на упоминание

    # Удаляем упоминание
    for entity in message.entities or []:
        if entity.type == "mention":
            mention = message.text[entity.offset: entity.offset + entity.length]
            if mention.lower() == f"@{BOT_USERNAME.lower()}":
                text = text.replace(mention, "", 1).strip()
                break

    if not text and not message.reply_to_message:
        return

    context_text = ""
    if message.reply_to_message:
        replied_text = (message.reply_to_message.text or "[Non-text сообщение, опиши по контексту]").strip()
        if replied_text:
            # Всегда комментируем reply, но добавляем текст пользователя как уточнение
            context_text = (
                "Прокомментируй сообщение ниже кратко, по делу и с юмором. "
                "Если есть дополнительный запрос — учти его:\n\n"
                f"{replied_text}\n\n"
            )

    final_query = context_text + text
    await process_llm(update, final_query)

# ─── Запуск бота ───
def main() -> None:
    application = ApplicationBuilder().token(InspectorGPT).build()

    # /start
    application.add_handler(CommandHandler("start", start))

    # Личка
    application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.PRIVATE, handle_private))

    # Группа
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.ChatType.PRIVATE, handle_group))

    print("Бот запущен. Privacy mode в @BotFather должен быть выключен для работы в группах!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
