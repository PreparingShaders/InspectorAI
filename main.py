import os
import asyncio
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

# ─── Инициализация Gemini клиента ───
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
Редкий тонкий юмор ок. Форматируй текст ответа под Telegram  
Только русский. Январь 2026.
'''

chat_histories = defaultdict(list)
authorized_users = set()

AUTH_QUESTION = "Тут у нас пароль. Нужно отгадать загадку. Скажи, за какое время разгоняется нива до 100 км/ч"

# --- ЭТАП 1: Прямое обращение к Google (Самый высокий приоритет) ---
# Эти модели работают через твой прокси/Direct API.
MODELS_PRIORITY = [
    'models/gemini-3-flash-preview',      # Твой текущий лидер (уже работает!)
    'models/gemini-2.0-flash-lite',       # Самая быстрая для простых команд
    'models/gemini-2.0-flash-exp'         # Хорошая альтернатива
]

# --- ЭТАП 2: OpenRouter (Только уникальные бесплатные модели) ---
OPENROUTER_MODELS = [
    "xiaomi/mimo-v2-flash:free",          # ХИТ 2026: 309B параметров, очень умная
    "deepseek/deepseek-r1:free",          # Новая логика (замена старому chat)
    "qwen/qwen3-235b-a22b:free",          # Новейший Qwen 3 (лучший для русского)
    "meta-llama/llama-4-maverick:free",    # Четвертое поколение Llama (Scout/Maverick)
    "mistralai/devstral-2-2512:free",     # Специальная модель для кодинга и логики
    "microsoft/phi-4:free",               # Маленькая, но очень качественная
    "nousresearch/hermes-3-llama-3.1-405b:free" # Запасной гигант
]
def is_bot_mentioned(message, bot_username: str) -> bool:
    if not message.entities:
        return False
    for entity in message.entities:
        if entity.type == "mention":
            mention_text = message.text[entity.offset: entity.offset + entity.length]
            if mention_text.lower() == f"@{bot_username.lower()}":
                return True
    return False


async def process_llm(update: Update, context, final_query: str):
    if not final_query or not final_query.strip():
        return

    chat_id = update.effective_chat.id
    # Для ответа в reply на сообщение пользователя
    reply_to_message_id = update.effective_message.message_id

    history = chat_histories.get(chat_id, [])
    history.append(Content(role="user", parts=[types.Part(text=final_query)]))
    chat_histories[chat_id] = history[-6:]

    # Отправляем начальное сообщение-статус
    try:
        status_msg = await context.bot.send_message(
            chat_id=chat_id,
            text="⚡ Запускаю модели...\nПробую Gemini...",
            reply_to_message_id=reply_to_message_id,
            disable_notification=True
        )
        status_message_id = status_msg.message_id
    except Exception as e:
        print(f"Не удалось отправить статус: {e}")
        return

    reply_text = None
    used_provider = None
    last_used_model = ""

    # Пробуем модели Gemini по очереди
    for current_model in MODELS_PRIORITY:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text=f"🔄 Пробую Gemini: {current_model.split('/')[-1]}..."
            )

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
                used_provider = "Gemini"
                last_used_model = current_model
                break

        except Exception as e:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text=f"❌ {current_model.split('/')[-1]} ошибка\nПробую следующую..."
            )
            await asyncio.sleep(0.5)
            continue

    # Если ничего не получилось — OpenRouter
    if not reply_text:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message_id,
            text="⚠️ Gemini недоступны\n→ Перехожу на OpenRouter..."
        )
        await asyncio.sleep(0.7)

        or_client = OpenAI(
            api_key=OPEN_ROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

        or_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            role = "user" if msg.role == "user" else "assistant"
            raw_text = msg.parts[0].text if hasattr(msg.parts[0], 'text') else str(msg.parts[0])
            or_messages.append({"role": role, "content": raw_text})

        for or_model in OPENROUTER_MODELS:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text=f"🔄 Пробую {or_model.split('/')[-1]} (OpenRouter)..."
                )

                response = or_client.chat.completions.create(
                    model=or_model,
                    messages=or_messages,
                    temperature=0.75,
                    max_tokens=512,
                    extra_headers={
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "InspectorGPT",
                    }
                )

                if response.choices and response.choices[0].message.content:
                    reply_text = response.choices[0].message.content.strip()
                    used_provider = "OR"
                    last_used_model = or_model
                    break

            except Exception as e:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text=f"❌ {or_model.split('/')[-1]} ошибка\nСледующая..."
                )
                await asyncio.sleep(0.5)
                continue

    # Финальная обработка ответа
    model_short_name = last_used_model.split('/')[-1] if last_used_model else "неизвестно"
    source_line = f"({used_provider}: {model_short_name})"

    if reply_text:
        chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))

        full_text = f"{source_line}\n\n{reply_text}"

        # Короткий ответ — редактируем статус
        if len(full_text) <= 4000:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text=full_text,
                    parse_mode="Markdown",
                    disable_web_page_preview=True
                )
                return
            except Exception:
                # Если не получилось отредактировать — отправим новое
                pass

        # Длинный ответ — разбиваем
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message_id,
            text=f"{source_line}\n\nОтвет длинный → отправляю частями..."
        )

        # Разбиение на части
        chunks = []
        current_chunk = ""
        for line in reply_text.splitlines(keepends=True):
            if len(current_chunk) + len(line) > 3900:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += line
        if current_chunk:
            chunks.append(current_chunk)

        for i, chunk in enumerate(chunks, 1):
            part_text = f"Часть {i}/{len(chunks)}\n\n{chunk.strip()}"
            if i == 1:
                part_text = f"{source_line}\n\n{part_text}"

            await context.bot.send_message(
                chat_id=chat_id,
                text=part_text,
                reply_to_message_id=reply_to_message_id,
                parse_mode="Markdown",
                disable_notification=True,
                disable_web_page_preview=True
            )
            await asyncio.sleep(0.4)  # небольшая пауза между частями

    else:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message_id,
            text="❌ Все модели сейчас недоступны.\nПопробуйте через минуту-две."
        )
async def start(update: Update, context) -> None:
    user_id = update.effective_user.id
    if user_id in authorized_users:
        await update.message.reply_text("Ты уже авторизован!")
    else:
        await update.message.reply_text(AUTH_QUESTION)


async def handle_private(update: Update, context) -> None:
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if user_id not in authorized_users:
        if text.lower() == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await update.message.reply_text(
                "Авторизация пройдена! 🎉\nТеперь можешь задавать вопросы."
            )
        else:
            await update.message.reply_text(
                "Неправильный пароль 😕\n\n"
                "Напиши /start и попробуй снова"
            )
        return

    # Если сообщение пустое после strip — не обрабатываем
    if not text:
        await update.message.reply_text("Напиши что-нибудь, я готов отвечать 😏")
        return

    await process_llm(update, context, text)


async def handle_group(update: Update, context) -> None:
    message = update.message
    if not message or not message.text:
        return

    if not is_bot_mentioned(message, BOT_USERNAME):
        return

    # Убираем @botname из текста
    clean_text = message.text
    for entity in message.entities or []:
        if entity.type == "mention":
            mention = message.text[entity.offset: entity.offset + entity.length]
            if mention.lower() == f"@{BOT_USERNAME.lower()}":
                clean_text = clean_text.replace(mention, "", 1).strip()
                break

    # Добавляем контекст из ответа, если сообщение — reply
    prompt = ""
    if message.reply_to_message and message.reply_to_message.text:
        prompt = f"Контекст (ответ на сообщение): {message.reply_to_message.text}\n\n"

    prompt += clean_text

    if not prompt.strip():
        await message.reply_text("Напиши что-нибудь после упоминания меня 😏")
        return

    await process_llm(update, context, prompt)


def main() -> None:
    if not InspectorGPT:
        print("Ошибка: Токен Telegram (InspectorGPT) не найден!")
        return
    application = ApplicationBuilder().token(InspectorGPT).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.PRIVATE, handle_private))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.ChatType.PRIVATE, handle_group))
    print("Бот запущен...")
    application.run_polling()


if __name__ == "__main__":
    main()