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


async def process_llm(update: Update, final_query: str):
    if not final_query or not final_query.strip():
        return

    chat_id = update.effective_chat.id
    history = chat_histories.get(chat_id, [])

    # Добавляем сообщение пользователя в историю (формат Google Content)
    history.append(Content(role="user", parts=[types.Part(text=final_query)]))
    chat_histories[chat_id] = history[-6:]  # Храним последние 6 реплик

    reply_text = None
    used_provider = None
    last_used_model = ""

    # --- ЭТАП 1: Прямое обращение к Gemini (Direct API) ---
    for current_model in MODELS_PRIORITY:
        try:
            print(f"🔄 Пробую Gemini Direct: {current_model}")
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
                break  # Успех, выходим из цикла Gemini

        except Exception as e:
            print(f"❌ Gemini {current_model} ошибка: {str(e)[:50]}")
            continue  # Пробуем следующую модель Gemini

    # --- ЭТАП 2: Fallback на OpenRouter (Если Gemini Direct не ответил) ---
    if not reply_text:
        print("⚠️ Все прямые Gemini недоступны. Перехожу к OpenRouter...")

        or_client = OpenAI(
            api_key=OPEN_ROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

        # Конвертируем историю из объектов Google в простые словари для OpenRouter
        or_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            role = "user" if msg.role == "user" else "assistant"
            # Извлекаем текст, даже если это объект Part
            raw_text = msg.parts[0].text if hasattr(msg.parts[0], 'text') else str(msg.parts[0])
            or_messages.append({"role": role, "content": raw_text})

        for or_model in OPENROUTER_MODELS:
            try:
                print(f"🔄 Пробую OpenRouter: {or_model}")
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
                    break  # Успех, выходим из цикла OpenRouter

            except Exception as e:
                print(f"❌ OR {or_model} ошибка: {str(e)[:100]}")
                continue  # Если эта модель на OpenRouter "лежит", пробуем следующую по списку

    # --- ФИНАЛ: Отправка ответа пользователю ---
    if reply_text:
        # Сохраняем ответ в историю
        chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))

        # Красивая пометка источника (берем только имя модели без пути)
        model_short_name = last_used_model.split('/')[-1]
        final_reply = f"({used_provider}: {model_short_name})\n {reply_text}"
    else:
        final_reply = "❌ К сожалению, все ИИ-модели сейчас заняты или недоступны. Попробуй через минуту."

    if update.message:
        try:
            await update.message.reply_text(final_reply[:4096])
        except Exception as e:
            print(f"Ошибка отправки сообщения: {e}")

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
            await update.message.reply_text("Авторизация пройдена!")
        else:
            await update.message.reply_text("Ты еще не авторизован, используй /start и введи пароль")
        return
    await process_llm(update, text)


async def handle_group(update: Update, context) -> None:
    message = update.message
    if not message or not message.text: return
    if not is_bot_mentioned(message, BOT_USERNAME): return

    text = message.text
    for entity in message.entities or []:
        if entity.type == "mention":
            mention = message.text[entity.offset: entity.offset + entity.length]
            if mention.lower() == f"@{BOT_USERNAME.lower()}":
                text = text.replace(mention, "", 1).strip()
                break

    context_text = ""
    if message.reply_to_message and message.reply_to_message.text:
        context_text = f"Контекст (ответ на сообщение): {message.reply_to_message.text}\n\n"

    await process_llm(update, context_text + text)


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