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
BOT_USERNAME = os.getenv('BOT_USERNAME', '').lstrip("@").lower()
CORRECT_PASSWORD = os.getenv('Password')
OPEN_ROUTER_API_KEY = os.getenv('OPEN_ROUTER_API_KEY')

# ─── Инициализация клиента Gemini ───
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        base_url="https://inspectorgpt.classname1984.workers.dev"
    )
)

SYSTEM_PROMPT = '''
Ты — ИИ помощник. 
1. Если запрос требует краткости — отвечай кратко (до 300 зн).
2. Если пользователь просит "напиши сочинение", "подробно", "статью" или указывает объем (например, 5к символов) — игнорируй ограничение краткости и пиши развернуто.
3. Точная информация + фактчекинг. Форматируй под Telegram.
4. Только русский язык. Январь 2026.
'''

chat_histories = defaultdict(list)
authorized_users = set()

AUTH_QUESTION = "Тут у нас пароль. Нужно отгадать загадку. Скажи, за какое время разгоняется нива до 100 км/ч"

MODELS_PRIORITY = [
    'models/gemini-3-flash-preview',
    'models/gemini-2.0-flash-lite',
    'models/gemini-2.0-flash-exp'
]

OPENROUTER_MODELS = [
    "xiaomi/mimo-v2-flash:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-4-maverick:free",
    "mistralai/devstral-2-2512:free",
    "microsoft/phi-4:free",
    "nousresearch/hermes-3-llama-3.1-405b:free"
]

def escape_md_v2_full(text: str) -> str:
    """Полное экранирование для MarkdownV2 — все специальные символы"""
    special = r'_*[]()~`>#+-=|{}.!'
    return ''.join('\\' + c if c in special else c for c in text)

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
    reply_to_message_id = update.effective_message.message_id

    # Обновляем историю
    history = chat_histories.get(chat_id, [])
    history.append(Content(role="user", parts=[types.Part(text=final_query)]))
    chat_histories[chat_id] = history[-6:]

    # Статус
    try:
        status_msg = await context.bot.send_message(
            chat_id=chat_id,
            text="⚡ Запускаю модели...",
            reply_to_message_id=reply_to_message_id
        )
        status_message_id = status_msg.message_id
    except:
        return

    reply_text = None
    used_provider = None
    last_used_model = ""

    # ВАЖНО: Обновленный промпт для баланса краткости и длины
    ADAPTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT + "\nВАЖНО: Если просят длинный текст или сочинение — пиши подробно, игнорируя лимит 300 зн. Используй HTML-теги: <b>жирный</b>, <i>курсив</i>."

    # ─── 1. Gemini ───
    for model_path in MODELS_PRIORITY:
        model_name = model_path.split('/')[-1]
        try:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=status_message_id,
                                                text=f"🔄 Gemini: {model_name}...")

            response = client.models.generate_content(
                model=model_path,
                contents=[Content(role="model", parts=[types.Part(text=ADAPTIVE_SYSTEM_PROMPT)])] + history,
                config=GenerateContentConfig(
                    temperature=0.75,
                    max_output_tokens=4000,  # Увеличено для длинных ответов
                    top_p=0.92
                )
            )
            if response and response.text:
                reply_text = response.text.strip()
                used_provider = "Gemini"
                last_used_model = model_path
                break
        except:
            continue

    # ─── 2. OpenRouter Fallback ───
    if not reply_text:
        or_client = OpenAI(api_key=OPEN_ROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        or_messages = [{"role": "system", "content": ADAPTIVE_SYSTEM_PROMPT}]
        for msg in history:
            role = "user" if msg.role == "user" else "assistant"
            text_part = msg.parts[0].text if hasattr(msg.parts[0], 'text') else str(msg.parts[0])
            or_messages.append({"role": role, "content": text_part})

        for model_path in OPENROUTER_MODELS:
            model_name = model_path.split('/')[-1]
            try:
                await context.bot.edit_message_text(chat_id=chat_id, message_id=status_message_id,
                                                    text=f"🔄 OR: {model_name}...")
                response = or_client.chat.completions.create(
                    model=model_path,
                    messages=or_messages,
                    temperature=0.75,
                    max_tokens=4000  # Увеличено
                )
                if response.choices and response.choices[0].message.content:
                    reply_text = response.choices[0].message.content.strip()
                    used_provider = "OR"
                    last_used_model = model_path
                    break
            except:
                continue

    # ─── 3. Финальная отправка (ВНЕ ЦИКЛОВ) ───
    if not reply_text:
        await context.bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="❌ Модели недоступны.")
        return

    # Сохраняем в историю
    chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))

    # Формируем заголовок с HTML
    model_short = last_used_model.split('/')[-1]
    full_reply = f"<b>{used_provider}: {model_short}</b>\n\n{reply_text}"

    # Telegram limit ~4096
    MAX_LEN = 4000

    if len(full_reply) <= MAX_LEN:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text=full_reply,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
        except Exception as e:
            # Если HTML сломался (например, ИИ забыл закрыть тег </b>), шлем чистым текстом
            await context.bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=full_reply,
                                                parse_mode=None)
    else:
        # Длинный текст: удаляем статус и шлем частями
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_message_id)
        except:
            pass

        # Разбивка по символам
        for i in range(0, len(full_reply), MAX_LEN):
            part = full_reply[i:i + MAX_LEN]
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=part,
                    parse_mode='HTML',
                    reply_to_message_id=reply_to_message_id if i == 0 else None
                )
            except:
                await context.bot.send_message(chat_id=chat_id, text=part, parse_mode=None)
            await asyncio.sleep(0.5)

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
            await update.message.reply_text("Авторизация пройдена! 🎉\nТеперь можешь задавать вопросы.")
        else:
            await update.message.reply_text("Неправильный пароль 😕\n\nНапиши /start и попробуй снова")
        return

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

    clean_text = message.text
    for entity in message.entities or []:
        if entity.type == "mention":
            mention = message.text[entity.offset: entity.offset + entity.length]
            if mention.lower() == f"@{BOT_USERNAME.lower()}":
                clean_text = clean_text.replace(mention, "", 1).strip()
                break

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