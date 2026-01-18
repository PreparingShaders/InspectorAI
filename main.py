import os, asyncio, re
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
1. Точная информация + фактчекинг.Укажи на сколько % это правда.
2. Если пользователь просит "напиши сочинение", "подробно", "статью" или указывает объем (например, 5к символов) — игнорируй ограничение краткости и пиши развернуто.
3. Если запрос требует краткости — отвечай кратко (до 300 зн).
4. Только русский язык. Январь 2026. Форматируй под Telegram.
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


def format_to_html(text: str) -> str:
    """Универсальная конвертация Markdown в HTML для всех провайдеров"""
    # 1. Сначала экранируем HTML-символы, которые могут быть в коде (чтобы не сломать парсинг)
    # Но если мы ожидаем, что модель УЖЕ может прислать HTML, этот шаг можно пропустить.
    # Для безопасности оставим только базовую замену:

    # 2. Конвертируем основные элементы Markdown в HTML
    # Жирный (Markdown: **текст** или __текст__)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'<b>\2</b>', text)
    # Курсив (Markdown: *текст* или _текст_)
    text = re.sub(r'(\*|_)(.*?)\1', r'<i>\2</i>', text)
    # Моноширинный код (Markdown: `текст`)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    # Блоки кода (Markdown: ```текст```)
    text = re.sub(r'```(?:.*?)\n?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)

    return text

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
    ADAPTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT + "\nВАЖНО: Если просят длинный текст или сочинение — пиши подробно, игнорируя лимит 300 зн.Январь 2026. Используй HTML-теги: <b>жирный</b>, <i>курсив</i>."

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

    # 1. Сохраняем в историю ОРИГИНАЛЬНЫЙ текст (без тегов)
    chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))

    # 2. Прогоняем текст через твою новую функцию форматирования
    formatted_text = format_to_html(reply_text)

    # 3. Формируем заголовок и итоговую строку
    model_short = last_used_model.split('/')[-1]
    full_reply = f"<b>{used_provider}: {model_short}</b>\n\n{formatted_text}"

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
        except Exception:
            # Если HTML всё же сломался, шлем без него (используем оригинальный reply_text)
            fallback_reply = f"{used_provider}: {model_short}\n\n{reply_text}"
            await context.bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=fallback_reply,
                                                parse_mode=None)
    else:
        # Длинный текст: удаляем статус и шлем частями
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_message_id)
        except:
            pass

        # Разбивка по абзацам, чтобы не разрывать теги
        paragraphs = full_reply.split('\n')
        current_chunk = ""

        for paragraph in paragraphs:
            # Если один абзац сам по себе длиннее лимита (редко, но бывает)
            if len(paragraph) > MAX_LEN:
                # Если в корзине что-то было, отправляем
                if current_chunk:
                    await context.bot.send_message(chat_id=chat_id, text=current_chunk, parse_mode='HTML')
                    current_chunk = ""

                # Режем гигантский абзац просто по символам (тут форматирование может слететь, но это крайний случай)
                for i in range(0, len(paragraph), MAX_LEN):
                    await context.bot.send_message(chat_id=chat_id, text=paragraph[i:i + MAX_LEN], parse_mode=None)
                continue

            # Если добавление абзаца превысит лимит — отправляем текущую корзину
            if len(current_chunk) + len(paragraph) + 1 > MAX_LEN:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=current_chunk, parse_mode='HTML')
                except:
                    await context.bot.send_message(chat_id=chat_id, text=current_chunk, parse_mode=None)
                current_chunk = paragraph + '\n'
                await asyncio.sleep(0.3)
            else:
                current_chunk += paragraph + '\n'

        # Отправляем остаток
        if current_chunk:
            try:
                await context.bot.send_message(chat_id=chat_id, text=current_chunk, parse_mode='HTML')
            except:
                await context.bot.send_message(chat_id=chat_id, text=current_chunk, parse_mode=None)

async def start(update: Update, context) -> None:
    user_id = update.effective_user.id
    if user_id in authorized_users:
        await update.message.reply_text("Ты уже авторизован!")
    else:
        await update.message.reply_text(AUTH_QUESTION)


async def handle_private(update: Update, context) -> None:
    user_id = update.effective_user.id
    message = update.message
    if not message: return

    # ИСПРАВЛЕНИЕ 1: Берем текст сообщения ИЛИ подпись к медиафайлу
    raw_text = message.text or message.caption or ""
    text = raw_text.strip()

    # Логика авторизации
    if user_id not in authorized_users:
        if text.lower() == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await message.reply_text("Авторизация пройдена! 🎉\nТеперь можешь задавать вопросы.")
        else:
            await message.reply_text("Неправильный пароль 😕\n\nНапиши /start и попробуй снова")
        return

    if not text:
        await message.reply_text("Напиши что-нибудь текстом или в подписи к фото 😏")
        return

    await process_llm(update, context, text)


async def handle_group(update: Update, context) -> None:
    message = update.message
    if not message: return

    # ИСПРАВЛЕНИЕ 2: Читаем текст из любого источника (сообщение/фото/видео)
    content_text = message.text or message.caption or ""
    if not content_text:
        return

    # Проверка упоминания (is_bot_mentioned уже есть в твоем коде)
    if not is_bot_mentioned(message, BOT_USERNAME):
        return

    clean_text = content_text

    # ИСПРАВЛЕНИЕ: приводим к list, чтобы сложение работало корректно
    all_entities = list(message.entities or []) + list(message.caption_entities or [])

    for entity in all_entities:
        if entity.type == "mention":
            mention = content_text[entity.offset: entity.offset + entity.length]
            if mention.lower() == f"@{BOT_USERNAME.lower()}":
                clean_text = clean_text.replace(mention, "", 1).strip()
                break

    prompt = ""
    # ИСПРАВЛЕНИЕ 4: Контекст ответа (тоже учитываем подписи)
    if message.reply_to_message:
        reply = message.reply_to_message
        reply_text = reply.text or reply.caption or ""
        if reply_text:
            prompt = f"Контекст (ответ на сообщение): {reply_text}\n\n"

    prompt += clean_text

    if not prompt.strip():
        await message.reply_text("Напиши вопрос после упоминания меня 😏")
        return

    await process_llm(update, context, prompt)


def main() -> None:
    if not InspectorGPT:
        print("Ошибка: Токен Telegram не найден!")
        return

    application = ApplicationBuilder().token(InspectorGPT).build()

    # Создаем универсальный фильтр: Текст ИЛИ Фото ИЛИ Видео ИЛИ Документы
    message_filter = filters.TEXT | filters.PHOTO | filters.VIDEO | filters.Document.ALL

    application.add_handler(CommandHandler("start", start))

    # Применяем фильтр для лички
    application.add_handler(MessageHandler(
        message_filter & filters.ChatType.PRIVATE,
        handle_private
    ))

    # Применяем фильтр для групп (исключая команды)
    application.add_handler(MessageHandler(
        message_filter & ~filters.COMMAND & ~filters.ChatType.PRIVATE,
        handle_group
    ))

    print("Бот запущен с поддержкой медиа-подписей...")
    application.run_polling()


if __name__ == "__main__":
    main()