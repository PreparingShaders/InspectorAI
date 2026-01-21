import os
import asyncio
import re
import time

from collections import defaultdict
from datetime import datetime
from faster_whisper import WhisperModel

from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)

from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Content

load_dotenv()

# ─── Конфигурация ───────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("InspectorGPT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BOT_USERNAME = os.getenv("BOT_USERNAME", "").lstrip("@").lower()
CORRECT_PASSWORD = os.getenv("Password")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
TO_DAY = datetime.now().isoformat()

GEMINI_MODELS = [
    "models/gemini-2.5-flash",        # Стабильная, мощная, основной выбор
    "models/gemini-2.5-flash-lite",   # Быстрая, высокие лимиты, дешевле
    "models/gemini-3-flash-preview",  # Новинка, может иметь жесткие лимиты (20 зап/день)
    "models/gemini-2.0-flash",        # Предыдущее поколение (если еще доступно)
]

OPENROUTER_MODELS = [
    "xiaomi/mimo-v2-flash:free",
    "allenai/molmo-2-8b:free",
    "google/gemma-3-27b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "mistralai/devstral-2512:free",
    "tngtech/deepseek-r1t2-chimera:free",
    'liquid/lfm-2.5-1.2b-thinking:free',
    'qwen/qwen3-next-80b-a3b-instruct:free',
]

# Сопоставление коротких кодов → полные имена моделей
GEMINI_MODEL_BY_ID = {str(i): path for i, path in enumerate(GEMINI_MODELS)}
OPENROUTER_MODEL_BY_ID = {str(i + 100): path for i, path in enumerate(OPENROUTER_MODELS)}

# ─── Хранение состояний ─────────────────────────────────────────────────────
chat_histories = defaultdict(list)
authorized_users = set()
user_selected_model = defaultdict(lambda: None)          # полное имя модели или None
user_selected_provider = defaultdict(lambda: "gemini")   # "gemini" или "openrouter"

# ─── Клиенты ────────────────────────────────────────────────────────────────
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(base_url="https://inspectorgpt.classname1984.workers.dev"),
)

openrouter_client = OpenAI(
    api_key=OPEN_ROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)
model_whisper = WhisperModel("base", device="cpu", compute_type="int8")

SYSTEM_PROMPT = f'''
Ты — ИИ помощник. Текущая дата={TO_DAY} 
1. Точная информация + фактчекинг.Проверка новостей на текущую дату.Укажи на сколько % это правда.
2. Дефолтный ответ 250 зн, если просят развернутый или подробный игнорируй ограничение.
3. Если требуется, можешь просматривать статьи в интернете и искать факты.
4. Уместный юмор 7 из 10, подколы разрешены.
5. Только русский язык.Форматируй под Telegram.
'''

AUTH_QUESTION = "Тут у нас пароль. Нужно отгадать загадку. Скажи, за какое время разгоняется нива до 100 км/ч"


def escape_md_v2_full(text: str) -> str:
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
    text = re.sub(r'(\*\*|__)(.*?)\1', r'<b>\2</b>', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'<i>\2</i>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    text = re.sub(r'```(?:.*?)\n?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    return text


def get_model_short_name(model_path: str, provider: str) -> str:
    if provider == "gemini":
        return model_path.split("/")[-1].replace("models/", "")
    else:
        return model_path.split("/")[-1].split(":")[0]


async def show_model_selection(update: Update, context):
    """Показать меню выбора модели"""
    user_id = update.effective_user.id
    keyboard = []

    keyboard.append([InlineKeyboardButton("Gemini:", callback_data="dummy")])
    for i, model in enumerate(GEMINI_MODELS):
        name = get_model_short_name(model, "gemini")
        prefix = "✅ " if user_selected_model[user_id] == model else ""
        keyboard.append([
            InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:g:{i}")
        ])

    keyboard.append([InlineKeyboardButton("──────────────", callback_data="dummy")])

    keyboard.append([InlineKeyboardButton("OpenRouter:", callback_data="dummy")])
    for i, model in enumerate(OPENROUTER_MODELS):
        name = get_model_short_name(model, "openrouter")
        prefix = "✅ " if user_selected_model[user_id] == model else ""
        keyboard.append([
            InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:o:{i+100}")
        ])

    keyboard.append([
        InlineKeyboardButton("Автоматический выбор", callback_data="sel:auto")
    ])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выбери модель:", reply_markup=reply_markup)


async def callback_handler(update: Update, context):
    query = update.callback_query
    await query.answer()

    data = query.data
    user_id = query.from_user.id

    if data == "dummy":
        return

    if data == "sel:auto":
        user_selected_model[user_id] = None
        user_selected_provider[user_id] = "gemini"
        await query.edit_message_text("Вернулся автоматический выбор моделей")
        return

    if not data.startswith("sel:"):
        return

    try:
        _, prov_short, idx_str = data.split(":")
        idx = int(idx_str)
    except:
        await query.edit_message_text("Ошибка выбора модели")
        return

    model_path = None
    provider = None

    if prov_short == "g":
        model_path = GEMINI_MODEL_BY_ID.get(idx_str)
        provider = "gemini"
    elif prov_short == "o":
        model_path = OPENROUTER_MODEL_BY_ID.get(idx_str)
        provider = "openrouter"

    if model_path:
        user_selected_model[user_id] = model_path
        user_selected_provider[user_id] = provider
        name = get_model_short_name(model_path, provider)
        await query.edit_message_text(f"Выбрана модель:\n{provider.upper()} → {name}")
    else:
        await query.edit_message_text("Не  удалось выбрать модель")

async def process_llm(update: Update, context, final_query: str):
    if not final_query.strip():
        return

    chat_id = update.effective_chat.id
    reply_to_message_id = update.effective_message.message_id
    user_id = update.effective_user.id

    history = chat_histories[chat_id]
    history.append(Content(role="user", parts=[types.Part(text=final_query)]))
    chat_histories[chat_id] = history[-4:]

    if update.effective_chat.type in ("group", "supergroup"):
        await asyncio.sleep(1.2)   # 1.0–1.8 сек обычно хватает

    status_msg = await context.bot.send_message(
        chat_id=chat_id,
        text="⚡ Запускаю модели...",
        reply_to_message_id=reply_to_message_id
    )
    status_id = status_msg.message_id

    reply_text = None
    used_provider = None
    used_model_path = None

    ADAPTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT + "\nИспользуй HTML-теги: <b>жирный</b>, <i>курсив</i>."

    selected_model = user_selected_model[user_id]
    selected_provider = user_selected_provider[user_id]

    # 1. Пробуем выбранную пользователем модель (если выбрана)
    if selected_model:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=status_id,
                text=f"🔄 Пробую выбранную модель: {selected_model.split('/')[-1]}..."
            )

            if selected_provider == "gemini":
                response = gemini_client.models.generate_content(
                    model=selected_model,
                    contents=[Content(role="model", parts=[types.Part(text=ADAPTIVE_SYSTEM_PROMPT)])] + history,
                    config=GenerateContentConfig(temperature=0.75, max_output_tokens=4000, top_p=0.92)
                )
                if response and response.text:
                    reply_text = response.text.strip()
                    used_provider = "Gemini"
                    used_model_path = selected_model

            else:  # openrouter
                messages = [{"role": "system", "content": ADAPTIVE_SYSTEM_PROMPT}]
                for msg in history:
                    role = "user" if msg.role == "user" else "assistant"
                    content = msg.parts[0].text if msg.parts else ""
                    messages.append({"role": role, "content": content})

                response = openrouter_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=0.75,
                    max_tokens=4000
                )
                if response.choices and response.choices[0].message.content:
                    reply_text = response.choices[0].message.content.strip()
                    used_provider = "OpenRouter"
                    used_model_path = selected_model


        except Exception:
            model_name = get_model_short_name(selected_model, selected_provider)
            prov = selected_provider.upper()
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_id,
                text=(
                    f"❌ Модель недоступна: {prov} → {model_name}\n\n"
                    "Выбери другую модель:\n"
                    "→ команда /model"
                )
            )
            await asyncio.sleep(3)
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_id)
            except Exception:
                pass
            return  # ← очень важно! Прерываем функцию

    # 2. Обычный перебор, если ничего не получилось
    if reply_text is None:
        # Gemini
        for model_path in GEMINI_MODELS:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=status_id,
                    text=f"🔄 Gemini: {model_path.split('/')[-1]}..."
                )

                response = gemini_client.models.generate_content(
                    model=model_path,
                    contents=[Content(role="model", parts=[types.Part(text=ADAPTIVE_SYSTEM_PROMPT)])] + history,
                    config=GenerateContentConfig(temperature=0.75, max_output_tokens=4000, top_p=0.92)
                )
                if response and response.text:
                    reply_text = response.text.strip()
                    used_provider = "Gemini"
                    used_model_path = model_path
                    break
            except Exception:
                continue

        # OpenRouter fallback
        if reply_text is None:
            messages = [{"role": "system", "content": ADAPTIVE_SYSTEM_PROMPT}]
            for msg in history:
                role = "user" if msg.role == "user" else "assistant"
                content = msg.parts[0].text if msg.parts else ""
                messages.append({"role": role, "content": content})

            for model_path in OPENROUTER_MODELS:
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=status_id,
                        text=f"🔄 OR: {model_path.split('/')[-1].split(':')[0]}..."
                    )
                    response = openrouter_client.chat.completions.create(
                        model=model_path,
                        messages=messages,
                        temperature=0.75,
                        max_tokens=4000
                    )
                    if response.choices and response.choices[0].message.content:
                        reply_text = response.choices[0].message.content.strip()
                        used_provider = "OpenRouter"
                        used_model_path = model_path
                        break
                except Exception:
                    continue

    # 3. Финальный результат
    if reply_text is None:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=status_id,
            text="❌ Все модели сейчас недоступны 😔"
        )
        return

    # Сохраняем ответ в историю
    chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))

    model_short = used_model_path.split("/")[-1].split(":")[0]
    full_reply = f"<b>{used_provider}: {model_short}</b>\n\n{format_to_html(reply_text)}"

    # Отправка (твой оригинальный код обработки длинных сообщений можно вставить сюда)
    MAX_LEN = 4000
    if len(full_reply) <= MAX_LEN:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_id,
                text=full_reply,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        except:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_id,
                text=full_reply,
                parse_mode=None
            )
    else:
        # Разбивка на части — можно оставить как было в оригинале
        await context.bot.delete_message(chat_id=chat_id, message_id=status_id)
        # ... здесь можно вставить твою логику разбиения на части


# ─── Handlers ───────────────────────────────────────────────────────────────

async def start(update: Update, context):
    user_id = update.effective_user.id
    if user_id in authorized_users:
        model = user_selected_model[user_id]
        text = "Ты уже авторизован!\n\n"
        if model:
            prov = user_selected_provider[user_id].upper()
            name = model.split("/")[-1].split(":")[0]
            text += f"Текущая модель: {prov} → {name}\n\n"
        text += "Сменить модель → /model"
        await update.message.reply_text(text)
    else:
        await update.message.reply_text(AUTH_QUESTION)


async def handle_private(update: Update, context):
    user_id = update.effective_user.id
    message = update.message
    if not message:
        return

    text = (message.text or message.caption or "").strip()

    if user_id not in authorized_users:
        if text.lower() == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await message.reply_text("Авторизация пройдена! 🎉\nТеперь можешь задавать вопросы.\n\n/model — выбор модели")
        else:
            await message.reply_text("Неправильный пароль 😕\nИспользуй /start")
        return

    if not text:
        await message.reply_text("Напиши что-нибудь текстом или в подписи к фото 😏")
        return

    await process_llm(update, context, text)


async def handle_group(update: Update, context):
    message = update.message
    if not message:
        return

    content = message.text or message.caption or ""
    if not content:
        return

    # 1. Список триггеров
    TRIGGERS = ["инспектор",
                "шелупонь", "ботик",
                "бубен", 'андрюха',
                'андрей', 'малыш',
                'андрей генадьевич']
    content_lower = content.lower()

    # Проверяем, есть ли хоть одно слово из списка в сообщении
    has_trigger_word = any(word in content_lower for word in TRIGGERS)

    # 2. Проверяем @упоминание
    is_mentioned = is_bot_mentioned(message, BOT_USERNAME)

    # Если нет ни триггера, ни @упоминания — выходим
    if not (has_trigger_word or is_mentioned):
        return

    # --- ОЧИСТКА ТЕКСТА ---
    clean_text = content

    # Удаляем @mention (твой оригинальный код)
    entities = []
    if message.entities: entities.extend(message.entities)
    if message.caption_entities: entities.extend(message.caption_entities)

    for entity in entities:
        if entity.type == "mention":
            mention = content[entity.offset: entity.offset + entity.length]
            if mention.lower() == f"@{BOT_USERNAME.lower()}":
                clean_text = clean_text.replace(mention, "", 1)
                break

    # Удаляем все слова-триггеры из текста, чтобы они не уходили в нейронку
    for word in TRIGGERS:
        # Регулярка убирает слово целиком, игнорируя регистр
        clean_text = re.sub(rf'\b{word}\b', '', clean_text, flags=re.IGNORECASE)

    # Убираем лишние запятые и пробелы, которые часто остаются после обращения
    clean_text = re.sub(r'^[,\.\s?!\-]+', '', clean_text).strip()

    # --- КОНТЕКСТ РЕПЛАЯ ---
    prompt = ""
    if message.reply_to_message:
        reply = message.reply_to_message
        reply_text = reply.text or reply.caption or ""
        if reply_text:
            prompt = f"Контекст (ответ на сообщение): {reply_text}\n\n"

    prompt += clean_text

    if not prompt:
        await message.reply_text("Я тут! Задай свой вопрос после обращения 😏")
        return

    await process_llm(update, context, prompt)


async def link_fixer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    # Проверяем наличие сообщения и текста
    if not message or not message.text:
        return

    text = message.text

    # Твои паттерны замен
    replacements = {
        r"(https?://)(www\.)?instagram\.com/": r"\1kkinstagram.com/",
        # Используем tnktok.com — он сейчас стабильнее для vt. ссылок
        r"(https?://)((vm|vt|www)\.)?tiktok\.com/": r"\1tnktok.com/",
        r"(https?://)(www\.)?twitter\.com/": r"\1fxtwitter.com/",
        r"(https?://)(www\.)?x\.com/": r"\1fxtwitter.com/",
        r"(https?://)(www\.)?youtube\.com/shorts/": r"\1youtube.com/watch?v=",
    }

    new_text = text
    found = False

    for pattern, replacement in replacements.items():
        if re.search(pattern, new_text):
            new_text = re.sub(pattern, replacement, new_text)
            found = True

    if found:
        # 1. Пишем промежуточное сообщение
        status_msg = await message.reply_text("🔍 Вижу ссылку для редактирования...правим эту хохму 🛠")

        # 2. Ждем 1-3 секунды для "эффекта работы"
        await asyncio.sleep(2.5)

        # 3. Отправляем финальную ссылку
        user_name = message.from_user.first_name
        final_caption = f"✅ <b>Ссылка готова (от {user_name}):</b>\n\n{new_text}"

        await context.bot.send_message(
            chat_id=message.chat_id,
            text=final_caption,
            parse_mode="HTML",
            disable_web_page_preview=False  # Оставляем превью для видео
        )

        # 4. Удаляем сообщение отправителя и статусное сообщение
        try:
            await message.delete()  # Удаляем оригинал с кривой ссылкой
            await status_msg.delete()  # Удаляем текст "начинаю операцию"
        except Exception as e:
            print(f"Ошибка при удалении сообщений: {e}")
            # Если бот не админ, он не сможет удалить сообщение пользователя


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.voice:
        return

    # Генерируем путь для временного файла
    file_path = f"voice_{message.voice.file_unique_id}.ogg"

    try:
        # 1. Скачиваем файл
        voice_file = await context.bot.get_file(message.voice.file_id)
        await voice_file.download_to_drive(file_path)

        # 2. Расшифровываем (в отдельном потоке, чтобы не блокировать бота)
        segments, info = await asyncio.to_thread(model_whisper.transcribe, file_path, beam_size=5)

        # 3. Собираем текст
        transcribed_text = "".join([segment.text for segment in segments]).strip()

        # 4. Отправляем результат, если текст не пустой
        if transcribed_text:
            await message.reply_text(
                f"<b>Транскрипция:</b>\n\n{transcribed_text}",
                parse_mode="HTML"
            )
        # Если текст пустой, бот просто промолчит или можно добавить логирование в консоль

    except Exception as e:
        print(f"Ошибка STT: {e}")
        # В случае ошибки можно отправить скрытое уведомление или просто проигнорировать

    finally:
        # Чистим за собой файл
        if os.path.exists(file_path):
            os.remove(file_path)

def main():
    if not BOT_TOKEN:
        print("Ошибка: Токен Telegram не найден!")
        return

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(MessageHandler(filters.Entity("url") | filters.Entity("text_link"), link_fixer))

    # 2. Голосовые сообщения
    # Этот фильтр реагирует на любое голосовое
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("model", show_model_selection))
    app.add_handler(CallbackQueryHandler(callback_handler))

    message_filter = filters.TEXT | filters.PHOTO | filters.VIDEO | filters.Document.ALL

    app.add_handler(MessageHandler(message_filter & filters.ChatType.PRIVATE, handle_private))
    app.add_handler(MessageHandler(message_filter & ~filters.COMMAND & ~filters.ChatType.PRIVATE, handle_group))

    print("Бот запущен. Команда выбора модели: /model")
    app.run_polling()


if __name__ == "__main__":
    main()