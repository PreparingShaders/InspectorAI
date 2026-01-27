#main
import os
import asyncio
import re
import time
import requests

from collections import defaultdict
from datetime import datetime
from faster_whisper import WhisperModel
from web_utils import get_web_context

from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions
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
WORKER_URL = "https://inspectorgpt.classname1984.workers.dev"
BLACKLISTED_MODELS = set()      # Сюда будем временно вносить "упавшие" модели
current_free_or_models = []     # Тут будет лежать актуальный список ID

# 1. OpenRouter: От быстрых/легких к умным/тяжелым
OPENROUTER_MODELS = [
    # --- СКОРОСТЬ И ОТКЛИК (Fast & Lite) ---
    "nvidia/nemotron-3-nano-30b-a3b:free",   # Лидер по скорости, идеален для мелких просьб
    "arcee-ai/trinity-mini:free",            # Очень быстрая "малютка"
    "mistralai/devstral-2512:free",          # Оптимизированная скорость от Mistral
    "z-ai/glm-4.5-air:free",                 # Сбалансированная "воздушная" модель

    # --- СРЕДНИЙ КЛАСС (General Purpose) ---
    "google/gemma-3-27b-it:free",            # Хороший баланс интеллекта и скорости
    "meta-llama/llama-3.3-70b-instruct:free", # Золотой стандарт качества для общего чата
    'google/gemini-2.0-flash-exp:free',

    # --- УМНЫЕ И ТЯЖЕЛЫЕ (Reasoning / Heavy) ---
    "deepseek/deepseek-r1-0528:free",        # Чистый DeepSeek R1 (высокий интеллект)
    "tngtech/tng-r1t-chimera:free",          # "Химера" на базе R1, мощная логика
    "tngtech/deepseek-r1t-chimera:free",     # Вариант с упором на рассуждения
    "tngtech/deepseek-r1t2-chimera:free",    # Самая свежая и тяжелая итерация "Химеры"
]

# 2. Google Gemini: Ставим в самый конец, как ты и просил
GEMINI_MODELS = [
    "models/gemini-2.0-flash",               # Самая быстрая из гугловских
    "models/gemini-2.5-flash-lite",          # Облегченная версия
    "models/gemini-2.5-flash",               # Стабильный флагман
    "models/gemini-3-flash-preview",         # Новинка (может быть медленнее из-за лимитов)
]

# Копируем в актуальный список
current_free_or_models = OPENROUTER_MODELS.copy()


def update_model_mappings():
    global OPENROUTER_MODEL_BY_ID, GEMINI_MODEL_BY_ID

    # Маппинг для Gemini (ID от 0 до 99)
    GEMINI_MODEL_BY_ID.clear()
    for i, path in enumerate(GEMINI_MODELS):
        GEMINI_MODEL_BY_ID[str(i)] = path

    # Маппинг для OpenRouter (ID от 100 и далее)
    OPENROUTER_MODEL_BY_ID.clear()
    for i, path in enumerate(current_free_or_models):
        OPENROUTER_MODEL_BY_ID[str(i + 100)] = path

    print("🔄 Списки моделей зафиксированы согласно приоритетам скорости.")

# Вызываем один раз при старте
GEMINI_MODEL_BY_ID = {}
OPENROUTER_MODEL_BY_ID = {}
update_model_mappings()

# ─── Хранение состояний ─────────────────────────────────────────────────────
chat_histories = defaultdict(list)
authorized_users = set()
user_selected_model = defaultdict(lambda: None)          # полное имя модели или None
user_selected_provider = defaultdict(lambda: "gemini")   # "gemini" или "openrouter"

# 1. Клиент для OpenRouter
# Берем РЕАЛЬНЫЙ ключ из .env через os.getenv
or_client = OpenAI(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"), # Без кавычек!
    base_url=f"{WORKER_URL}/v1",
    timeout=45.0
)

# 2. Клиент для Gemini
# Берем РЕАЛЬНЫЙ ключ из .env через os.getenv
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"), # Без кавычек!
    http_options=types.HttpOptions(base_url=WORKER_URL,
                                   timeout=45000),
)

model_whisper = WhisperModel("base", device="cpu", compute_type="int8")

SYSTEM_PROMPT = f'''
Ты — InspectorGPT. Текущая дата: {TO_DAY}. 
Твой стиль: лаконичность, тонкий английский юмор и беспристрастность.

### ТВОИ РОЛИ:

1. ПОМОЩНИК (Обычный чат):
- Твоя цель — подробно и ясно помочь разобраться в вопросе.
- Не читай нотаций, отвечай по делу, предоставляя всю необходимую информацию.
- Используй иронию, если вопрос того требует.

2. ИНСПЕКТОР (Режим проверки фактов):
- Ты — аналитик-криминалист. Ты не веришь на слово.
- Твоя задача: противопоставить факты из интернета друг другу.
- Ищи логические дыры и манипуляции. Не принимай ничью сторону.
- Итог: подробный разбор + вердикт с точностью в % правды.
- Структурируй ответ списками или абзацами, без таблиц.

### ПРАВИЛА ОФОРМЛЕНИЯ:
- Только русский язык.
- Используй только эти HTML-теги: <b>жирный</b>, <i>курсив</i>, <code>код</code>, <pre>pre</pre>.
- Не используй <table>, <tr>, <td> или любые другие теги.
- ВАЖНО: Всегда закрывай теги!
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
    update_model_mappings()
    user_id = update.effective_user.id
    keyboard = []

    # --- Секция OpenRouter ---
    keyboard.append([InlineKeyboardButton("🎁 OpenRouter (Приоритет):", callback_data="dummy")])

    # Группируем модели OR по две в ряд
    or_buttons = []
    for i, model in enumerate(current_free_or_models):
        name = get_model_short_name(model, "openrouter")
        prefix = "✅ " if user_selected_model[user_id] == model else ""
        or_buttons.append(InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:o:{i + 100}"))
        if len(or_buttons) == 2:
            keyboard.append(or_buttons)
            or_buttons = []
    if or_buttons: keyboard.append(or_buttons)  # Добавляем остаток

    keyboard.append([InlineKeyboardButton("──────────────", callback_data="dummy")])

    # --- Секция Gemini ---
    keyboard.append([InlineKeyboardButton("✨ Gemini (Резерв):", callback_data="dummy")])

    gem_buttons = []
    for i, model in enumerate(GEMINI_MODELS):
        name = get_model_short_name(model, "gemini")
        prefix = "✅ " if user_selected_model[user_id] == model else ""
        gem_buttons.append(InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:g:{i}"))
        if len(gem_buttons) == 2:
            keyboard.append(gem_buttons)
            gem_buttons = []
    if gem_buttons: keyboard.append(gem_buttons)

    # --- Системные кнопки ---
    keyboard.append([InlineKeyboardButton("🤖 Автоматический выбор (OR -> Gem)", callback_data="sel:auto")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    text = "<b>Выбор модели ИИ</b>\nАвтовыбор сначала пробует OpenRouter, затем Gemini."

    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.effective_message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")

# --- Обновленный блок process_llm (логика перебора) ---

async def process_llm(update: Update, context, final_query: str, thread_id=None):
    if not final_query.strip():
        return

    chat_id = update.effective_chat.id
    reply_to_message_id = update.effective_message.message_id
    user_id = update.effective_user.id

    # Обновляем историю
    history = chat_histories[chat_id]
    history.append(Content(role="user", parts=[types.Part(text=final_query)]))
    chat_histories[chat_id] = history[-4:]

    if thread_id is None and update.effective_message:
        thread_id = update.effective_message.message_thread_id

    status_msg = await context.bot.send_message(
        chat_id=chat_id,
        text="⚡ Запускаю нейросети...",
        reply_to_message_id=reply_to_message_id,
        message_thread_id=thread_id
    )
    status_id = status_msg.message_id

    reply_text = None
    used_provider = None
    used_model_path = None

    ADAPTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT + "\nИспользуй HTML-теги: <b>жирный</b>, <i>курсив</i>."
    selected_model = user_selected_model[user_id]
    selected_provider = user_selected_provider[user_id]

    # --- БЛОК ФАКТЧЕКИНГА ---
    check_words = ["чекай", "проверь", 'факты', 'новости']
    is_factcheck = ("ОБЪЕКТ ПРОВЕРКИ:" in final_query) or any(word in final_query.lower() for word in check_words)

    if is_factcheck:
        try:
            if "ОБЪЕКТ ПРОВЕРКИ:" in final_query:
                search_query = final_query.split("ОБЪЕКТ ПРОВЕРКИ:")[1].split("\n\nВОПРОС:")[0].strip()
            else:
                search_query = final_query

            if search_query:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=status_id,
                    text=f"🔍 Режим детектива: проверяю «{search_query[:50]}...»"
                )
                web_data = await get_web_context(search_query)

                if web_data:
                    # Инъекция жесткой инструкции для режима Инспектора
                    final_query = (
                        f"⚠️ АКТИВИРОВАН РЕЖИМ ИНСПЕКТОРА ⚠️\n\n"
                        f"КОНТЕКСТ ИЗ СЕТИ:\n{web_data}\n\n"
                        f"ОБЪЕКТ АНАЛИЗА: {search_query}\n\n"
                        f"ИНСТРУКЦИЯ:\n"
                        f"1. Проведи перекрестный анализ данных выше.\n"
                        f"2. Выдели факты, которые подтверждаются, и те, что противоречат друг другу.\n"
                        f"3. Оцени надежность источников.\n"
                        f"4. Сформулируй беспристрастный вывод и укажи вероятность правды в %.\n"
                        f"5. Дай полный и детальный ответ, без сокращений.\n"
                        f"6. Структурируй ответ списками или абзацами. Не используй таблицы или тег <table>."
                    )
                    history[-1] = Content(role="user", parts=[types.Part(text=final_query)])

        except Exception as e:
            print(f"⚠️ Ошибка поиска: {e}")

    # --- 1. ЕСЛИ МОДЕЛЬ ВЫБРАНА ВРУЧНУЮ ---
    if selected_model:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=status_id,
                text=f"🔄 Пробую: {get_model_short_name(selected_model, selected_provider)}..."
            )
            if selected_provider == "gemini":
                response = gemini_client.models.generate_content(
                    model=selected_model,
                    contents=[Content(role="model", parts=[types.Part(text=ADAPTIVE_SYSTEM_PROMPT)])] + history,
                    config=GenerateContentConfig(temperature=0.75, max_output_tokens=4000)
                )
                if response and response.text:
                    reply_text = response.text.strip()
                    used_provider, used_model_path = "Gemini", selected_model
            else:
                messages = [{"role": "system", "content": ADAPTIVE_SYSTEM_PROMPT}]
                for msg in history:
                    messages.append(
                        {"role": "user" if msg.role == "user" else "assistant", "content": msg.parts[0].text})

                response = or_client.chat.completions.create(model=selected_model, messages=messages, temperature=0.75)
                if response.choices[0].message.content:
                    reply_text = response.choices[0].message.content.strip()
                    used_provider, used_model_path = "OpenRouter", selected_model
        except Exception as e:
            print(f"Ошибка выбранной модели: {e}")

    # --- 2. АВТОПЕРЕБОР (OpenRouter -> Gemini) ---
    if reply_text is None:
        # Сначала весь OpenRouter (т.к. он в приоритете)
        for model_path in current_free_or_models:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=status_id,
                    text=f"🔄 OR: {model_path.split('/')[-1].split(':')[0]}..."
                )
                messages = [{"role": "system", "content": ADAPTIVE_SYSTEM_PROMPT}]
                for msg in history:
                    messages.append(
                        {"role": "user" if msg.role == "user" else "assistant", "content": msg.parts[0].text})

                response = or_client.chat.completions.create(model=model_path, messages=messages, timeout=25)
                if response.choices and response.choices[0].message.content:
                    reply_text = response.choices[0].message.content.strip()
                    used_provider, used_model_path = "OpenRouter", model_path
                    break
            except:
                continue

        # Если OpenRouter не помог, идем в Gemini
        if reply_text is None:
            for model_path in GEMINI_MODELS:
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=status_id,
                        text=f"🔄 Gemini: {model_path.split('/')[-1]}..."
                    )
                    response = gemini_client.models.generate_content(
                        model=model_path,
                        contents=[Content(role="model", parts=[types.Part(text=ADAPTIVE_SYSTEM_PROMPT)])] + history
                    )
                    if response and response.text:
                        reply_text = response.text.strip()
                        used_provider, used_model_path = "Gemini", model_path
                        break
                except:
                    continue

    # --- 3. ФИНАЛЬНЫЙ ОТВЕТ ---
    if reply_text is None:
        await context.bot.edit_message_text(chat_id=chat_id, message_id=status_id,
                                            text="❌ Все модели сейчас недоступны 😔")
        return

    chat_histories[chat_id].append(Content(role="model", parts=[types.Part(text=reply_text)]))
    model_short = used_model_path.split("/")[-1].split(":")[0]
    full_reply = f"<b>{used_provider}: {model_short}</b>\n\n{format_to_html(reply_text)}"

    if len(full_reply) <= 4000:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=status_id,
                text=full_reply, parse_mode="HTML", disable_web_page_preview=True
            )
        except Exception:
            clean_reply = re.sub(r'<[^>]+>', '', full_reply)
            await context.bot.edit_message_text(chat_id=chat_id, message_id=status_id, text=clean_reply)
    else:
        # Для очень длинных текстов
        await context.bot.delete_message(chat_id=chat_id, message_id=status_id)
        for i in range(0, len(full_reply), 4000):
            chunk = full_reply[i:i + 4000]
            try:
                await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode="HTML")
            except Exception:
                clean_chunk = re.sub(r'<[^>]+>', '', chunk)
                await context.bot.send_message(chat_id=chat_id, text=clean_chunk)

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
    if not message: return

    # Проверка авторизации
    if user_id not in authorized_users:
        query_text = message.text.strip().lower()
        if query_text == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await message.reply_text("Авторизация успешна! Теперь можно пользоваться ботом.\nИспользуй /model")
        else:
            await message.reply_text("Неверный пароль. Попробуй еще раз.\nПопробуй /start")
        return

    query_text = message.text or message.caption or ""
    if not query_text.strip(): return

    # Условия
    is_forwarded = bool(message.forward_origin)
    check_words = ["чекай", "проверь", "факты", "новости"]
    has_check_word = any(word in query_text.lower() for word in check_words)
    is_reply = bool(message.reply_to_message)

    # ЛОГИКА:
    # 1. Форвард или обычное сообщение/реплай с чек-вордом -> ИНСПЕКТОР
    if is_forwarded or has_check_word:
        target_text = query_text
        # Если это реплай с чек-вордом, берем текст из того, кому отвечаем
        if is_reply and has_check_word:
            target_text = message.reply_to_message.text or message.reply_to_message.caption or query_text

        final_prompt = f"ОБЪЕКТ ПРОВЕРКИ: {target_text}\n\nВОПРОС: Проведи инспекцию фактов."

    # 2. Обычное сообщение или обычный реплай -> ПОМОЩНИК (API)
    else:
        final_prompt = query_text

    await process_llm(update, context, final_prompt)


async def handle_group(update: Update, context):
    message = update.message
    if not message: return

    content = message.text or message.caption or ""
    if not content: return

    content_lower = content.lower().strip()
    TRIGGERS = ["инспектор", "шелупонь", "ботик", "бубен", "андрюха", "андрей", "малыш", "андрей генадьевич"]
    CHECK_WORDS = ["чекай", "проверь", "факты", "новости"]

    # Проверки условий
    has_trigger = any(re.search(rf'\b{re.escape(word)}\b', content_lower) for word in TRIGGERS)
    has_check_word = any(word in content_lower for word in CHECK_WORDS)
    is_reply = bool(message.reply_to_message)

    # Если к боту не обратились по имени — игнорим
    if not has_trigger:
        return

    # Очистка текста от триггера для чистого запроса
    clean_text = content
    for word in TRIGGERS:
        clean_text = re.sub(rf'\b{re.escape(word)}\b', '', clean_text, flags=re.IGNORECASE).strip()

    # ЛОГИКА:
    # 1. Реплай + Триггер + Чек-ворд -> ИНСПЕКТОР (проверяем чужой реплай)
    if is_reply and has_check_word:
        target_text = message.reply_to_message.text or message.reply_to_message.caption or ""
        prompt = f"ОБЪЕКТ ПРОВЕРКИ: {target_text}\n\nВОПРОС: Инспектор, проверь это."

    # 2. Просто сообщение + Триггер + Чек-ворд -> ИНСПЕКТОР (проверяем само сообщение)
    elif has_check_word:
        prompt = f"ОБЪЕКТ ПРОВЕРКИ: {clean_text}\n\nВОПРОС: Проверь факты."

    # 3. Реплай + Триггер (без чек-ворда) -> ПОМОЩНИК (API с контекстом)
    elif is_reply:
        target_text = message.reply_to_message.text or message.reply_to_message.caption or ""
        prompt = f"Контекст: {target_text}\n\nВопрос: {clean_text}"

    # 4. Просто Триггер -> ПОМОЩНИК (API)
    else:
        prompt = clean_text

    await process_llm(update, context, prompt, thread_id=message.message_thread_id)



async def link_fixer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.text:
        return

    text = message.text
    thread_id = message.message_thread_id

    # Список только тех сервисов, которые мы ПРАВИМ
    replacements = {
        r"(https?://)(www\.)?instagram\.com/": r"\1kkinstagram.com/",
        r"(https?://)((vm|vt|www)\.)?tiktok\.com/": r"\1vxtiktok.com/",
        r"(https?://)(www\.)?twitter\.com/": r"\1fxtwitter.com/",
        r"(https?://)(www\.)?x\.com/": r"\1fxtwitter.com/",
    }

    new_text = text
    found = False
    target_url = None

    # Проверяем, есть ли в сообщении хотя бы одна ссылка из нашего списка
    for pattern, replacement in replacements.items():
        match = re.search(pattern, text)
        if match:
            # Если нашли — заменяем и помечаем, что сообщение нужно переотправить
            new_text = re.sub(pattern, replacement, new_text)
            found = True
            # Запоминаем исправленную ссылку для "хитрости" с превью
            if not target_url:
                url_match = re.search(r"https?://\S+", new_text)
                if url_match:
                    target_url = url_match.group(0)

    # Если это была обычная ссылка (не из списка), функция просто завершится здесь
    if not found:
        return

    user_name = message.from_user.first_name

    # Формируем скрытую ссылку для форсирования превью
    hidden_link = f'<a href="{target_url}">\u200b</a>' if target_url else ""
    final_caption = f"{hidden_link}✅ <b>От {user_name}:</b>\n{new_text}"

    # Удаляем старое
    try:
        await message.delete()
    except:
        pass

    # Ждем, чтобы Telegram "протрезвел" и был готов загрузить новое видео
    await asyncio.sleep(1.2)

    await context.bot.send_message(
        chat_id=message.chat_id,
        text=final_caption,
        parse_mode="HTML",
        message_thread_id=thread_id,
        link_preview_options=LinkPreviewOptions(
            is_disabled=False,
            prefer_large_media=True,
            show_above_text=False
        )
    )


async def callback_handler(update: Update, context):
    query = update.callback_query
    await query.answer()  # Убирает "часики" на кнопке

    data = query.data
    user_id = query.from_user.id

    if data == "open_menu":
        await show_model_selection(update, context)
        return

    if data == "dummy":
        return

    if data == "sel:auto":
        user_selected_model[user_id] = None
        user_selected_provider[user_id] = "openrouter"  # Начинаем перебор с OR
        await query.edit_message_text("✅ Включен автоматический выбор (OR → Gemini)")
        return

    if not data.startswith("sel:"):
        return

    try:
        # Парсим данные вида "sel:o:105" (provider_short : index)
        _, prov_short, idx_str = data.split(":")
    except ValueError:
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

        # Сразу предлагаем обновить меню, чтобы увидеть "галочку"
        keyboard = [[InlineKeyboardButton("🔙 Назад к списку", callback_data="open_menu")]]
        await query.edit_message_text(
            f"🎯 Выбрана модель:\n<b>{provider.upper()}</b> → <code>{name}</code>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await query.edit_message_text("❌ Ошибка: Модель не найдена в текущем списке.")

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

    # --- ГРУППА -1 (Обработка ссылок) ---
    # Мы добавляем параметр group=-1. Бот сначала зайдет сюда.
    app.add_handler(MessageHandler(
        (filters.Entity("url") | filters.Entity("text_link")) & ~filters.COMMAND,
        link_fixer
    ), group=-1)

    # --- ГРУППА 0 (Основная логика) ---
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("model", show_model_selection))
    app.add_handler(CallbackQueryHandler(callback_handler))

    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    message_filter = filters.TEXT | filters.PHOTO | filters.VIDEO | filters.Document.ALL

    app.add_handler(MessageHandler(message_filter & filters.ChatType.PRIVATE, handle_private))
    app.add_handler(MessageHandler(message_filter & ~filters.COMMAND & ~filters.ChatType.PRIVATE, handle_group))

    print("Бот запущен. Команда выбора модели: /model")
    app.run_polling()


if __name__ == "__main__":
    main()