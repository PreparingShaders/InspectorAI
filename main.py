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

# 2. Теперь сама функция (она теперь видит BLACKLISTED_MODELS)
def fetch_free_openrouter_models():
    """Запрашивает список, фильтрует бесплатные и сортирует по контексту"""
    url = f"{WORKER_URL}/v1/models"
    headers = {"Authorization": f"Bearer {OPEN_ROUTER_API_KEY}"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            try:
                data = response.json().get('data', [])
            except ValueError:
                print("⚠️ Ошибка: API вернул не JSON")
                return None

            free_models_data = []
            for m in data:
                m_id = m['id']
                pricing = m.get('pricing', {})
                # Берем контекст, если его нет — ставим 0
                context_length = int(m.get('context_length', 0))

                # Проверка на бесплатность (цена 0 или тег :free)
                is_free = (":free" in m_id) or (
                        float(pricing.get('prompt', 1)) == 0 and
                        float(pricing.get('completion', 1)) == 0
                )

                # ВОТ ТУТ ОНА ИСПОЛЬЗУЕТ СПИСОК
                if is_free and m_id not in BLACKLISTED_MODELS:
                    free_models_data.append({
                        'id': m_id,
                        'context': context_length
                    })

            # Сортировка: сначала большой контекст, потом по алфавиту
            sorted_models = sorted(
                free_models_data,
                key=lambda x: (-x['context'], x['id'])
            )

            return [m['id'] for m in sorted_models]

    except Exception as e:
        print(f"⚠️ Ошибка при запросе моделей: {e}")
    return None

# Изначально заполняем из твоего списка
# (если API не ответит, бот не останется с пустым меню)
current_free_or_models = OPENROUTER_MODELS.copy()


def update_model_mappings():
    global OPENROUTER_MODEL_BY_ID, current_free_or_models, BLACKLISTED_MODELS

    # ПЕРЕД обновлением очищаем блэклист, чтобы дать моделям "второй шанс"
    BLACKLISTED_MODELS.clear()

    new_models = fetch_free_openrouter_models()
    if new_models:
        current_free_or_models = new_models

    OPENROUTER_MODEL_BY_ID.clear()
    for i, path in enumerate(current_free_or_models):
        OPENROUTER_MODEL_BY_ID[str(i + 100)] = path
    print(f"🔄 Списки моделей обновлены. Дали второй шанс всем упавшим моделям.")

# Первичная инициализация словарей
GEMINI_MODEL_BY_ID = {str(i): path for i, path in enumerate(GEMINI_MODELS)}
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
Ты — ИИ помощник.Текущая дата={TO_DAY}.Старайся ответить коротко. 
1. Точная информация + фактчекинг.Проверка новостей на {TO_DAY}.Укажи на сколько % это правда.
2. Стандартный ответ 100 зн, если просят развернутый или подробный игнорируй ограничение.
3. Если требуется, можешь просматривать статьи в интернете и искать факты.
4. Уместный тонкий английский юмор 8 из 10, подколы разрешены.
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
    # 1. Обновляем маппинги (список моделей)
    update_model_mappings()

    # 2. Получаем ID пользователя (универсально для команд и кнопок)
    user_id = update.effective_user.id
    keyboard = []

    # --- Секция Gemini ---
    keyboard.append([InlineKeyboardButton("✨ Gemini Models:", callback_data="dummy")])
    for i, model in enumerate(GEMINI_MODELS):
        name = get_model_short_name(model, "gemini")
        prefix = "✅ " if user_selected_model[user_id] == model else ""
        keyboard.append([InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:g:{i}")])

    keyboard.append([InlineKeyboardButton("──────────────", callback_data="dummy")])

    # --- Секция OpenRouter (динамическая) ---
    keyboard.append([InlineKeyboardButton("🎁 OpenRouter FREE:", callback_data="dummy")])
    for i, model in enumerate(current_free_or_models):
        name = get_model_short_name(model, "openrouter")
        prefix = "✅ " if user_selected_model[user_id] == model else ""
        keyboard.append([
            InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:o:{i + 100}")
        ])

    keyboard.append([InlineKeyboardButton("🤖 Автоматический выбор", callback_data="sel:auto")])

    reply_markup = InlineKeyboardMarkup(keyboard)

    # --- УНИВЕРСАЛЬНАЯ ОТПРАВКА ---
    # effective_message сам определит, откуда отвечать (на сообщение или на кнопку)
    if update.effective_message:
        await update.effective_message.reply_text(
            "Выбери модель из актуального списка:",
            reply_markup=reply_markup
        )
    else:
        # Резервный вариант через прямой вызов метода бота
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Выбери модель из актуального списка:",
            reply_markup=reply_markup
        )
async def callback_handler(update: Update, context):
    query = update.callback_query
    await query.answer()

    data = query.data
    user_id = query.from_user.id

    if data == "open_menu":
        # Передаем update целиком, а не query
        await show_model_selection(update, context)
        return

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

async def process_llm(update: Update, context, final_query: str, thread_id=None):
    if not final_query.strip():
        return

    chat_id = update.effective_chat.id
    reply_to_message_id = update.effective_message.message_id
    user_id = update.effective_user.id

    history = chat_histories[chat_id]
    history.append(Content(role="user", parts=[types.Part(text=final_query)]))
    chat_histories[chat_id] = history[-4:]

    if thread_id is None and update.effective_message:
        thread_id = update.effective_message.message_thread_id

    status_msg = await context.bot.send_message(
        chat_id=chat_id,
        text="⚡ Запускаю модели...",
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
    # --- БЛОК ФАКТЧЕКИНГА В process_llm ---
    check_words = ["правда", "фейк", "чекай", "проверь", "реально", "серьезно", "что думаешь", "новости"]

    # Теперь фактчекинг включается ЕСЛИ:
    # 1. Есть ключевое слово
    # 2. ИЛИ есть метка "ОБЪЕКТ ПРОВЕРКИ" (которую мы вешаем в handle_private для пересланных постов)
    is_factcheck = ("ОБЪЕКТ ПРОВЕРКИ:" in final_query) or any(word in final_query.lower() for word in check_words)

    if is_factcheck:
        try:
            # 1. Выделяем чистую новость для поиска
            if "ОБЪЕКТ ПРОВЕРКИ:" in final_query:
                # Берем только текст между метками
                search_query = final_query.split("ОБЪЕКТ ПРОВЕРКИ:")[1].split("\n\nВОПРОС:")[0].strip()
            else:
                search_query = final_query

                # Если запрос пустой (например, юзер переслал пустую картинку), выходим из поиска
            if not search_query:
                raise ValueError("Пустой запрос для поиска")

            # 2. Визуальный фидбек в телеграм
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=status_id,
                text=f"🔍 Проверяю новость:\n«{search_query[:60]}...»"
            )

            # 3. Запрос в web_utils (ищем ТОЛЬКО новость)
            web_data = await get_web_context(search_query)

            if web_data:
                # Экранируем спецсимволы, чтобы не ломать HTML верстку бота
                clean_web_data = web_data.replace("<", "&lt;").replace(">", "&gt;")

                # Формируем расширенный промпт для нейросети
                # В блоке if web_data внутри process_llm:
                final_query = (
                    f"ДАННЫЕ МОНИТОРИНГА СМИ:\n{web_data}\n\n"
                    f"ОБЪЕКТ ПРОВЕРКИ: \"{search_query}\"\n\n"
                    f"ЗАДАНИЕ:\n"
                    f"1. Проанализируй, какие СМИ из списка 'Доверенных' написали об этом, а какие проигнорировали.\n"
                    f"2. Сделай вывод: если новость есть только в одном типе СМИ (например, только в государственных или только в оппозиционных), это повод для сомнения.\n"
                    f"3. Выдай краткий ответ. В конце добавь блок 'Рейтинг упоминаний', где наглядно покажи, кто из 12 источников подтвердил инфо."
                )

                # Обновляем историю сообщений для модели
                history[-1] = Content(role="user", parts=[types.Part(text=final_query)])
            else:
                # Если поиск ничего не дал, можно добавить легкий сарказм или просьбу уточнить
                print(f"🔍 Поиск по запросу '{search_query}' не дал результатов.")
        except Exception as e:
            print(f"⚠️ Ошибка фактчекинга: {e}")

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

                response = or_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    temperature=0.75,
                    max_tokens=4000
                )
                if response.choices and response.choices[0].message.content:
                    reply_text = response.choices[0].message.content.strip()
                    used_provider = "OpenRouter"
                    used_model_path = selected_model

        except Exception as e:
            model_name = get_model_short_name(selected_model, selected_provider)
            # Создаем кнопку, которая просто вызовет меню моделей
            keyboard = [[InlineKeyboardButton("🔄 Выбрать другую модель", callback_data="open_menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_id,
                text=f"❌ Модель {model_name} сейчас недоступна.\nПопробуй выбрать другую из списка ниже:",
                reply_markup=reply_markup
            )
            return  # Прерываем, сообщение НЕ удаляем

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

            for model_path in current_free_or_models:
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=status_id,
                        text=f"🔄 OR: {model_path.split('/')[-1].split(':')[0]}..."
                    )
                    response = or_client.chat.completions.create(
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

    MAX_LEN = 4000
    if len(full_reply) <= MAX_LEN:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_id,
                text=full_reply,
                parse_mode="HTML",  # Пытаемся отправить красиво
                disable_web_page_preview=True
            )
        except Exception as e:
            print(f"⚠️ Ошибка HTML: {e}")
            # Если не вышло (кривые теги), отправляем чистый текст
            # Strip tags - удаляем теги, чтобы не было мусора <b>
            clean_reply = re.sub(r'<[^>]+>', '', full_reply)
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_id,
                text=clean_reply,  # Отправляем без HTML
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

    # 1. Проверка авторизации (твой код без изменений)
    if user_id not in authorized_users:
        text = (message.text or "").strip()
        if text.lower() == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await message.reply_text(
                "Авторизация пройдена! 🎉\nТеперь можешь задавать вопросы.\n\n/model — выбор модели")
        else:
            await message.reply_text("Неправильный пароль 😕\nИспользуй /start")
        return

    # 2. Извлечение текста
    query_text = message.text or message.caption or ""

    if not query_text.strip():
        await message.reply_text("Я вижу сообщение, но не вижу в нём текста для анализа. 😏")
        return

    # Проверяем, является ли сообщение пересланным
    # В новых версиях PTB используется forward_origin
    is_forwarded = bool(message.forward_origin)

    if is_forwarded:
        # Если переслано — помечаем как объект для фактчекинга
        final_prompt = f"ОБЪЕКТ ПРОВЕРКИ: {query_text}\n\nВОПРОС: Это правда?"
    else:
        # Обычное сообщение
        final_prompt = query_text


    # 3. Отправка в LLM
    await process_llm(update, context, final_prompt)

async def handle_group(update: Update, context):
    message = update.message
    if not message:
        return

    content = message.text or message.caption or ""
    if not content:
        return

    # --- 1. ПРОВЕРКА: КТО КОГО ПОЗВАЛ ---
    is_reply_to_me = False
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.username == BOT_USERNAME:
            is_reply_to_me = True

    TRIGGERS = ["инспектор", "шелупонь", "ботик", "бубен", "андрюха", "андрей", "малыш", "андрей генадьевич"]
    content_lower = content.lower().strip()

    # Регулярка для поиска триггера именно в начале
    has_trigger_word = any(re.search(rf'^\s*\b{re.escape(word)}\b', content_lower) for word in TRIGGERS)
    is_mentioned = is_bot_mentioned(message, BOT_USERNAME)

    # Условие входа (если не к боту — выходим)
    if not (has_trigger_word or is_mentioned or is_reply_to_me):
        return

    # --- 2. ОЧИСТКА ТЕКСТА ---
    clean_text = content
    entities = (message.entities or []) + (message.caption_entities or [])

    # Убираем @mention
    for entity in entities:
        if entity.type == "mention":
            mention = content[entity.offset: entity.offset + entity.length]
            if mention.lower() == f"@{BOT_USERNAME.lower()}":
                clean_text = clean_text.replace(mention, "", 1)
                break

    # Убираем триггерное слово
    for word in TRIGGERS:
        clean_text = re.sub(rf'^\s*\b{re.escape(word)}\b[,\.\s\-]*', '', clean_text, flags=re.IGNORECASE, count=1)

    clean_text = re.sub(r'^[,\.\s?!\-]+', '', clean_text).strip()

    # --- 3. ФОРМИРОВАНИЕ ПРОМПТА ---
    prompt = ""
    if message.reply_to_message:
        reply = message.reply_to_message
        reply_text = reply.text or reply.caption or ""

        if reply_text:
            # Проверка на ключевые слова фактчекинга
            check_keywords = ["правда", "фейк", "чекай", "проверь", "реально", "серьезно",'прокомментируй']
            is_check_request = any(word in clean_text.lower() for word in check_keywords)

            if is_check_request:
                # Помечаем для process_llm, что это объект для поиска в сети
                prompt = f"ОБЪЕКТ ПРОВЕРКИ: {reply_text}\n\nВОПРОС: "
            else:
                prompt = f"Контекст (ответ на сообщение): {reply_text}\n\n"

    prompt += clean_text

    # Если после чистки пусто и это не просто реплей боту — даем подсказку
    if not clean_text and not is_reply_to_me:
        await message.reply_text("Я тут! Задай свой вопрос после обращения 😏")
        return

    # --- 4. ОТПРАВКА В ОБРАБОТКУ ---
    thread_id = message.message_thread_id
    await process_llm(update, context, prompt, thread_id=thread_id)

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
