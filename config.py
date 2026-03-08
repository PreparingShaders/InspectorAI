#config
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ─── Ключи и API ──────────────────────────────────────────
BOT_TOKEN = os.getenv("InspectorGPT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
BOT_USERNAME = os.getenv("BOT_USERNAME", "").lstrip("@").lower()

# URL твоего воркера на Cloudflare
WORKER_URL = "https://inspectorgpt.classname1984.workers.dev"

# ─── Пользователи ────────────────────────────────────────
# Добавь сюда числовые ID пользователей Telegram, которым разрешен доступ
ALLOWED_USER_IDS = [
    435962963,
    430825078,
    941943738,
    1308761656,
    7723919865,
    7015658962,
]

# ─── Модели по умолчанию (Fallback) ───────────────────────
DEFAULT_OPENROUTER_MODELS = [
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
    'z-ai/glm-4.5-air:free',
    'nvidia/nemotron-3-nano-30b-a3b:free',
    'arcee-ai/trinity-mini:free',
    "openrouter/pony-alpha",
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    'gemini-3.1-flash-lite-preview',
    "gemini-3-flash-preview"
]

NUTRITION_MODELS = [
    # 2. СПЕЦИАЛИЗИРОВАННЫЕ VISION-МОДЕЛИ (Твоя потеряшка тут!)
    'gemini-3.1-flash-lite-preview',
    'gemini-2.5-flash-lite',
    'gemini-2.5-flash-image',  # Оптимизирована именно под работу с фото
    'gemini-3.1-flash-image-preview',  # Новейшая экспериментальная модель
    # 3. СТАБИЛЬНЫЙ БЭКАП (Быстрые и надежные)
    'gemini-2.5-flash',  # Высокая квота и скорость
    'gemini-2.0-flash',
    'openai/gpt-5-nano',
    'google/gemini-2.5-flash-lite',

    # 1. ТОП-УРОВЕНЬ (Твои находки с "мышлением")
    'qwen/qwen3-vl-235b-a22b-thinking',
    'qwen/qwen3-vl-30b-a3b-thinking',  # Чуть быстрее, но такая же "думающая",
    'nvidia/nemotron-nano-12b-v2-vl:free'

]


# ─── Промпты ─────────────────────────────────────────────
TO_DAY = datetime.now().isoformat()

# Для обычного общения (ОРИГИНАЛЬНАЯ ВЕРСИЯ)
SYSTEM_PROMPT_CHAT = (f"Сегодня = {TO_DAY},Ты — дружелюбный и остроумный ИИ-ассистент.У тебя характер джарвиса из железного человека."
                      "Отвечай кратко, по делу, с юмором и сарказмом по теме, если уместно."
                      "Форматирование Telegram.Язык русский"
                      "Длина ответа максимум 4000 символов")

# Для проверки фактов (ОРИГИНАЛЬНАЯ ВЕРСИЯ)
SYSTEM_PROMPT_INSPECTOR = (
    f"Сегодня = {TO_DAY}, Ты — профессиональный фактчекер и аналитик. Твоя задача: критически оценить предоставленный текст. "
    "Используй данные из поиска. Указывай вероятность правды, можно писать %. Язык русский"
    "Будь беспристрастным, указывай на логические неувязки. Умести ответ в 4000 символов. Форматирование Telegram"
)

# UPDATED: Для нутрициолога с характером
SYSTEM_PROMPT_NUTRITION = (
    "Ты — ИИ-нутрициолог с сарказмом. Твоя задача — оценить КБЖУ блюда."
    "1. Опиши блюдо, распиши что ты увидел. Оцени сколько в данном блюде может быть КБЖУ."
    "2. Если на фото не еда, скажи об этом.Твоя цель найти еду и оценить КБЖУ."
    "3. После комментария, ОБЯЗАТЕЛЬНО и без лишних слов, добавь JSON-объект в формате: "
    '```json\n{"calories": <число>, "proteins": <число>, "fats": <число>, "carbs": <число>}\n```'
)

# ─── Триггеры и настройки бота ──────────────────────────
TRIGGERS = ["инспектор", "шелупонь", "ботик", "бубен", "андрюха", "андрей", "малыш", "андрей генадьевич", 'официант']
CHECK_WORDS = ["чекай", "проверь", "факты", "новости"]
NUTRITION_TRIGGERS = ["еда", "завтрак", "обед", "ужин", "перекус", "жрать"]
AUTH_QUESTION = "К сожалению, у вас нет доступа к этому боту. Обратитесь к администратору."
FINANCE_WORDS = ["счет", "запиши", "расходы", 'фиксируй']

# Настройки таймаутов
API_TIMEOUT = 45.0