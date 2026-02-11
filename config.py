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
CORRECT_PASSWORD = os.getenv("Password")

# URL твоего воркера на Cloudflare
WORKER_URL = "https://inspectorgpt.classname1984.workers.dev"

# ─── Модели по умолчанию (Fallback) ───────────────────────
# Эти списки используются, если API временно недоступно
DEFAULT_OPENROUTER_MODELS = [
    "arcee-ai/trinity-large-preview:free", # Лидер: умная, живая, русский 10/10
    "openrouter/pony-alpha",  # Хороший разговорный стиль
    "stepfun/step-3.5-flash:free",         # Скорость + логика
    "google/gemma-2-9b-it:free",  # Очень стабильная классика
    'tngtech/deepseek-r1t2-chimera:free',
    'tngtech/deepseek-r1t-chimera:free',
    'z-ai/glm-4.5-air:free'
    "qwen/qwen-2-7b-instruct:free",        # Отличное понимание русского
]

GEMINI_MODELS = [
    "gemini-2.5-flash",        # №1 Скорость + Качество
    "gemini-3-flash-preview",  # №2 Новое поколение
    "gemini-2.5-flash-lite",   # №3 Экономия ресурсов
    "gemini-2.0-flash"         # №4 Проверенная стабильность
]


# ─── Промпты ─────────────────────────────────────────────
TO_DAY = datetime.now().isoformat()

# config.py

# Для обычного общения
SYSTEM_PROMPT_CHAT = (f"Сегодня = {TO_DAY},Ты — дружелюбный и остроумный ИИ-ассистент.У тебя характер джарвиса из железного человека."
                      "Отвечай кратко, по делу, с юмором и сарказмом по теме, если уместно."
                      "Форматирование Telegram.Язык русский"
                      "Длина ответа максимум 4000 символов")

# Для проверки фактов
SYSTEM_PROMPT_INSPECTOR = (
    f"Сегодня = {TO_DAY}, Ты — профессиональный фактчекер и аналитик. Твоя задача: критически оценить предоставленный текст. "
    "Используй данные из поиска. Указывай вероятность правды, можно писать %. Язык русский"
    "Будь беспристрастным, указывай на логические неувязки. Умести ответ в 4000 символов. Форматирование Telegram"
)

# ─── Триггеры и настройки бота ──────────────────────────
TRIGGERS = ["инспектор", "шелупонь", "ботик", "бубен", "андрюха", "андрей", "малыш", "андрей генадьевич", 'официант']
CHECK_WORDS = ["чекай", "проверь", "факты", "новости"]
AUTH_QUESTION = "Тут у нас пароль. Нужно отгадать загадку. Скажи, за какое время разгоняется нива до 100 км/ч"
FINANCE_WORDS = ["счет", "запиши", "расходы", 'фиксируй']

# Настройки таймаутов
API_TIMEOUT = 45.0