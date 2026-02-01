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
    'nvidia/nemotron-3-nano-30b-a3b:free',
    'mistralai/devstral-2512:free',
    'tngtech/deepseek-r1t2-chimera:free',
    'google/gemma-3-27b-it:free',
    'z-ai/glm-4.5-air:free',
]

GEMINI_MODELS = [
    "models/gemini-2.0-flash",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash",
    "models/gemini-3-flash-preview",
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
TRIGGERS = ["инспектор", "шелупонь", "ботик", "бубен", "андрюха", "андрей", "малыш", "андрей генадьевич"]
CHECK_WORDS = ["чекай", "проверь", "факты", "новости"]
AUTH_QUESTION = "Тут у нас пароль. Нужно отгадать загадку. Скажи, за какое время разгоняется нива до 100 км/ч"
FINANCE_WORDS = ["счет", "запиши", "расходы", 'фиксируй']

# Настройки таймаутов
API_TIMEOUT = 45.0