import sqlite3
import logging
import os
from datetime import datetime, timedelta, timezone

# Путь к базе данных в корне проекта
DB_PATH = os.path.join(os.path.dirname(__file__), 'nutrition.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_moscow_date():
    # Создаем смещение UTC+3
    offset = timezone(timedelta(hours=3))
    # Получаем текущую дату в этом часовом поясе
    return datetime.now(offset).strftime('%Y-%m-%d')


def init_db():
    """Инициализация таблиц при старте бота"""
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # 1. АНКЕТА (Настройки и цели)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS user_profiles
                       (
                           user_id
                           INTEGER
                           PRIMARY
                           KEY,
                           age
                           INTEGER,
                           gender
                           TEXT,
                           height
                           REAL,
                           start_weight
                           REAL,
                           target_weight
                           REAL,
                           activity_level
                           REAL,
                           bmr
                           REAL,
                           tdee
                           REAL,
                           target_calories
                           INTEGER,
                           target_proteins
                           INTEGER,
                           target_fats
                           INTEGER,
                           target_carbs
                           INTEGER,
                           updated_at
                           DATETIME
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       """)

        # 2. ЕЖЕДНЕВНЫЙ ТРЕКЕР (Для графиков)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS daily_log
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           INTEGER,
                           date
                           DATE
                           DEFAULT
                       (
                           CURRENT_DATE
                       ),
                           current_weight REAL,
                           consumed_calories INTEGER DEFAULT 0,
                           consumed_proteins INTEGER DEFAULT 0,
                           consumed_fats INTEGER DEFAULT 0,
                           consumed_carbs INTEGER DEFAULT 0,
                           UNIQUE
                       (
                           user_id,
                           date
                       ),
                           FOREIGN KEY
                       (
                           user_id
                       ) REFERENCES user_profiles
                       (
                           user_id
                       )
                           )
                       """)
        conn.commit()
        logging.info("✅ База данных питания инициализирована.")
    except Exception as e:
        logging.error(f"❌ Ошибка инициализации БД: {e}")
    finally:
        conn.close()


def save_profile(user_id, data):
    """Сохранение или обновление анкеты пользователя"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        keys = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        # Динамическое создание SQL для вставки/обновления
        sql = f"INSERT OR REPLACE INTO user_profiles (user_id, {keys}) VALUES (?, {placeholders})"
        cursor.execute(sql, [user_id] + list(data.values()))
        conn.commit()
    finally:
        conn.close()


def update_daily_log(user_id, calories=0, proteins=0, fats=0, carbs=0, weight=None):
    date_today = get_moscow_date()  # Получаем "сегодня" по Москве
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Используем ? вместо CURRENT_DATE
        cursor.execute("""
            INSERT OR IGNORE INTO daily_log 
            (user_id, date, consumed_calories, consumed_proteins, consumed_fats, consumed_carbs) 
            VALUES (?, ?, 0, 0, 0, 0)
        """, (user_id, date_today))

        sql = """
            UPDATE daily_log 
            SET consumed_calories = consumed_calories + ?, 
                consumed_proteins = consumed_proteins + ?, 
                consumed_fats     = consumed_fats + ?, 
                consumed_carbs    = consumed_carbs + ?
        """
        params = [calories, proteins, fats, carbs]

        if weight:
            sql += ", current_weight = ?"
            params.append(weight)

        # Здесь тоже меняем на ?
        sql += " WHERE user_id = ? AND date = ?"
        params.extend([user_id, date_today])

        cursor.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def get_user_status(user_id):
    date_today = get_moscow_date()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.*,
                   l.consumed_calories,
                   l.consumed_proteins,
                   l.consumed_fats,
                   l.consumed_carbs,
                   l.current_weight
            FROM user_profiles p
            LEFT JOIN daily_log l ON p.user_id = l.user_id AND l.date = ?
            WHERE p.user_id = ?
        """, (date_today, user_id))
        return cursor.fetchone()
    finally:
        conn.close()