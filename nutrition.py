# nutrition.py
import sqlite3
import logging
from datetime import datetime, date

# --- Настройки Базы Данных ---
DB_NAME = "nutrition.db"

def get_db_connection():
    """Создает и возвращает соединение с БД. Позволяет обращаться к колонкам по имени."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Инициализирует таблицы в базе данных, если они не существуют."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Таблица 1: Профили пользователей с их целями и нормами
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id INTEGER PRIMARY KEY,
                    goal TEXT NOT NULL, -- 'recomposition', 'mass_gain', 'weight_loss'
                    age INTEGER,
                    gender TEXT,
                    height REAL,
                    weight REAL,
                    activity_level REAL,
                    target_calories INTEGER,
                    target_proteins INTEGER,
                    target_fats INTEGER,
                    target_carbs INTEGER,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица 2: Логи приемов пищи
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS food_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    calories INTEGER NOT NULL,
                    proteins INTEGER NOT NULL,
                    fats INTEGER NOT NULL,
                    carbs INTEGER NOT NULL,
                    description TEXT,
                    FOREIGN KEY (user_id) REFERENCES profiles (user_id)
                )
            """)
            
            conn.commit()
            logging.info("База данных 'nutrition.db' успешно инициализирована.")
            
    except sqlite3.Error as e:
        logging.error(f"Ошибка при инициализации 'nutrition.db': {e}")
        raise

# --- Логика Расчета Плана Питания ---

def calculate_nutrition_plan(profile_data: dict) -> dict:
    """
    Рассчитывает и возвращает дневную норму КБЖУ на основе цели пользователя.
    """
    weight = profile_data.get('weight', 70)
    height = profile_data.get('height', 175)
    age = profile_data.get('age', 30)
    gender = profile_data.get('gender', 'male')
    activity = profile_data.get('activity_level', 1.2)
    goal = profile_data.get('goal', 'recomposition')

    # Расчет BMR (Базовый обмен веществ) по формуле Миффлина-Сан Жеора
    if gender.lower() in ['муж', 'м', 'male']:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    # TDEE (Общие энергозатраты с учетом активности)
    tdee = bmr * activity

    # Корректировка калорий в зависимости от цели
    if goal == 'weight_loss':
        target_calories = int(tdee * 0.80)  # Дефицит 20%
    elif goal == 'mass_gain':
        target_calories = int(tdee * 1.15)  # Профицит 15%
    else: # 'recomposition'
        target_calories = int(tdee * 0.90)  # Небольшой дефицит 10%

    # Расчет макронутриентов
    protein_multiplier = 1.8 if goal == 'mass_gain' else 2.0
    target_proteins = int(weight * protein_multiplier)
    target_fats = int(weight * 0.9)
    calories_from_protein_fat = (target_proteins * 4) + (target_fats * 9)
    target_carbs = int((target_calories - calories_from_protein_fat) / 4)

    return {
        'target_calories': target_calories,
        'target_proteins': target_proteins,
        'target_fats': target_fats,
        'target_carbs': target_carbs
    }

# --- Функции для работы с БД (CRUD) ---

def update_user_profile(user_id: int, profile_data: dict):
    """Создает или обновляет профиль пользователя в БД."""
    query = """
        INSERT OR REPLACE INTO profiles (
            user_id, goal, age, gender, height, weight, activity_level, 
            target_calories, target_proteins, target_fats, target_carbs, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        user_id,
        profile_data['goal'],
        profile_data['age'],
        profile_data['gender'],
        profile_data['height'],
        profile_data['weight'],
        profile_data['activity_level'],
        profile_data['target_calories'],
        profile_data['target_proteins'],
        profile_data['target_fats'],
        profile_data['target_carbs'],
        datetime.now()
    )
    try:
        with get_db_connection() as conn:
            conn.execute(query, params)
            conn.commit()
        logging.info(f"Профиль для user_id {user_id} обновлен.")
    except sqlite3.Error as e:
        logging.error(f"Ошибка при обновлении профиля {user_id}: {e}")

def get_user_profile(user_id: int) -> dict:
    """Возвращает профиль пользователя из БД в виде словаря."""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении профиля {user_id}: {e}")
        return None

def add_food_log(user_id: int, meal_data: dict):
    """Добавляет запись о приеме пищи в БД."""
    query = """
        INSERT INTO food_logs (user_id, calories, proteins, fats, carbs, description)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    params = (
        user_id,
        meal_data['calories'],
        meal_data['proteins'],
        meal_data['fats'],
        meal_data['carbs'],
        meal_data.get('description', '')
    )
    try:
        with get_db_connection() as conn:
            conn.execute(query, params)
            conn.commit()
        logging.info(f"Лог еды для user_id {user_id} добавлен.")
    except sqlite3.Error as e:
        logging.error(f"Ошибка при добавлении лога еды для {user_id}: {e}")

def get_daily_summary(user_id: int) -> dict:
    """Возвращает суммарное КБЖУ за сегодня."""
    today = date.today().strftime("%Y-%m-%d")
    query = """
        SELECT 
            SUM(calories) as total_calories,
            SUM(proteins) as total_proteins,
            SUM(fats) as total_fats,
            SUM(carbs) as total_carbs
        FROM food_logs 
        WHERE user_id = ? AND DATE(timestamp) = ?
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(query, (user_id, today))
            row = cursor.fetchone()
            # Возвращаем словарь с нулями, если за сегодня еще ничего не было съедено
            return {
                'total_calories': row['total_calories'] or 0,
                'total_proteins': row['total_proteins'] or 0,
                'total_fats': row['total_fats'] or 0,
                'total_carbs': row['total_carbs'] or 0
            } if row else {'total_calories': 0, 'total_proteins': 0, 'total_fats': 0, 'total_carbs': 0}
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении дневной сводки для {user_id}: {e}")
        return None

# Этот блок выполнится, если запустить файл напрямую: python nutrition.py
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Файл nutrition.py выполнен. База данных должна быть готова.")
