# nutrition.py
import sqlite3
import logging
from datetime import datetime, date, timedelta
from collections import defaultdict
import pytz

# --- Настройки Базы Данных ---
DB_NAME = "nutrition.db"
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

def get_db_connection():
    """Создает и возвращает соединение с БД. Позволяет обращаться к колонкам по имени."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # ... (код без изменений)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id INTEGER PRIMARY KEY, goal TEXT NOT NULL, age INTEGER, gender TEXT,
                    height REAL, weight REAL, activity_level REAL, target_calories INTEGER,
                    target_proteins INTEGER, target_fats INTEGER, target_carbs INTEGER,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS food_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, calories INTEGER NOT NULL,
                    proteins INTEGER NOT NULL, fats INTEGER NOT NULL, carbs INTEGER NOT NULL,
                    description TEXT, FOREIGN KEY (user_id) REFERENCES profiles (user_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nutrition_adjustments (
                    adjustment_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
                    date DATE NOT NULL, calories_adjustment INTEGER DEFAULT 0,
                    proteins_adjustment INTEGER DEFAULT 0, fats_adjustment INTEGER DEFAULT 0,
                    carbs_adjustment INTEGER DEFAULT 0, UNIQUE(user_id, date),
                    FOREIGN KEY (user_id) REFERENCES profiles (user_id)
                )
            """)
            conn.commit()
            logging.info("База данных 'nutrition.db' успешно инициализирована/обновлена.")
    except sqlite3.Error as e:
        logging.error(f"Ошибка при инициализации 'nutrition.db': {e}")
        raise

def get_now_in_moscow() -> datetime:
    """Возвращает текущее время в Московском часовом поясе."""
    return datetime.now(MOSCOW_TZ)

# --- Логика Расчета Плана Питания ---
def calculate_nutrition_plan(profile_data: dict) -> dict:
    # ... (код без изменений)
    weight = profile_data.get('weight', 70)
    height = profile_data.get('height', 175)
    age = profile_data.get('age', 30)
    gender = profile_data.get('gender', 'male')
    activity = profile_data.get('activity_level', 1.2)
    goal = profile_data.get('goal', 'recomposition')
    if gender.lower() in ['муж', 'м', 'male']:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    tdee = bmr * activity
    if goal == 'weight_loss':
        target_calories = int(tdee * 0.80)
    elif goal == 'mass_gain':
        target_calories = int(tdee * 1.15)
    else:
        target_calories = int(tdee * 0.90)
    protein_multiplier = 1.8 if goal == 'mass_gain' else 2.0
    target_proteins = int(weight * protein_multiplier)
    target_fats = int(weight * 0.9)
    calories_from_protein_fat = (target_proteins * 4) + (target_fats * 9)
    target_carbs = int((target_calories - calories_from_protein_fat) / 4)
    return {
        'target_calories': target_calories, 'target_proteins': target_proteins,
        'target_fats': target_fats, 'target_carbs': target_carbs
    }

# --- Функции для работы с БД (CRUD) ---
def update_user_profile(user_id: int, profile_data: dict):
    """UPDATED: Использует время MSK при обновлении профиля."""
    query = """
        INSERT OR REPLACE INTO profiles (
            user_id, goal, age, gender, height, weight, activity_level, 
            target_calories, target_proteins, target_fats, target_carbs, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        user_id, profile_data['goal'], profile_data['age'], profile_data['gender'],
        profile_data['height'], profile_data['weight'], profile_data['activity_level'],
        profile_data['target_calories'], profile_data['target_proteins'],
        profile_data['target_fats'], profile_data['target_carbs'], get_now_in_moscow()
    )
    try:
        with get_db_connection() as conn:
            conn.execute(query, params)
            conn.commit()
        logging.info(f"Профиль для user_id {user_id} обновлен.")
    except sqlite3.Error as e:
        logging.error(f"Ошибка при обновлении профиля {user_id}: {e}")

def get_user_profile(user_id: int) -> dict:
    # ... (код без изменений)
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
            return dict(row) if row else None
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении профиля {user_id}: {e}")
        return None

def add_food_log(user_id: int, meal_data: dict):
    """UPDATED: Использует время MSK при добавлении лога."""
    query = "INSERT INTO food_logs (user_id, timestamp, calories, proteins, fats, carbs, description) VALUES (?, ?, ?, ?, ?, ?, ?)"
    params = (
        user_id, get_now_in_moscow(), meal_data['calories'], meal_data['proteins'],
        meal_data['fats'], meal_data['carbs'], meal_data.get('description', '')
    )
    try:
        with get_db_connection() as conn:
            conn.execute(query, params)
            conn.commit()
        logging.info(f"Лог еды для user_id {user_id} добавлен.")
    except sqlite3.Error as e:
        logging.error(f"Ошибка при добавлении лога еды для {user_id}: {e}")

def get_daily_summary(user_id: int, on_date: date = None) -> dict:
    """UPDATED: Учитывает, что даты в БД могут быть в разных форматах."""
    target_date = on_date or get_now_in_moscow().date()
    query = "SELECT SUM(calories), SUM(proteins), SUM(fats), SUM(carbs) FROM food_logs WHERE user_id = ? AND DATE(timestamp) = ?"
    try:
        with get_db_connection() as conn:
            row = conn.execute(query, (user_id, target_date.strftime("%Y-%m-%d"))).fetchone()
            return {
                'total_calories': row[0] or 0, 'total_proteins': row[1] or 0,
                'total_fats': row[2] or 0, 'total_carbs': row[3] or 0
            }
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении дневной сводки для {user_id}: {e}")
        return {'total_calories': 0, 'total_proteins': 0, 'total_fats': 0, 'total_carbs': 0}

def get_adjusted_target(user_id: int, on_date: date = None) -> dict:
    # ... (код без изменений)
    target_date = on_date or get_now_in_moscow().date()
    base_profile = get_user_profile(user_id)
    if not base_profile: return None
    try:
        with get_db_connection() as conn:
            adj_row = conn.execute("SELECT * FROM nutrition_adjustments WHERE user_id = ? AND date = ?", (user_id, target_date.strftime("%Y-%m-%d"))).fetchone()
        if adj_row:
            return {
                'target_calories': base_profile['target_calories'] + adj_row['calories_adjustment'],
                'target_proteins': base_profile['target_proteins'] + adj_row['proteins_adjustment'],
                'target_fats': base_profile['target_fats'] + adj_row['fats_adjustment'],
                'target_carbs': base_profile['target_carbs'] + adj_row['carbs_adjustment']
            }
        return {k: base_profile[k] for k in ['target_calories', 'target_proteins', 'target_fats', 'target_carbs']}
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении скорректированной цели для {user_id}: {e}")
        return None

def get_remaining_macros(user_id: int) -> dict:
    # ... (код без изменений)
    todays_target = get_adjusted_target(user_id, on_date=get_now_in_moscow().date())
    if not todays_target: return None 
    todays_summary = get_daily_summary(user_id, on_date=get_now_in_moscow().date())
    return {
        'remaining_calories': todays_target['target_calories'] - todays_summary['total_calories'],
        'remaining_proteins': todays_target['target_proteins'] - todays_summary['total_proteins'],
        'remaining_fats': todays_target['target_fats'] - todays_summary['total_fats'],
        'remaining_carbs': todays_target['target_carbs'] - todays_summary['total_carbs']
    }

def apply_cheat_meal_plan(user_id: int, cheat_meal_data: dict):
    """UPDATED: Использует время MSK."""
    if not cheat_meal_data: return
    try:
        with get_db_connection() as conn:
            log_query = "INSERT INTO food_logs (user_id, timestamp, calories, proteins, fats, carbs, description) VALUES (?, ?, ?, ?, ?, ?, ?)"
            log_params = (
                user_id, get_now_in_moscow(), cheat_meal_data['calories'], cheat_meal_data['proteins'],
                cheat_meal_data['fats'], cheat_meal_data['carbs'],
                cheat_meal_data.get('description', 'Читмил')
            )
            conn.execute(log_query, log_params)
            
            today_str = get_now_in_moscow().date().strftime("%Y-%m-%d")
            adj_query = "INSERT INTO nutrition_adjustments (user_id, date, calories_adjustment) VALUES (?, ?, ?) ON CONFLICT(user_id, date) DO UPDATE SET calories_adjustment = calories_adjustment + excluded.calories_adjustment;"
            conn.execute(adj_query, (user_id, today_str, cheat_meal_data['calories']))
            
            conn.commit()
        logging.info(f"План спасения (сегодняшний) для {user_id} применен.")
    except sqlite3.Error as e:
        logging.error(f"Ошибка при применении плана спасения для {user_id}: {e}")

def get_historical_summary(user_id: int, days: int = 7) -> dict:
    """UPDATED: Использует время MSK."""
    summaries = defaultdict(lambda: {'calories': 0, 'proteins': 0, 'fats': 0, 'carbs': 0})
    start_date = get_now_in_moscow().date() - timedelta(days=days - 1)
    
    query = """
        SELECT DATE(timestamp) as log_date, SUM(calories), SUM(proteins), SUM(fats), SUM(carbs)
        FROM food_logs
        WHERE user_id = ? AND DATE(timestamp) >= ?
        GROUP BY log_date
        ORDER BY log_date ASC
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(query, (user_id, start_date.strftime("%Y-%m-%d")))
            for row in cursor.fetchall():
                log_date = row['log_date']
                summaries[log_date] = {
                    'calories': row[1] or 0, 'proteins': row[2] or 0,
                    'fats': row[3] or 0, 'carbs': row[4] or 0
                }
        return summaries
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении исторической сводки для {user_id}: {e}")
        return {}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Файл nutrition.py выполнен. База данных должна быть готова.")
