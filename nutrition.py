# nutrition.py
import sqlite3
import logging
from datetime import datetime

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
    # Здесь будет реализована логика расчета в зависимости от 'goal'
    # Пока что это заглушка
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
    # Белок: 2 г/кг для похудения/рекомпозиции, 1.8 г/кг для набора массы
    protein_multiplier = 1.8 if goal == 'mass_gain' else 2.0
    target_proteins = int(weight * protein_multiplier)
    
    # Жиры: 0.8-1 г/кг
    target_fats = int(weight * 0.9)
    
    # Углеводы: все остальное
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
    # TODO: Реализовать сохранение данных в таблицу profiles
    pass

def get_user_profile(user_id: int) -> dict:
    """Возвращает профиль пользователя из БД."""
    # TODO: Реализовать получение данных из таблицы profiles
    pass

def add_food_log(user_id: int, meal_data: dict):
    """Добавляет запись о приеме пищи в БД."""
    # TODO: Реализовать добавление записи в таблицу food_logs
    pass

def get_daily_summary(user_id: int) -> dict:
    """Возвращает суммарное КБЖУ за сегодня."""
    # TODO: Реализовать получение и суммирование записей из food_logs за текущий день
    pass


# Этот блок выполнится, если запустить файл напрямую: python nutrition.py
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Файл nutrition.py выполнен. База данных должна быть готова.")
