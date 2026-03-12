import sqlite3
import logging
from datetime import datetime

# --- Настройки Базы Данных ---
DB_NAME = "workouts.db"

def get_db_connection():
    """Создает и возвращает соединение с БД. Позволяет обращаться к колонкам по имени."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Инициализирует или обновляет структуру базы данных для тренировок."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Таблица для шаблонов тренировок
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workouts (
                    workout_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Таблица для упражнений в шаблоне тренировки
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exercises (
                    exercise_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workout_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    planned_sets INTEGER NOT NULL,
                    planned_reps TEXT NOT NULL,
                    comment TEXT,
                    order_in_workout INTEGER NOT NULL,
                    FOREIGN KEY (workout_id) REFERENCES workouts (workout_id) ON DELETE CASCADE
                )
            """)
            # Таблица для записей о выполненных тренировках
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logged_workouts (
                    logged_workout_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    workout_id INTEGER, -- Может быть NULL, если тренировка была "быстрой" или удалена
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    FOREIGN KEY (workout_id) REFERENCES workouts (workout_id) ON DELETE SET NULL
                )
            """)
            # Таблица для записей о выполненных подходах
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logged_sets (
                    logged_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_workout_id INTEGER NOT NULL,
                    exercise_id INTEGER, -- Может быть NULL, если упражнение было удалено из шаблона
                    set_number INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    reps_performed INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (logged_workout_id) REFERENCES logged_workouts (logged_workout_id) ON DELETE CASCADE,
                    FOREIGN KEY (exercise_id) REFERENCES exercises (exercise_id) ON DELETE SET NULL
                )
            """)
            conn.commit()
            logging.info("База данных 'workouts.db' успешно инициализирована/обновлена.")
    except sqlite3.Error as e:
        logging.error(f"Ошибка при инициализации 'workouts.db': {e}")
        raise

# --- Функции для работы с шаблонами тренировок ---

def add_workout_template(user_id: int, name: str) -> int:
    """Добавляет новый шаблон тренировки и возвращает его ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO workouts (user_id, name) VALUES (?, ?)",
            (user_id, name)
        )
        conn.commit()
        return cursor.lastrowid

def get_workout_templates(user_id: int) -> list[dict]:
    """Возвращает все шаблоны тренировок для пользователя."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT workout_id, name FROM workouts WHERE user_id = ?",
            (user_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

def get_workout_template_by_id(workout_id: int, user_id: int) -> dict | None:
    """Возвращает шаблон тренировки по ID."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT workout_id, name FROM workouts WHERE workout_id = ? AND user_id = ?",
            (workout_id, user_id)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

def update_workout_template_name(workout_id: int, user_id: int, new_name: str):
    """Обновляет название шаблона тренировки."""
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE workouts SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE workout_id = ? AND user_id = ?",
            (new_name, workout_id, user_id)
        )
        conn.commit()

def delete_workout_template(workout_id: int, user_id: int):
    """Удаляет шаблон тренировки и все связанные упражнения."""
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM workouts WHERE workout_id = ? AND user_id = ?",
            (workout_id, user_id)
        )
        conn.commit()

# --- Функции для работы с упражнениями в шаблоне ---

def add_exercise_to_workout(
    workout_id: int, name: str, planned_sets: int, planned_reps: str, comment: str | None
) -> int:
    """Добавляет упражнение к шаблону тренировки."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Определяем следующий порядок
        cursor.execute(
            "SELECT MAX(order_in_workout) FROM exercises WHERE workout_id = ?",
            (workout_id,)
        )
        max_order = cursor.fetchone()[0]
        next_order = (max_order if max_order is not None else 0) + 1

        cursor.execute(
            "INSERT INTO exercises (workout_id, name, planned_sets, planned_reps, comment, order_in_workout) VALUES (?, ?, ?, ?, ?, ?)",
            (workout_id, name, planned_sets, planned_reps, comment, next_order)
        )
        conn.commit()
        return cursor.lastrowid

def get_exercises_for_workout(workout_id: int) -> list[dict]:
    """Возвращает все упражнения для данного шаблона тренировки, отсортированные по порядку."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT exercise_id, name, planned_sets, planned_reps, comment, order_in_workout FROM exercises WHERE workout_id = ? ORDER BY order_in_workout",
            (workout_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

def get_exercise_by_id(exercise_id: int) -> dict | None:
    """Возвращает упражнение по ID."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT exercise_id, workout_id, name, planned_sets, planned_reps, comment, order_in_workout FROM exercises WHERE exercise_id = ?",
            (exercise_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

def update_exercise(
    exercise_id: int, name: str, planned_sets: int, planned_reps: str, comment: str | None
):
    """Обновляет данные упражнения."""
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE exercises SET name = ?, planned_sets = ?, planned_reps = ?, comment = ? WHERE exercise_id = ?",
            (name, planned_sets, planned_reps, comment, exercise_id)
        )
        conn.commit()

def delete_exercise(exercise_id: int):
    """Удаляет упражнение из шаблона."""
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM exercises WHERE exercise_id = ?",
            (exercise_id,)
        )
        conn.commit()

# --- Функции для работы с выполненными тренировками и подходами ---

def start_logged_workout(user_id: int, workout_id: int | None) -> int:
    """Начинает запись новой выполненной тренировки."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logged_workouts (user_id, workout_id) VALUES (?, ?)",
            (user_id, workout_id)
        )
        conn.commit()
        return cursor.lastrowid

def end_logged_workout(logged_workout_id: int):
    """Завершает запись выполненной тренировки, устанавливая end_time."""
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE logged_workouts SET end_time = CURRENT_TIMESTAMP WHERE logged_workout_id = ?",
            (logged_workout_id,)
        )
        conn.commit()

def add_logged_set(
    logged_workout_id: int, exercise_id: int | None, set_number: int, weight: float, reps_performed: int
):
    """Добавляет запись о выполненном подходе."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO logged_sets (logged_workout_id, exercise_id, set_number, weight, reps_performed) VALUES (?, ?, ?, ?, ?)",
            (logged_workout_id, exercise_id, set_number, weight, reps_performed)
        )
        conn.commit()

def get_last_set_data_for_exercise(user_id: int, exercise_id: int) -> dict | None:
    """
    Возвращает данные последнего выполненного подхода для конкретного упражнения
    (по его ID) для данного пользователя.
    """
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT
                ls.weight,
                ls.reps_performed
            FROM
                logged_sets ls
            JOIN
                logged_workouts lw ON ls.logged_workout_id = lw.logged_workout_id
            WHERE
                lw.user_id = ? AND ls.exercise_id = ?
            ORDER BY
                ls.timestamp DESC
            LIMIT 1
            """,
            (user_id, exercise_id)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

# --- Функции для статистики ---

def get_all_unique_exercises(user_id: int) -> list[dict]:
    """
    Возвращает список уникальных упражнений, которые пользователь когда-либо выполнял.
    """
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT DISTINCT
                e.exercise_id,
                e.name
            FROM
                exercises e
            JOIN
                logged_sets ls ON e.exercise_id = ls.exercise_id
            JOIN
                logged_workouts lw ON ls.logged_workout_id = lw.logged_workout_id
            WHERE
                lw.user_id = ?
            ORDER BY
                e.name
            """,
            (user_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

def get_exercise_progression(user_id: int, exercise_id: int) -> list[dict]:
    """
    Возвращает полную историю выполненных подходов для конкретного упражнения.
    """
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT
                ls.weight,
                ls.reps_performed,
                lw.start_time as workout_date
            FROM
                logged_sets ls
            JOIN
                logged_workouts lw ON ls.logged_workout_id = lw.logged_workout_id
            WHERE
                lw.user_id = ? AND ls.exercise_id = ?
            ORDER BY
                lw.start_time, ls.set_number
            """,
            (user_id, exercise_id)
        )
        return [dict(row) for row in cursor.fetchall()]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Файл workouts.py выполнен. База данных 'workouts.db' должна быть готова.")
