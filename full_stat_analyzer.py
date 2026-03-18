import logging
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import groupby
from operator import itemgetter

# Предполагается, что эти функции существуют и возвращают данные в нужном формате
from nutrition import get_user_profile, get_historical_summary
from workouts import get_exercise_progression, get_all_unique_exercises

# --- Настройки ---
ANALYSIS_PERIOD_DAYS = 60
TOP_N_EXERCISES = 5 # Количество ключевых упражнений для анализа

def _calculate_one_rep_max(weight: float, reps: int) -> float:
    """
    Рассчитывает одноповторный максимум (1ПМ) по формуле Эпли.
    """
    if reps < 1:
        return 0
    # Формула Эпли: вес * (1 + повторения / 30)
    return weight * (1 + reps / 30)

async def get_data_for_full_analysis(user_id: int) -> dict:
    """
    Собирает, агрегирует и структурирует данные о питании, тренировках и профиле
    пользователя для последующего анализа ИИ.
    """
    logging.info(f"Начинаем сбор данных для полного анализа для user_id: {user_id}")
    
    final_data = {}

    # 1. Получение профиля пользователя
    logging.info("Шаг 1: Получение профиля пользователя...")
    user_profile = get_user_profile(user_id)
    if user_profile:
        final_data['user_profile'] = {
            'age': user_profile.get('age'),
            'height': user_profile.get('height'),
            'weight': user_profile.get('weight'),
            'goal': user_profile.get('goal')
        }
    else:
        logging.warning(f"Профиль для user_id {user_id} не найден.")
        final_data['user_profile'] = {}

    # 2. Анализ питания
    logging.info("Шаг 2: Анализ питания...")
    nutrition_summary = get_historical_summary(user_id, days=ANALYSIS_PERIOD_DAYS)
    total_calories, total_proteins, total_fats, total_carbs = 0, 0, 0, 0
    days_with_logs = len(nutrition_summary)

    if days_with_logs > 0:
        for day_data in nutrition_summary.values():
            total_calories += day_data.get('calories', 0)
            total_proteins += day_data.get('proteins', 0)
            total_fats += day_data.get('fats', 0)
            total_carbs += day_data.get('carbs', 0)
        
        final_data['nutrition_summary'] = {
            'average_calories': round(total_calories / days_with_logs),
            'average_proteins': round(total_proteins / days_with_logs),
            'average_fats': round(total_fats / days_with_logs),
            'average_carbs': round(total_carbs / days_with_logs),
            'logging_days': days_with_logs
        }
    else:
        logging.info(f"Нет данных о питании для user_id {user_id} за последние {ANALYSIS_PERIOD_DAYS} дней.")
        final_data['nutrition_summary'] = {}

    # 3. Анализ тренировок
    logging.info("Шаг 3: Анализ тренировок...")
    all_exercises = get_all_unique_exercises(user_id)
    exercise_peak_rms = []

    for exercise in all_exercises:
        progression = get_exercise_progression(user_id, exercise['exercise_id'])
        if not progression:
            continue
        
        max_rm = 0
        for s in progression:
            rm = _calculate_one_rep_max(s['weight'], s['reps_performed'])
            if rm > max_rm:
                max_rm = rm
        
        if max_rm > 0:
            exercise_peak_rms.append({'exercise_id': exercise['exercise_id'], 'name': exercise['name'], 'peak_rm': max_rm})

    # Сортируем упражнения по пиковому 1ПМ и берем топ-N
    top_exercises = sorted(exercise_peak_rms, key=itemgetter('peak_rm'), reverse=True)[:TOP_N_EXERCISES]

    progression_analysis = {}
    for top_ex in top_exercises:
        progression = get_exercise_progression(user_id, top_ex['exercise_id'])
        
        # Группируем подходы по дате тренировки
        grouped_by_date = groupby(progression, key=lambda x: datetime.fromisoformat(x['workout_date']).date())
        
        best_sets_per_day = []
        for date, sets in grouped_by_date:
            # Для каждого дня находим подход с лучшим 1ПМ
            best_set_for_day = max(
                (s for s in sets if s['reps_performed'] > 0), 
                key=lambda s: _calculate_one_rep_max(s['weight'], s['reps_performed']),
                default=None
            )
            if best_set_for_day:
                best_sets_per_day.append({
                    'date': date.isoformat(),
                    'weight': best_set_for_day['weight'],
                    'reps': best_set_for_day['reps_performed'],
                    'estimated_1rm': round(_calculate_one_rep_max(best_set_for_day['weight'], best_set_for_day['reps_performed']), 2)
                })
        
        # Сортируем историю по дате
        sorted_history = sorted(best_sets_per_day, key=itemgetter('date'))
        progression_analysis[top_ex['name']] = sorted_history

    final_data['workout_summary'] = {
        'progression_analysis': progression_analysis
    }

    logging.info(f"Сбор данных для user_id: {user_id} завершен.")
    return final_data
