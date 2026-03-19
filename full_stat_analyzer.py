import logging
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import groupby
from operator import itemgetter

from nutrition import get_user_profile, get_historical_summary
from workouts import get_exercise_progression, get_all_unique_exercises

# --- Настройки ---
ANALYSIS_PERIOD_DAYS = 60
TOP_N_EXERCISES = 5
RECENT_ACTIVITY_DAYS = 21
EFFECTIVE_LOAD_THRESHOLD = 0.6  # 60% от 1ПМ

# --- Новые функции анализа ---

def _calculate_one_rep_max(weight: float, reps: int) -> float:
    if reps < 1: return 0
    if reps == 1: return weight
    return weight / (1.0278 - 0.0278 * reps)

def _filter_outliers(data: list, key: str) -> list:
    if len(data) < 5: return data
    values = [d[key] for d in data]
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [d for d in data if lower_bound <= d[key] <= upper_bound]

def _calculate_bmr(weight: float, height: float, age: int, gender: str = "male") -> float:
    # Формула Миффлина-Сан Жеора. Пол по умолчанию 'male' как допущение.
    if gender == "female":
        return 10 * weight + 6.25 * height - 5 * age - 161
    return 10 * weight + 6.25 * height - 5 * age + 5

def _get_exercise_tier_and_pr_window(relative_1rm: float) -> (int, int):
    if relative_1rm > 1.5: return (1, 30)  # Tier 1 (Присед, Становая)
    if 0.75 <= relative_1rm <= 1.5: return (2, 21)  # Tier 2 (Жим)
    return (3, 14)  # Tier 3 (Изоляция)

# --- Основная функция ---

async def get_data_for_full_analysis(user_id: int) -> dict:
    logging.info(f"Начинаем углубленный анализ для user_id: {user_id}")
    final_data = {}
    user_weight = 0

    # 1. Профиль и базовые расчеты
    user_profile = get_user_profile(user_id)
    if user_profile and user_profile.get('weight'):
        user_weight = user_profile['weight']
        final_data['user_profile'] = user_profile
        final_data['user_profile']['calculated_bmr'] = round(_calculate_bmr(
            user_weight, user_profile['height'], user_profile['age']
        ))
    else:
        logging.warning(f"Неполный профиль для user_id {user_id}.")
        final_data['user_profile'] = {}

    # 2. Анализ питания (без изменений)
    nutrition_summary = get_historical_summary(user_id, days=ANALYSIS_PERIOD_DAYS)
    if nutrition_summary:
        days_with_logs = len(nutrition_summary)
        totals = defaultdict(float)
        for day_data in nutrition_summary.values():
            for key in ['calories', 'proteins', 'fats', 'carbs']:
                totals[key] += day_data.get(key, 0)
        final_data['nutrition_summary'] = {
            f'average_{k}': round(v / days_with_logs) for k, v in totals.items()
        }
        final_data['nutrition_summary']['logging_days'] = days_with_logs
    else:
        final_data['nutrition_summary'] = {}

    # 3. Углубленный анализ тренировок
    logging.info("Шаг 3: Углубленный анализ тренировок...")
    
    # Шаг 3.1: Находим актуальную базу упражнений
    all_exercises_history = []
    all_unique_ex = get_all_unique_exercises(user_id)
    
    for ex in all_unique_ex:
        progression = get_exercise_progression(user_id, ex['exercise_id'])
        if not progression: continue
        
        peak_1rm = max((_calculate_one_rep_max(s['weight'], s['reps_performed']) for s in progression), default=0)
        all_exercises_history.append({
            'id': ex['exercise_id'],
            'name': ex['name'],
            'peak_1rm': peak_1rm,
            'all_sets': progression
        })

    strongest_exercises = sorted(all_exercises_history, key=itemgetter('peak_1rm'), reverse=True)[:10]
    
    recent_activity_sets = [
        s['id'] for s in strongest_exercises 
        if any(datetime.fromisoformat(p['workout_date']).date() > (datetime.now() - timedelta(days=RECENT_ACTIVITY_DAYS)).date() for p in s['all_sets'])
    ]
    
    actual_base_ids = {s['id'] for s in strongest_exercises if s['id'] in recent_activity_sets}
    actual_base_exercises = [ex for ex in strongest_exercises if ex['id'] in actual_base_ids][:TOP_N_EXERCISES]

    # Шаг 3.2: Анализируем каждое упражнение из актуальной базы
    progression_analysis = {}
    overall_strength_trend = []
    weekly_tonnage_and_efficiency = defaultdict(lambda: {'tonnage': 0, 'best_1rm': 0})

    for ex_data in actual_base_exercises:
        # Фильтруем выбросы
        all_sets = ex_data['all_sets']
        for s in all_sets: s['e1rm'] = _calculate_one_rep_max(s['weight'], s['reps_performed'])
        filtered_sets = _filter_outliers(all_sets, 'e1rm')

        # Группируем по дням и находим лучший сет за день
        best_sets_per_day = []
        for date, sets_on_day in groupby(sorted(filtered_sets, key=itemgetter('workout_date')), key=lambda x: datetime.fromisoformat(x['workout_date']).date()):
            best_set = max(sets_on_day, key=itemgetter('e1rm'))
            best_sets_per_day.append({
                'date': date.isoformat(),
                'estimated_1rm': best_set['e1rm'],
                'relative_1rm': best_set['e1rm'] / user_weight if user_weight else 0
            })
            overall_strength_trend.append({
                'date': date.isoformat(),
                'exercise_name': ex_data['name'],
                'estimated_1rm': best_set['e1rm']
            })

        # Считаем еженедельный тоннаж и лучший 1ПМ для Efficiency Score
        for s in filtered_sets:
            if s['weight'] >= ex_data['peak_1rm'] * EFFECTIVE_LOAD_THRESHOLD:
                week_start = (datetime.fromisoformat(s['workout_date']) - timedelta(days=datetime.fromisoformat(s['workout_date']).weekday())).strftime('%Y-%m-%d')
                weekly_tonnage_and_efficiency[week_start]['tonnage'] += s['weight'] * s['reps_performed']
                if s['e1rm'] > weekly_tonnage_and_efficiency[week_start]['best_1rm']:
                    weekly_tonnage_and_efficiency[week_start]['best_1rm'] = s['e1rm']

        # Анализ тренда (скользящее среднее)
        trend_verdict = "not_enough_data"
        if len(best_sets_per_day) >= 6:
            current_avg = np.mean([s['estimated_1rm'] for s in best_sets_per_day[-3:]])
            previous_avg = np.mean([s['estimated_1rm'] for s in best_sets_per_day[-6:-3]])
            if current_avg > previous_avg * 1.02: trend_verdict = "positive"
            elif current_avg < previous_avg * 0.98: trend_verdict = "negative"
            else: trend_verdict = "stagnant"

        # PR Window
        last_record_date = max((datetime.fromisoformat(s['date']) for s in best_sets_per_day), default=None) if best_sets_per_day else None
        days_since_record = (datetime.now() - last_record_date).days if last_record_date else 999
        relative_peak = ex_data['peak_1rm'] / user_weight if user_weight else 0
        _, pr_window_days = _get_exercise_tier_and_pr_window(relative_peak)
        
        pr_status = "red"
        if days_since_record <= pr_window_days: pr_status = "green"
        elif pr_window_days < days_since_record <= pr_window_days + 14: pr_status = "yellow"

        # Собираем все в один объект
        progression_analysis[ex_data['name']] = {
            'pr_status': pr_status,
            'days_since_last_record': days_since_record,
            'trend_analysis': {
                'verdict': trend_verdict,
            },
            'raw_history': sorted(best_sets_per_day, key=itemgetter('date'), reverse=True)[:10] # Последние 10 для контекста
        }

    # Формируем Efficiency Score и флаг перетрена
    efficiency_analysis = []
    for week, data in sorted(weekly_tonnage_and_efficiency.items(), reverse=True)[:4]: # Последние 4 недели
        score = data['tonnage'] / data['best_1rm'] if data['best_1rm'] > 0 else 0
        efficiency_analysis.append({'week_start': week, 'efficiency_score': round(score)})
    
    overtrain_risk = False
    if len(efficiency_analysis) > 1 and efficiency_analysis[0]['efficiency_score'] < efficiency_analysis[1]['efficiency_score']:
        # Проверяем, есть ли хоть одно упражнение в красной зоне
        if any(ex['pr_status'] == 'red' for ex in progression_analysis.values()):
            overtrain_risk = True

    final_data['workout_summary'] = {
        'progression_analysis': progression_analysis,
        'efficiency_analysis': efficiency_analysis,
        'overtrain_risk': overtrain_risk,
        'overall_strength_trend': sorted(overall_strength_trend, key=itemgetter('date'))
    }

    logging.info(f"Углубленный анализ для user_id: {user_id} завершен.")
    return final_data
