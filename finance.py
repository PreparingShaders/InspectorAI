#finance.py
import json
import os
import logging

DB_FILE = "finance_db.json"


def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Ошибка загрузки БД: {e}")
        return {}


def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def register_user(user_id, name):
    """Добавляет пользователя в базу, если его там нет"""
    db = load_db()
    uid = str(user_id)
    if uid not in db:
        db[uid] = {
            "name": name,
            "debts": {}  # Кому этот юзер должен: {"id_друга": сумма}
        }
        save_db(db)
    return db


def apply_expense(payer_id, participant_ids, total_amount):
    """
    Распределяет сумму чека между участниками.
    payer_id: кто платил (int/str)
    participant_ids: список тех, за кого платили (list of str)
    total_amount: общая сумма чека (float)
    """
    db = load_db()
    payer_id = str(payer_id)

    # Считаем долю на каждого (включая плательщика)
    count = len(participant_ids) + 1
    share = round(total_amount / count, 2)

    for p_id in participant_ids:
        p_id = str(p_id)
        if p_id == payer_id:
            continue

        # ЛОГИКА ВЗАИМОЗАЧЕТА:
        # 1. Проверяем, не должен ли ПЛАТЕЛЬЩИК уже этому человеку?
        payer_debts = db[payer_id].get("debts", {})
        debt_to_friend = payer_debts.get(p_id, 0)

        if debt_to_friend > 0:
            if debt_to_friend >= share:
                # Если мой старый долг больше новой доли, просто уменьшаем мой долг
                db[payer_id]["debts"][p_id] = round(debt_to_friend - share, 2)
            else:
                # Если мой долг меньше, обнуляем его и остаток вешаем на друга
                remainder = round(share - debt_to_friend, 2)
                db[payer_id]["debts"][p_id] = 0
                db[p_id]["debts"][payer_id] = round(db[p_id]["debts"].get(payer_id, 0) + remainder, 2)
        else:
            # 2. Если плательщик ничего не был должен, просто увеличиваем долг друга перед ним
            db[p_id]["debts"][payer_id] = round(db[p_id]["debts"].get(payer_id, 0) + share, 2)

    save_db(db)
    return share


def get_detailed_report():
    """Формирует отчет: кто, кому и сколько должен"""
    db = load_db()
    lines = []

    for debtor_id, info in db.items():
        debtor_name = info["name"]
        debts = info.get("debts", {})

        for creditor_id, amount in debts.items():
            if amount > 0.01:  # Игнорируем копейки из-за округления
                creditor_name = db.get(creditor_id, {}).get("name", "Unknown")
                lines.append(f"• <b>{debtor_name}</b> ➡️ <b>{creditor_name}</b>: <code>{amount}</code> р.")

    if not lines:
        return "✨ <b>Все в расчете!</b> Долгов нет."

    return "<b>💸 Список долгов:</b>\n\n" + "\n".join(lines)


def get_all_users_except(exclude_id):
    """Возвращает список всех известных юзеров для кнопок"""
    db = load_db()
    exclude_id = str(exclude_id)
    return {uid: info["name"] for uid, info in db.items() if uid != exclude_id}


def settle_debt(debtor_id, creditor_id, amount):
    """
    Списание долга (процесс отдачи денег).
    debtor_id: кто отдает (тот, кто был должен)
    creditor_id: кому отдают
    amount: сколько денег передали (float)
    """
    db = load_db()
    debtor_id = str(debtor_id)
    creditor_id = str(creditor_id)

    if debtor_id not in db or creditor_id not in db:
        return False, "Один из пользователей не найден в базе."

    # Проверяем текущий долг
    current_debt = db[debtor_id].get("debts", {}).get(creditor_id, 0)

    if current_debt <= 0:
        return False, f"<b>{db[debtor_id]['name']}</b> ничего не должен <b>{db[creditor_id]['name']}</b>."

    if amount > current_debt:
        return False, f"Сумма (<code>{amount}</code>) больше долга (<code>{current_debt}</code>). Ланистеры не платят лишнего!"

    # Списываем долг
    new_debt = round(current_debt - amount, 2)
    db[debtor_id]["debts"][creditor_id] = new_debt

    # Если долг обнулился, можно почистить ключ (по желанию)
    if new_debt == 0:
        del db[debtor_id]["debts"][creditor_id]

    save_db(db)
    return True, f"✅ <b>{db[debtor_id]['name']}</b> вернул <b>{db[creditor_id]['name']}</b> <code>{amount}</code> р.\nОстаток долга: <code>{new_debt}</code> р."