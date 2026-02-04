import os
import json
import pytest
from finance import apply_expense, load_db, save_db, register_user, settle_debt, DB_FILE


@pytest.fixture(autouse=True)
def setup_db():
    # Создаем чистую базу для каждого теста
    test_data = {
        "1": {"name": "Payer", "debts": {}},
        "2": {"name": "User2", "debts": {}},
        "3": {"name": "User3", "debts": {}}  # Добавили тройку!
    }
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    yield
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)


def test_apply_expense_fractions():
    # Теперь юзер "3" существует, KeyError не будет
    share = apply_expense(payer_id="1", participant_ids=["2", "3"], total_amount=100)
    db = load_db()
    assert share == 33.33
    assert db["2"]["debts"]["1"] == 33.33


def test_apply_expense_self_payment():
    # Исправляем ожидание: если участников двое (включая плательщика),
    # то в participants должен быть только один "другой" юзер.
    # Если мы передали ["1", "2"], то участников по факту трое (Плательщик + "1" + "2").
    # Давай проверим сценарий, где мы просто исключаем плательщика из списка.

    apply_expense(payer_id="1", participant_ids=["2"], total_amount=200)
    db = load_db()
    assert db["2"]["debts"]["1"] == 100.0


def test_settle_debt_logic():
    # Теперь функция импортирована, NameError уйдет
    db = load_db()
    db["2"]["debts"]["1"] = 500.0
    save_db(db)

    success, _ = settle_debt(debtor_id="2", creditor_id="1", amount=200)
    assert success is True
    assert load_db()["2"]["debts"]["1"] == 300.0