#test_finance.py
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


def test_apply_expense_three_people():
    # Платит 1-й, участвуют 2-й и 3-й. Итого 3 человека.
    # 300 / 3 = 100р на каждого.
    apply_expense(payer_id="1", participant_ids=["2", "3"], total_amount=300)
    db = load_db()
    assert db["2"]["debts"]["1"] == 100.0
    assert db["3"]["debts"]["1"] == 100.0


def test_settle_debt_logic():
    # Теперь функция импортирована, NameError уйдет
    db = load_db()
    db["2"]["debts"]["1"] = 500.0
    save_db(db)

    success, _ = settle_debt(debtor_id="2", creditor_id="1", amount=200)
    assert success is True
    assert load_db()["2"]["debts"]["1"] == 300.0


def test_mutual_offset_logic():
    # 1. Сначала Юзер 2 должен Юзеру 1 (50р)
    db = load_db()
    db["2"]["debts"]["1"] = 50.0
    save_db(db)

    # 2. Теперь Юзер 1 платит за Юзера 2 (через общий чек в 100р на двоих)
    # Доля Юзера 2 составит 50р.
    # По логике взаимозачета: старый долг 50 - новая доля 50 = 0.
    apply_expense(payer_id="2", participant_ids=["1"], total_amount=100)

    db = load_db()
    assert db["2"]["debts"].get("1", 0) == 0
    assert db["1"]["debts"].get("2", 0) == 0


def test_web_of_debts_final():
    # 1. А (1) платит 1500 за А, Б, В. Доля каждого 500.
    # Участники: Б(2), В(3). Всего 3 чел.
    apply_expense(payer_id="1", participant_ids=["2", "3"], total_amount=1500)

    # 2. Б (2) платит 600 за А, Б, В. Доля каждого 200.
    # Участники: А(1), В(3). Всего 3 чел.
    apply_expense(payer_id="2", participant_ids=["1", "3"], total_amount=600)

    # 3. В (3) платит 2100 за А, Б, В. Доля каждого 700.
    # Участники: А(1), Б(2). Всего 3 чел.
    apply_expense(payer_id="3", participant_ids=["1", "2"], total_amount=2100)

    # 4. Б (2) платит 350 за Б, А. Доля каждого 175.
    # Участники: А(1). Всего 2 чел.
    apply_expense(payer_id="2", participant_ids=["1"], total_amount=350)

    # 5. В (3) платит 1250 за В, Б. Доля каждого 625.
    # Участники: Б(2). Всего 2 чел.
    apply_expense(payer_id="3", participant_ids=["2"], total_amount=1250)

    db = load_db()

    # 1. Проверяем А должен В (Юзер 1 должен Юзеру 3)
    # После всех шагов А должен В 200р
    assert db["1"]["debts"].get("3") == 200.0

    # 2. Проверяем Б должен А (Юзер 2 должен Юзеру 1)
    # Б был должен 500, потом отдал 200 (шаг 2), потом отдал 175 (шаг 4).
    # 500 - 200 - 175 = 125.
    assert db["2"]["debts"].get("1") == 125.0

    # 3. Проверяем Б должен В (Юзер 2 должен Юзеру 3)
    # Шаг 3 (700 за Б) + Шаг 5 (625 за Б) = 1325.
    # Но в шаге 2 Б заплатил за В (200), значит 1325 - 200 = 1125.
    assert db["2"]["debts"].get("3") == 1125.0

    # Проверка на отсутствие "левых" ключей (чтобы не упало с KeyError)
    assert "2" not in db["1"].get("debts", {})