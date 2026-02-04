import pytest
from InspectorAI.finance import apply_expense, load_db, register_user
import os
from InspectorAI.finance import DB_FILE

@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    yield

def test_apply_expense_logic():
    # Регистрируем тестовых юзеров
    register_user("1", "Payer")
    register_user("2", "User B")
    register_user("3", "User C")

    # А платит 1500 за троих (доля по 500)
    # Участники: 2 и 3 (плательщик 1 учитывается внутри функции)
    share = apply_expense(payer_id="1", participant_ids=["2", "3"], total_amount=1500)

    assert share == 500.0
    db = load_db()
    assert db["2"]["debts"]["1"] == 500.0
    assert db["3"]["debts"]["1"] == 500.0


def test_mutual_offset():
    # 1. Сначала регистрируем участников в пустой базе
    register_user("1", "User A")
    register_user("2", "User B")

    # 2. СОЗДАЕМ начальный долг:
    # А платит 1000 за двоих (себя и Б). Доля Б = 500.
    apply_expense(payer_id="1", participant_ids=["2"], total_amount=1000)

    # Промежуточная проверка (по желанию): убедимся, что Б должен 500
    db_before = load_db()
    assert db_before["2"]["debts"]["1"] == 500.0

    # 3. ТЕПЕРЬ Б платит за А (делаем взаимозачет)
    # Б платит 400 за двоих. Доля А = 200.
    apply_expense(payer_id="2", participant_ids=["1"], total_amount=400)

    # 4. ФИНАЛЬНАЯ ПРОВЕРКА
    db_after = load_db()
    # Было 500, Б "отдал" 200 своим платежом, осталось 300.
    assert db_after["2"]["debts"]["1"] == 300.0