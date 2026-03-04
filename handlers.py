#handlers
import re
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import time
from llm_service import process_llm
from database import get_user_status, update_daily_log  # Добавь импорт нужной функции

from database import save_profile, get_user_status, update_daily_log
from config import NUTRITION_TRIGGERS # Убедись, что добавил это в config.py

# Финансы
from finance import (
    register_user, apply_expense,
    get_detailed_report, get_all_users_except, settle_debt
)

# LLM и Сервисы
from llm_service import (
    update_model_mappings, current_free_or_models,
    GEMINI_MODEL_BY_ID, OPENROUTER_MODEL_BY_ID, process_llm
)

# Инструменты и Конфиг
from utils import handle_voice_transcription, get_model_short_name
from config import (
    CORRECT_PASSWORD, AUTH_QUESTION, TRIGGERS, CHECK_WORDS,
    GEMINI_MODELS, FINANCE_WORDS, NUTRITION_TRIGGERS
)

# Состояния пользователей
authorized_users = set()
user_selected_model = {}  # {user_id: model_path}
user_selected_provider = {}  # {user_id: "gemini" или "openrouter"}


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    message = update.message
    user_data = {}

    if not message: return

    # --- 0. ПРЕДОХРАНИТЕЛЬ ---
    is_waiting = context.user_data.get('waiting_for_food', False)
    start_time = context.user_data.get('food_mode_timestamp', 0)

    # Если режим НЕ активен ИЛИ прошло больше 5 минут — ВЫХОДИМ
    if not is_waiting or (time.time() - start_time > 300):
        if is_waiting:
            context.user_data['waiting_for_food'] = False
            await message.reply_text("⏰ Время ожидания фото вышло. Напишите 'я поел' снова.")
        return  # Важно: return должен быть здесь, чтобы игнорировать левые фото

    # --- 1. АВТОРИЗАЦИЯ И ПРОФИЛЬ (Выносим из условий IF выше) ---
    if user_id not in authorized_users:
        await message.reply_text(AUTH_QUESTION)
        return

    db_row = get_user_status(user_id)
    if not db_row:
        await message.reply_text("⚠️ Сначала заполни профиль: /profile")
        return

    # Превращаем Row в словарь, чтобы работал .get()
    user_data = dict(db_row)

    # --- 2. СКАЧИВАНИЕ ФОТО ---
    try:
        photo_file = await message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
    except Exception as e:
        logging.error(f"Ошибка скачивания: {e}")
        return

    # ВАЖНО: Сбрасываем флаг сразу, чтобы следующее фото не считалось едой без команды
    context.user_data['waiting_for_food'] = False

    # --- 3. ФОРМИРОВАНИЕ ПРОМПТА ---
    consumed_cal = user_data.get('consumed_calories', 0) or 0
    nutrition_prompt = f"""
    Ты — ИИ помошник с характером Джарвиса, строгий, но любящий нутрициолог.
    Хвали если питание сбалансированние и подшучивай если нет.
    Проанализируй фото еды. Старайся ответить коротко.
    Цель пользователя: {user_data['target_calories']} ккал.
    Уже съедено: {consumed_cal} ккал.

    1. Оцени КБЖУ блюда.
    2. Выдай краткий анализ.
    3. Коротко порекомендуй варианты еды для закрытия КБЖУ.
    4. В САМОМ КОНЦЕ добавь строку: [ADD_DB: калории, белки, жиры, углеводы] (только числа).
    """

    # --- 4. ВЫЗОВ ИИ ---
    response_text = await process_llm(
        update, context, query=nutrition_prompt,
        image_data=photo_bytes, mode="nutrition"
    )

    # --- 5. ПАРСИНГ И ОБНОВЛЕНИЕ БАЗЫ (Твой отчет вернулся!) ---
    if response_text:
        match = re.search(r"\[ADD_DB:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]", response_text)

        if match:
            new_cal, new_prot, new_fat, new_carb = map(lambda x: int(float(x)), match.groups())

            # Записываем в базу
            update_daily_log(user_id, calories=new_cal, proteins=new_prot, fats=new_fat, carbs=new_carb)

            # Расчет остатков
            total_c = (user_data['consumed_calories'] or 0) + new_cal
            total_p = (user_data['consumed_proteins'] or 0) + new_prot
            total_f = (user_data['consumed_fats'] or 0) + new_fat
            total_carb = (user_data['consumed_carbs'] or 0) + new_carb

            rem_c = (user_data['target_calories'] or 0) - total_c
            rem_p = (user_data['target_proteins'] or 0) - total_p
            rem_f = (user_data['target_fats'] or 0) - total_f
            rem_carb = (user_data['target_carbs'] or 0) - total_carb

            def fmt(val):
                return f"{val}" if val >= 0 else f"⚠️ {val} (Перебор, хорош жрать!)"

            # --- 6. ВЫВОД ОТЧЕТА ---
            await message.reply_text(
                f"✅ **Данные внесены!**\n\n"
                f"📊 **Остаток на сегодня:**\n"
                f"🔥 Калории: `{fmt(rem_c)}` ккал\n"
                f"🥩 Белки: `{fmt(rem_p)}` г\n"
                f"🥑 Жиры: `{fmt(rem_f)}` г\n"
                f"🍞 Углеводы: `{fmt(rem_carb)}` г",
                parse_mode="Markdown"
            )
        else:
            await message.reply_text(response_text)
            await message.reply_text("⚠️ ИИ не смог выдать тех-строку [ADD_DB]. Данные в базу не внесены.")
    else:
        await message.reply_text("❌ Не удалось получить ответ от ИИ.")


def calculate_targets(weight, height, age, gender, activity):
    """
    Расчет по формуле Миффлина-Сан Жеора.
    Рекомпозиция: Белок 2г/кг, Жиры 0.8г/кг, Дефицит 5%.
    """
    # Расчет BMR (Базовый обмен веществ)
    if gender.lower() in ['муж', 'м', 'male']:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    # TDEE (Общие энергозатраты с учетом активности)
    tdee = bmr * activity

    # Цель для рекомпозиции (небольшой дефицит для сжигания жира)
    target_cal = int(tdee * 0.95)

    # Макронутриенты
    proteins = int(weight * 2.0)  # Основа для мышц
    fats = int(weight * 0.8)  # Для гормонов
    # Остаток калорий отдаем под углеводы (1г углей = 4ккал)
    carbs = int((target_cal - (proteins * 4) - (fats * 9)) / 4)

    return {
        'bmr': bmr, 'tdee': tdee,
        'cal': target_cal, 'prot': proteins, 'fat': fats, 'carb': carbs
    }


async def send_participant_selector(update: Update, context: ContextTypes.DEFAULT_TYPE):

    payer_id = update.effective_user.id
    chat_id = update.effective_chat.id
    amount = context.user_data.get('tmp_amount')
    selected = context.user_data.get('tmp_participants', [])

    # 1. Получаем ВСЕХ из базы
    all_known_users = get_all_users_except(payer_id)
    keyboard = []

    # 2. Фильтруем только тех, кто в этом чате
    for uid, name in all_known_users.items():
        try:
            # Проверяем статус участника
            member = await context.bot.get_chat_member(chat_id, int(uid))
            # Если юзер не вышел и не забанен
            if member.status not in ['left', 'kicked']:
                label = f"✅ {name}" if uid in selected else name
                keyboard.append([InlineKeyboardButton(label, callback_data=f"f_toggle:{uid}")])
        except Exception as e:
            # Если ошибка (юзер не найден в чате), просто идем дальше
            logging.error(f"Ошибка проверки юзера {uid}: {e}")
            continue

    # 3. Если вдруг в списке никого нет (кроме плательщика)
    if not keyboard:
        text = f"💰 <b>Сумма: {amount} р.</b>\n\n❌ В базе нет других участников этой группы.\nПусть они напишут что-нибудь в чат, чтобы я их запомнил!"
        keyboard.append([InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")])
    else:
        text = f"💰 <b>Сумма: {amount} р.</b>\n\nВыбери тех, кто скидывается (кроме тебя):"
        keyboard.append([
            InlineKeyboardButton("🚀 РАССЧИТАТЬ", callback_data="f_confirm"),
            InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")
        ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        if update.callback_query:
            await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
        else:
            await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")
    except Exception as e:
        logging.error(f"Ошибка отправки кнопок: {e}")

# --- ОСНОВНЫЕ ХЕНДЛЕРЫ ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in authorized_users:
        model = user_selected_model.get(user_id)
        text = "Ты уже авторизован!\n\n"
        if model:
            prov = user_selected_provider.get(user_id, "").upper()
            name = get_model_short_name(model, prov.lower())
            text += f"Текущая модель: {prov} → {name}\n\n"
        else:
            text += "Режим: 🤖 Автоматический выбор\n\n"
        text += "Сменить модель → /model"
        await update.message.reply_text(text)
    else:
        await update.message.reply_text(AUTH_QUESTION)


async def show_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    update_model_mappings()
    user_id = update.effective_user.id
    keyboard = []

    # OpenRouter
    keyboard.append([InlineKeyboardButton("🎁 OpenRouter (Free):", callback_data="dummy")])
    or_buttons = []
    for i, model in enumerate(current_free_or_models):
        name = get_model_short_name(model, "openrouter")
        prefix = "✅ " if user_selected_model.get(user_id) == model else ""
        or_buttons.append(InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:o:{i + 100}"))
        if len(or_buttons) == 2:
            keyboard.append(or_buttons)
            or_buttons = []
    if or_buttons: keyboard.append(or_buttons)

    # Gemini
    keyboard.append([InlineKeyboardButton("──────────────", callback_data="dummy")])
    keyboard.append([InlineKeyboardButton("✨ Gemini (Резерв):", callback_data="dummy")])
    gem_buttons = []
    for i, model in enumerate(GEMINI_MODELS):
        name = get_model_short_name(model, "gemini")
        prefix = "✅ " if user_selected_model.get(user_id) == model else ""
        gem_buttons.append(InlineKeyboardButton(f"{prefix}{name}", callback_data=f"sel:g:{i}"))
        if len(gem_buttons) == 2:
            keyboard.append(gem_buttons)
            gem_buttons = []
    if gem_buttons: keyboard.append(gem_buttons)

    keyboard.append([InlineKeyboardButton("🤖 Автовыбор", callback_data="sel:auto")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    text = "<b>Выбор модели ИИ</b>"
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.effective_message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.voice: return
    text = await handle_voice_transcription(message)
    if text:
        await message.reply_text(f"🎤 <b>Распознано:</b>\n<i>{text}</i>", parse_mode="HTML")
        if update.effective_chat.type in ["group", "supergroup"]:
            await handle_group(update, context, voice_text=text)
        else:
            await handle_private(update, context, voice_text=text)


async def handle_private(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    user_id = update.effective_user.id
    message = update.message
    if not message: return

    raw_text = voice_text or message.text or message.caption or ""
    if not raw_text and not message.photo: return

    # --- 1. ПРОВЕРКА СОСТОЯНИЯ (АНКЕТА) ---
    if context.user_data.get('nutrition_state') == 'WAITING_PROFILE_DATA':
        try:
            # Парсим строку: 30, муж, 180, 90, 80, 1.2
            parts = [p.strip() for p in raw_text.split(',')]
            if len(parts) < 6:
                await message.reply_text("⚠️ Нужно 6 параметров через запятую! Попробуй еще раз.")
                return

            age = int(parts[0])
            gender = parts[1].lower()
            height = float(parts[2])
            weight = float(parts[3])
            target_w = float(parts[4])
            activity = float(parts[5])

            # Считаем нормы
            res = calculate_targets(weight, height, age, gender, activity)

            # Сохраняем в SQLite (импортируй save_profile из database)
            from database import save_profile
            profile_data = {
                'age': age, 'gender': gender, 'height': height,
                'start_weight': weight, 'target_weight': target_w,
                'activity_level': activity, 'bmr': res['bmr'], 'tdee': res['tdee'],
                'target_calories': res['cal'], 'target_proteins': res['prot'],
                'target_fats': res['fat'], 'target_carbs': res['carb']
            }
            save_profile(user_id, profile_data)

            # Сбрасываем состояние
            context.user_data.pop('nutrition_state', None)

            await message.reply_text(
                f"✅ <b>Профиль настроен!</b>\n\n"
                f"Ваш BMR: <b>{int(res['bmr'])} ккал</b>\n"
                f"Цель (Ккал): <b>{res['cal']}</b>\n"
                f"Белки: <b>{res['prot']}г</b> | Жиры: <b>{res['fat']}г</b> | Угли: <b>{res['carb']}г</b>\n\n"
                f"Теперь я буду учитывать это при анализе ваших фото еды.",
                parse_mode="HTML"
            )
            return
        except Exception as e:
            logging.error(f"Ошибка парсинга анкеты: {e}")
            await message.reply_text("❌ Ошибка в данных. Проверь, чтобы были только числа (кроме пола).")
            return

    # --- ЗАЩИТА ОТ ДУБЛЕЙ (АЛЬБОМОВ) ---
    current_time = time.time()
    last_text = context.user_data.get('last_msg_text', "")
    last_msg_id = context.user_data.get('last_msg_id', 0)
    last_time = context.user_data.get('last_msg_time', 0)

    if raw_text == last_text and message.message_id != last_msg_id and (current_time - last_time) < 1.5:
        return

    context.user_data['last_msg_text'] = raw_text
    context.user_data['last_msg_id'] = message.message_id
    context.user_data['last_msg_time'] = current_time
    # ----------------------------------

    # Блок авторизации (теперь user_id на месте)
    if user_id not in authorized_users:
        if raw_text.strip().lower() == CORRECT_PASSWORD.lower():
            authorized_users.add(user_id)
            await message.reply_text("✅ Доступ разрешен!")
            return
        await message.reply_text(AUTH_QUESTION)
        return

    # --- 1. ТРИГГЕРЫ АКТИВАЦИИ РЕЖИМА ЕДЫ ---
    # Список фраз-команд
    start_food_triggers = NUTRITION_TRIGGERS

    if any(trigger in raw_text.lower() for trigger in start_food_triggers):
        context.user_data['waiting_for_food'] = True
        context.user_data['food_mode_timestamp'] = time.time()
        await message.reply_text("🥗 Режим нутрициолога включен! Теперь жду фото (у вас есть 5 минут).")
        return

    # --- 2. ЖЕСТКИЙ ФИЛЬТР ФОТО ---
    photo = message.photo[-1] if message.photo else None
    image_data = None
    mode = "chat"  # По умолчанию режим чата

async def handle_group(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    message = update.message
    if not message: return

    # Регистрация участника
    register_user(update.effective_user.id, update.effective_user.first_name)

    raw_text = voice_text or message.text or message.caption or ""
    text_lower = raw_text.lower()

    # Проверка состояний финансов
    state = context.user_data.get('finance_state')

    if state in ['WAITING_AMOUNT', 'WAITING_PAYBACK_AMOUNT']:
        amount_match = re.search(r"(\d+(?:[.,]\d+)?)", raw_text)
        if amount_match:
            amount = float(amount_match.group(1).replace(',', '.'))

            if state == 'WAITING_AMOUNT':
                context.user_data.update(
                    {'tmp_amount': amount, 'tmp_participants': [], 'finance_state': 'SELECT_PARTICIPANTS'})
                await send_participant_selector(update, context)

            else:  # WAITING_PAYBACK_AMOUNT
                from finance import settle_debt
                creditor_id = context.user_data.get('tmp_creditor_id')
                success, text = settle_debt(update.effective_user.id, creditor_id, amount)
                if success:
                    context.user_data.clear()
                    await message.reply_text(text, parse_mode="HTML")
                else:
                    await message.reply_text(f"❌ {text}\nПопробуй еще раз или напиши 'отмена'.", parse_mode="HTML")
            return

    # Проверка триггера (имя бота)
    trigger_pattern = rf"^({'|'.join(map(re.escape, TRIGGERS))})\b"
    if not re.search(trigger_pattern, text_lower): return

    user_query = re.sub(trigger_pattern, '', raw_text, flags=re.IGNORECASE).strip().lstrip(',. ')
    query_lower = user_query.lower()

    # Финансовый блок
    # 1. Сначала проверяем жесткие команды баланса (независимо от FINANCE_WORDS)
    if any(w in query_lower for w in ["баланс", "задолжность", "кто кому"]):
        await message.reply_text(get_detailed_report(), parse_mode="HTML")
        return

    # 2. Потом проверяем FINANCE_WORDS для начала записи чека
    if any(word in query_lower for word in FINANCE_WORDS):
        context.user_data['finance_state'] = 'WAITING_AMOUNT'
        await message.reply_text("💵 <b>Введите сумму расхода:</b>", parse_mode="HTML")
        return

    # 3. Час расплаты (списание долга)
    if any(w in query_lower for w in ["час расплаты", "вернуть долг", "отдать долг", 'ланистеры платят долги']):
        from finance import load_db
        db = load_db()
        my_debts = db.get(str(update.effective_user.id), {}).get("debts", {})

        # Фильтруем только реальные долги (> 0)
        active_debts = {k: v for k, v in my_debts.items() if v > 0}

        if not active_debts:
            await message.reply_text("✨ Ты никому ничего не должен. Спи спокойно, Ланистер!")
            return

        if len(active_debts) == 1:
            # Если должен только одному человеку
            creditor_id = list(active_debts.keys())[0]
            context.user_data.update({
                'finance_state': 'WAITING_PAYBACK_AMOUNT',
                'tmp_creditor_id': creditor_id
            })
            creditor_name = db.get(creditor_id, {}).get("name", "Друг")
            await message.reply_text(f"💰 Сколько возвращаем для <b>{creditor_name}</b>?", parse_mode="HTML")
        else:
            # Если должен нескольким — строим клавиатуру
            keyboard = []
            for c_id, amt in active_debts.items():
                c_name = db.get(c_id, {}).get("name", "Unknown")
                keyboard.append(
                    [InlineKeyboardButton(f"{c_name} (долг: {amt} р.)", callback_data=f"pay_select:{c_id}")])

            keyboard.append([InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")])
            await message.reply_text("Кому именно ты возвращаешь долг?", reply_markup=InlineKeyboardMarkup(keyboard))
        return


    # Режим LLM
    is_factcheck = any(word in query_lower for word in CHECK_WORDS)
    mode = "inspector" if is_factcheck else "chat"

    # Исправленная логика получения текста из реплая
    if message.reply_to_message:
        # Проверяем и текст, и подпись под фото/видео
        reply_text = message.reply_to_message.text or message.reply_to_message.caption or ""
        final_prompt = f"Контекст: {reply_text}\nВопрос: {user_query}"
    else:
        final_prompt = user_query

    await process_llm(
        update, context, final_prompt,
        user_selected_model.get(update.effective_user.id),
        user_selected_provider.get(update.effective_user.id),
        thread_id=message.message_thread_id,
        mode=mode
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    # --- ФИНАНСЫ В CALLBACK ---
    if data.startswith("f_toggle:"):
        uid = data.split(":")[1]
        participants = context.user_data.get('tmp_participants', [])
        if uid in participants:
            participants.remove(uid)
        else:
            participants.append(uid)
        context.user_data['tmp_participants'] = participants
        await send_participant_selector(update, context)
        await query.answer()
        return

    if data.startswith("pay_select:"):
        creditor_id = data.split(":")[1]
        from finance import load_db
        db = load_db()
        creditor_name = db.get(creditor_id, {}).get("name", "Друг")

        context.user_data.update({
            'finance_state': 'WAITING_PAYBACK_AMOUNT',
            'tmp_creditor_id': creditor_id
        })
        await query.edit_message_text(f"💰 Сколько возвращаем для <b>{creditor_name}</b>?", parse_mode="HTML")
        return

    if data == "f_confirm":
        participants = context.user_data.get('tmp_participants')
        if not participants:
            await query.answer("Выбери хотя бы одного!", show_alert=True)
            return
        share = apply_expense(user_id, participants, context.user_data.get('tmp_amount'))
        await query.edit_message_text(f"✅ Записано!\nКаждый (включая тебя) должен по {share} р.")
        context.user_data.clear()
        return

    if data == "f_cancel":
        context.user_data.clear()
        await query.edit_message_text("❌ Расчет отменен.")
        return

    # --- МОДЕЛИ В CALLBACK ---
    await query.answer()
    if data == "open_menu":
        await show_model_selection(update, context)
    elif data == "sel:auto":
        user_selected_model[user_id] = user_selected_provider[user_id] = None
        await query.edit_message_text("🤖 Автовыбор включен.")
    elif data.startswith("sel:"):
        _, prov_code, idx = data.split(":")
        provider = "gemini" if prov_code == "g" else "openrouter"
        model_path = GEMINI_MODEL_BY_ID.get(idx) if prov_code == "g" else OPENROUTER_MODEL_BY_ID.get(idx)

        if model_path:
            user_selected_model[user_id], user_selected_provider[user_id] = model_path, provider
            name = get_model_short_name(model_path, provider)
            await query.edit_message_text(f"🎯 Выбрана модель: <b>{name}</b>", parse_mode="HTML")

async def profile_setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Триггер для начала заполнения анкеты"""
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.message.reply_text(AUTH_QUESTION)
        return

    context.user_data['nutrition_state'] = 'WAITING_PROFILE_DATA'
    await update.message.reply_text(
        "📝 <b>Введите данные через запятую:</b>\n"
        "<code>Возраст, Пол (муж/жен), Рост, Вес, Цель (вес), Активность</code>\n\n"
        "<b>Пример:</b> <code>28, муж, 182, 95, 85, 1.2</code>",
        parse_mode="HTML"
    )