import html
import logging
import re

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, error
from telegram.ext import ContextTypes

from InspectorAI.finance import (
    register_user,
    apply_expense,
    get_detailed_report,
    get_all_users_except,
    settle_debt,
    load_db,
)
from InspectorAI.config import TRIGGERS, FINANCE_WORDS
from InspectorAI.handlers.state import user_selected_model
from InspectorAI.llm_service import process_llm


async def send_participant_selector(update: Update, context: ContextTypes.DEFAULT_TYPE):
    payer_id = update.effective_user.id
    chat_id = update.effective_chat.id
    amount = context.user_data.get('tmp_amount', 0)
    
    selected = context.user_data.get('tmp_participants', [])
    all_known_users = get_all_users_except(payer_id)
    keyboard = []
    
    for uid, name in all_known_users.items():
        try:
            member = await context.bot.get_chat_member(chat_id, int(uid))
            if member.status not in ['left', 'kicked']:
                label = f"✅ {name}" if uid in selected else name
                keyboard.append([InlineKeyboardButton(label, callback_data=f"f_toggle:{uid}")])
        except error.BadRequest as e:
            if "participant_id_invalid" in str(e).lower():
                continue
            logging.error(f"Ошибка проверки юзера {uid} в чате {chat_id}: {e}")
        except Exception as e:
            logging.error(f"Неожиданная ошибка проверки юзера {uid}: {e}")
            continue
            
    if not keyboard:
        text = f"💰 <b>Сумма: {amount} р.</b>\n\n❌ В базе нет других участников этой группы."
        keyboard.append([InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")])
    else:
        text = f"💰 <b>Сумма: {amount} р.</b>\n\nВыбери тех, кто скидывается:"
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
        logging.error(f"Ошибка отправки кнопок выбора участников: {e}")


async def handle_group_finance(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
    """
    Handles finance-related messages in a group chat.
    Returns True if the message was handled, False otherwise.
    """
    message = update.message
    user_id = update.effective_user.id
    register_user(user_id, update.effective_user.first_name)
    
    text_lower = text.lower()
    state = context.user_data.get('finance_state')

    # State-based logic for amount input
    if state in ['WAITING_AMOUNT', 'WAITING_PAYBACK_AMOUNT']:
        amount_match = re.search(r"(\d+(?:[.,]\d+)?)", text)
        if amount_match:
            amount = float(amount_match.group(1).replace(',', '.'))
            if state == 'WAITING_AMOUNT':
                context.user_data.update(
                    {'tmp_amount': amount, 'tmp_participants': [], 'finance_state': 'SELECT_PARTICIPANTS'})
                await send_participant_selector(update, context)
            else: # WAITING_PAYBACK_AMOUNT
                creditor_id = context.user_data.get('tmp_creditor_id')
                success, response_text = settle_debt(user_id, creditor_id, amount)
                if success:
                    context.user_data.clear()
                    await message.reply_text(response_text, parse_mode="HTML")
                else:
                    await message.reply_text(f"❌ {response_text}\nПопробуй еще раз или напиши 'отмена'.", parse_mode="HTML")
            return True
        # If no amount found, let it fall through to other handlers

    # Trigger-based logic
    trigger_pattern = rf"^({'|'.join(map(re.escape, TRIGGERS))})\b"
    if not re.search(trigger_pattern, text_lower):
        return False

    user_query = re.sub(trigger_pattern, '', text, flags=re.IGNORECASE).strip().lstrip(',. ')
    query_lower = user_query.lower()

    if any(w in query_lower for w in ["баланс", "задолжность", "кто кому"]):
        await message.reply_text(get_detailed_report(), parse_mode="HTML")
        return True

    if any(word in query_lower for word in FINANCE_WORDS):
        context.user_data['finance_state'] = 'WAITING_AMOUNT'
        await message.reply_text("💵 <b>Введите сумму расхода:</b>", parse_mode="HTML")
        return True

    if any(w in query_lower for w in ["час расплаты", "вернуть долг", "отдать долг", 'ланистеры платят долги']):
        db = load_db()
        my_debts = db.get(str(user_id), {}).get("debts", {})
        active_debts = {k: v for k, v in my_debts.items() if v > 0}
        
        if not active_debts:
            await message.reply_text("✨ Ты никому ничего не должен.")
            return True
            
        if len(active_debts) == 1:
            creditor_id = list(active_debts.keys())[0]
            context.user_data.update({
                'finance_state': 'WAITING_PAYBACK_AMOUNT',
                'tmp_creditor_id': creditor_id
            })
            creditor_name = db.get(creditor_id, {}).get("name", "Друг")
            await message.reply_text(f"💰 Сколько возвращаем для <b>{html.escape(creditor_name)}</b>?", parse_mode="HTML")
        else:
            keyboard = []
            for c_id, amt in active_debts.items():
                c_name = db.get(c_id, {}).get("name", "Unknown")
                keyboard.append(
                    [InlineKeyboardButton(f"{c_name} (долг: {amt} р.)", callback_data=f"pay_select:{c_id}")])
            keyboard.append([InlineKeyboardButton("❌ ОТМЕНА", callback_data="f_cancel")])
            await message.reply_text("Кому именно ты возвращаешь долг?", reply_markup=InlineKeyboardMarkup(keyboard))
        return True

    # If no finance keyword matched, let it fall through to the general LLM handler
    return False


async def handle_finance_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Handles finance-related callback queries.
    Returns True if the callback was handled, False otherwise.
    """
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    if data.startswith("f_toggle:"):
        uid = data.split(":")[1]
        participants = context.user_data.get('tmp_participants', [])
        if uid in participants:
            participants.remove(uid)
        else:
            participants.append(uid)
        context.user_data['tmp_participants'] = participants
        await send_participant_selector(update, context)
        return True

    elif data.startswith("pay_select:"):
        creditor_id = data.split(":")[1]
        db = load_db()
        creditor_name = db.get(creditor_id, {}).get("name", "Друг")
        context.user_data.update({
            'finance_state': 'WAITING_PAYBACK_AMOUNT',
            'tmp_creditor_id': creditor_id
        })
        await query.edit_message_text(f"💰 Сколько возвращаем для <b>{html.escape(creditor_name)}</b>?", parse_mode="HTML")
        return True

    elif data == "f_confirm":
        participants = context.user_data.get('tmp_participants')
        if not participants:
            await query.answer("Выбери хотя бы одного!", show_alert=True)
            return True
        amount = context.user_data.get('tmp_amount')
        share = apply_expense(user_id, participants, amount)
        await query.edit_message_text(f"✅ Записано!\nКаждый (включая тебя) должен по {share:.2f} р.")
        context.user_data.clear()
        return True

    elif data == "f_cancel":
        context.user_data.clear()
        await query.edit_message_text("❌ Расчет отменен.")
        return True

    return False