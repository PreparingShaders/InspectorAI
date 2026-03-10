import html
import json
import logging
import re
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes, ConversationHandler

from InspectorAI.handlers.state import authorized_users
from InspectorAI.config import AUTH_QUESTION


def get_main_keyboard() -> ReplyKeyboardMarkup:
    keyboard = [
        [KeyboardButton("🥗 Нутрициолог"), KeyboardButton("🏋️ Тренировки")],
        [KeyboardButton("🤖 Сменить модель")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in authorized_users:
        await update.effective_message.reply_text(
            "Клавиатура на месте! Чем займемся?",
            reply_markup=get_main_keyboard()
        )
    else:
        await update.effective_message.reply_text(AUTH_QUESTION)


async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_message.reply_text(
        'Действие отменено.', reply_markup=get_main_keyboard()
    )
    context.user_data.clear()
    return ConversationHandler.END


def parse_llm_json(response_text: str) -> dict | None:
    if not response_text:
        return None
    match = re.search(r'(.*?)```json\n(.*?)\n```', response_text, re.DOTALL)
    if match:
        comment_text, json_text = match.group(1).strip(), match.group(2)
        try:
            from InspectorAI.utils import to_html
            data = json.loads(json_text)
            if comment_text:
                data['comment'] = to_html(comment_text)
            return data
        except json.JSONDecodeError:
            logging.warning("Found JSON block but failed to parse.")
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse as JSON: {response_text}")
        return None


def format_meal_data_for_display(meal_data: dict, model_name: str | None) -> str:
    if not meal_data:
        return "Не удалось распознать данные о еде."
    
    from InspectorAI.utils import get_model_short_name
    comment = meal_data.get('comment', '')
    text = ""
    if model_name:
        short_model_name = get_model_short_name(model_name, "gemini" if "gemini" in model_name else "openrouter")
        text += f"🤖 <b>Модель:</b> <code>{html.escape(short_model_name)}</code>\n\n"
    if comment:
        text += f"{comment}\n\n"
    text += (
        f"<b>📊 Оценка КБЖУ:</b>\n"
        f"🔥 Калории: <code>{meal_data.get('calories', 0)}</code>\n"
        f"🥩 Белки: <code>{meal_data.get('proteins', 0)} г</code>\n"
        f"🥑 Жиры: <code>{meal_data.get('fats', 0)} г</code>\n"
        f"🍞 Углеводы: <code>{meal_data.get('carbs', 0)} г</code>"
    )
    return text