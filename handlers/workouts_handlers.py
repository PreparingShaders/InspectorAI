import html
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from handlers.state import authorized_users
from config import AUTH_QUESTION
from handlers.base import get_main_keyboard


def get_workouts_inline_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("💪 Мои тренировки", callback_data="workouts_my"),
         InlineKeyboardButton("➕ Добавить тренировку", callback_data="workouts_add")],
        [InlineKeyboardButton("📈 Статистика тренировок", callback_data="workouts_stats")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)


async def show_workouts_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return
    
    text = "<b>🏋️ Меню Тренировок:</b>"
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=get_workouts_inline_keyboard(), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(text, reply_markup=get_workouts_inline_keyboard(), parse_mode="HTML")


async def handle_workouts_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Handles callbacks related to workouts menu."""
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    if not data.startswith("workouts_") and data != "main_menu":
        return False

    await query.answer()

    if data == "workouts_my":
        await query.edit_message_text("Здесь будут ваши тренировки.") # Placeholder
    elif data == "workouts_add":
        await query.edit_message_text("Здесь будет добавление тренировки.") # Placeholder
    elif data == "workouts_stats":
        await query.edit_message_text("Здесь будет статистика тренировок.") # Placeholder
    elif data == "main_menu":
        await query.edit_message_text("Клавиатура на месте! Чем займемся?", reply_markup=get_main_keyboard())
    
    return True