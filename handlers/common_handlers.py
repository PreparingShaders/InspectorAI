import html
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from config import (
    AUTH_QUESTION, CHECK_WORDS, NUTRITION_TRIGGERS,
    GEMINI_MODELS, NUTRITION_MODELS, TRIGGERS
)
from llm_service import (
    process_llm, update_model_mappings, current_free_or_models
)
from utils import handle_voice_transcription, get_model_short_name
from handlers.state import (
    authorized_users, user_selected_model, user_selected_nutrition_model, user_selected_provider
)
from handlers.finance_handlers import handle_group_finance, handle_finance_callback
from handlers.nutrition_handlers import handle_nutrition_photo, confirm_meal_callback, handle_nutrition_callback
from handlers.workouts_handlers import handle_workouts_callback # Новый импорт


async def show_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return

    update_model_mappings()
    keyboard = []

    def create_buttons(models, prefix, start_idx, selected_model_dict):
        buttons = []
        for i, model in enumerate(models):
            name = get_model_short_name(model, "gemini" if "gemini" in model else "openrouter")
            current_selection = selected_model_dict.get(user_id)
            is_selected = current_selection == model
            button_prefix = "✅ " if is_selected else ""
            callback_data = f"sel:{prefix}:{i + start_idx}"
            buttons.append(InlineKeyboardButton(f"{button_prefix}{name}", callback_data=callback_data))
        return [buttons[i:i + 2] for i in range(0, len(buttons), 2)]

    keyboard.append([InlineKeyboardButton("💬 Модели для общения:", callback_data="dummy")])
    keyboard.extend(create_buttons(current_free_or_models, "o", 100, user_selected_model))
    keyboard.extend(create_buttons(GEMINI_MODELS, "g", 0, user_selected_model))
    
    keyboard.append([InlineKeyboardButton("──────────────", callback_data="dummy")])
    
    keyboard.append([InlineKeyboardButton("🥗 Модели для анализа еды:", callback_data="dummy")])
    keyboard.extend(create_buttons(NUTRITION_MODELS, "n", 200, user_selected_nutrition_model))

    keyboard.append([InlineKeyboardButton("🤖 Автовыбор для чата", callback_data="sel:auto_chat")])
    keyboard.append([InlineKeyboardButton("🥗 Автовыбор для еды", callback_data="sel:auto_nutrition")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    text = "<b>Выбор модели ИИ</b>"
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.effective_message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")


async def handle_private(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    user_id = update.effective_user.id
    message = update.message
    if not message or user_id not in authorized_users:
        await message.reply_text(AUTH_QUESTION)
        return

    raw_text = voice_text or message.text or message.caption or ""
    photo = message.photo[-1] if message.photo else None
    
    if photo and any(word in (raw_text or "").lower() for word in NUTRITION_TRIGGERS):
        await handle_nutrition_photo(update, context)
        return

    final_prompt = raw_text
    mode = "chat"
    if bool(message.forward_origin) or any(word in raw_text.lower() for word in CHECK_WORDS):
        mode = "inspector"
        if message.reply_to_message:
            reply_text = message.reply_to_message.text or message.reply_to_message.caption or ""
            final_prompt = f"Контекст: {reply_text}\nВопрос: {raw_text}"
    elif message.reply_to_message:
        context_text = message.reply_to_message.text or message.reply_to_message.caption or ""
        final_prompt = f"КОНТЕКСТ ПРЕДЫДУЩЕГО СООБЩЕНИЯ:\n{context_text}\n\nТЕКУЩИЙ ЗАПРОС:\n{raw_text}"

    if final_prompt:
        await process_llm(
            update, context, final_prompt,
            selected_model=user_selected_model.get(user_id),
            mode=mode
        )


async def handle_group(update: Update, context: ContextTypes.DEFAULT_TYPE, voice_text: str = None):
    message = update.message
    if not message: return
    
    raw_text = voice_text or message.text or message.caption or ""
    
    if await handle_group_finance(update, context, raw_text):
        return

    trigger_pattern = rf"^({'|'.join(map(re.escape, TRIGGERS))})\b"
    if not re.search(trigger_pattern, raw_text.lower()):
        return

    user_query = re.sub(trigger_pattern, '', raw_text, flags=re.IGNORECASE).strip().lstrip(',. ')
    
    is_factcheck = any(word in user_query.lower() for word in CHECK_WORDS)
    mode = "inspector" if is_factcheck else "chat"
    
    if message.reply_to_message:
        reply_text = message.reply_to_message.text or message.reply_to_message.caption or ""
        final_prompt = f"Контекст: {reply_text}\nВопрос: {user_query}"
    else:
        final_prompt = user_query
        
    if final_prompt:
        await process_llm(
            update, context, final_prompt,
            selected_model=user_selected_model.get(update.effective_user.id),
            mode=mode,
            thread_id=message.message_thread_id
        )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.voice: return
    
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await message.reply_text(AUTH_QUESTION)
        return

    text = await handle_voice_transcription(message)
    if text:
        await message.reply_text(f"🎤 <b>Распознано:</b>\n<i>{html.escape(text)}</i>", parse_mode="HTML")
        if update.effective_chat.type in ["group", "supergroup"]:
            await handle_group(update, context, voice_text=text)
        else:
            await handle_private(update, context, voice_text=text)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    if user_id not in authorized_users:
        await query.answer(text=AUTH_QUESTION, show_alert=True)
        return
    
    await query.answer()

    if await handle_finance_callback(update, context):
        return

    if await confirm_meal_callback(update, context):
        return

    if await handle_nutrition_callback(update, context):
        return

    if await handle_workouts_callback(update, context): # Новый обработчик для тренировок
        return

    if data.startswith("sel:"):
        parts = data.split(":")
        action = parts[1]

        if action == "auto_chat":
            user_selected_model.pop(user_id, None)
            user_selected_provider.pop(user_id, None)
            await query.edit_message_text("🤖 Автовыбор для чата включен.")
            return
        if action == "auto_nutrition":
            user_selected_nutrition_model.pop(user_id, None)
            await query.edit_message_text("🥗 Автовыбор для еды включен.")
            return

        prov_code, idx = parts[1], int(parts[2])
        
        model_path, provider, model_dict, name_prefix = None, None, None, ""

        if prov_code == 'g':
            model_path = GEMINI_MODELS[idx]
            provider = "gemini"
            model_dict = user_selected_model
            name_prefix = "💬"
        elif prov_code == 'o':
            model_path = current_free_or_models[idx - 100]
            provider = "openrouter"
            model_dict = user_selected_model
            name_prefix = "💬"
        elif prov_code == 'n':
            model_path = NUTRITION_MODELS[idx - 200]
            provider = "gemini" if "gemini" in model_path else "openrouter"
            model_dict = user_selected_nutrition_model
            name_prefix = "🥗"

        if model_path and model_dict is not None:
            model_dict[user_id] = model_path
            if prov_code in ['g', 'o']:
                 user_selected_provider[user_id] = provider
            name = get_model_short_name(model_path, provider)
            await query.edit_message_text(f"{name_prefix} Выбрана модель: <b>{html.escape(name)}</b>", parse_mode="HTML")
    
    elif data == "open_menu":
        await show_model_selection(update, context)