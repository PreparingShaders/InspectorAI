import html
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler, MessageHandler, CallbackQueryHandler, filters, CommandHandler
from telegram.error import BadRequest # Импортируем BadRequest

from handlers.state import authorized_users
from config import AUTH_QUESTION
from handlers.base import get_main_keyboard, cancel_conversation
from workouts import (
    add_workout_template, add_exercise_to_workout, get_workout_templates,
    delete_workout_template, get_exercises_for_workout, get_workout_template_by_id,
    update_workout_template_name, update_exercise, delete_exercise,
    get_last_set_data_for_exercise, start_logged_workout, add_logged_set, end_logged_workout,
    get_exercise_by_id
)

# Conversation states for adding a workout
(
    ADD_WORKOUT_NAME,
    ADD_EXERCISE_NAME,
    ADD_EXERCISE_SETS,
    ADD_EXERCISE_REPS,
    ADD_EXERCISE_COMMENT,
    CONFIRM_ADD_EXERCISE,
    # Новые состояния для редактирования/просмотра
    VIEW_WORKOUT_DETAILS,
    EDIT_WORKOUT_NAME_STATE,
    EDIT_EXERCISE_SELECT,
    EDIT_EXERCISE_NAME_STATE,
    EDIT_EXERCISE_SETS_STATE,
    EDIT_EXERCISE_REPS_STATE,
    EDIT_EXERCISE_COMMENT_STATE,
    CONFIRM_DELETE_WORKOUT,
    CONFIRM_DELETE_EXERCISE,
    # Новые состояния для запуска тренировки
    START_WORKOUT_SESSION,
    LOG_SET_WEIGHT,
    LOG_SET_REPS,
    CONFIRM_SET_LOG,
    NEXT_SET_OR_EXERCISE
) = range(20) # Увеличиваем диапазон состояний

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


# --- Handlers for adding a workout ---
async def start_add_workout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return ConversationHandler.END

    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.delete() # Удаляем сообщение с кнопками меню тренировок
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Начнем создание новой тренировки! Как ее назовем? (Например: 'День ног', 'Фулбоди А')",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Отмена", callback_data="cancel_add_workout")]])
    )
    return ADD_WORKOUT_NAME

async def get_workout_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    workout_name = update.message.text.strip()
    if not workout_name:
        await update.message.reply_text("Название тренировки не может быть пустым. Пожалуйста, введите название.")
        return ADD_WORKOUT_NAME
    
    user_id = update.effective_user.id
    workout_id = add_workout_template(user_id, workout_name)
    
    context.user_data['current_workout_id'] = workout_id
    context.user_data['current_workout_name'] = workout_name
    context.user_data['exercises_count'] = 0 # Для отслеживания количества добавленных упражнений

    await update.message.reply_text(
        f"Отлично! Тренировка '<b>{html.escape(workout_name)}</b>' создана.\n"
        "Теперь давай добавим первое упражнение. Как оно называется?",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Отмена", callback_data="cancel_add_workout")]])
    )
    return ADD_EXERCISE_NAME

async def get_exercise_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    exercise_name = update.message.text.strip()
    if not exercise_name:
        await update.message.reply_text("Название упражнения не может быть пустым. Пожалуйста, введите название.")
        return ADD_EXERCISE_NAME
    
    context.user_data['current_exercise_name'] = exercise_name
    await update.message.reply_text(
        f"Сколько подходов планируется для упражнения '<b>{html.escape(exercise_name)}</b>'? (Например: 3)",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Отмена", callback_data="cancel_add_workout")]])
    )
    return ADD_EXERCISE_SETS

async def get_exercise_sets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        sets = int(update.message.text.strip())
        if sets <= 0:
            raise ValueError
        context.user_data['current_exercise_sets'] = sets
        await update.message.reply_text(
            f"Сколько повторений в каждом подходе для '<b>{html.escape(context.user_data['current_exercise_name'])}</b>'? (Например: 8-12)",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Отмена", callback_data="cancel_add_workout")]])
        )
        return ADD_EXERCISE_REPS
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите корректное число подходов (целое положительное число).")
        return ADD_EXERCISE_SETS

async def get_exercise_reps(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reps = update.message.text.strip()
    if not reps:
        await update.message.reply_text("Количество повторений не может быть пустым. Пожалуйста, введите количество повторений.")
        return ADD_EXERCISE_REPS
    
    context.user_data['current_exercise_reps'] = reps
    await update.message.reply_text(
        f"Есть ли комментарий к упражнению '<b>{html.escape(context.user_data['current_exercise_name'])}</b>'? (Например: 'Ставь ноги чуть вперед, садись глубоко.')\n"
        "Или отправьте '-' если комментария нет.",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Отмена", callback_data="cancel_add_workout")]])
    )
    return ADD_EXERCISE_COMMENT

async def get_exercise_comment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    comment = update.message.text.strip()
    context.user_data['current_exercise_comment'] = comment if comment != '-' else None
    
    workout_id = context.user_data['current_workout_id']
    exercise_name = context.user_data['current_exercise_name']
    planned_sets = context.user_data['current_exercise_sets']
    planned_reps = context.user_data['current_exercise_reps']
    exercise_comment = context.user_data['current_exercise_comment']

    add_exercise_to_workout(workout_id, exercise_name, planned_sets, planned_reps, exercise_comment)
    context.user_data['exercises_count'] += 1

    keyboard = [
        [InlineKeyboardButton("➕ Добавить еще упражнение", callback_data="add_another_exercise")],
        [InlineKeyboardButton("✅ Завершить создание тренировки", callback_data="finish_workout_creation")],
        [InlineKeyboardButton("❌ Отмена", callback_data="cancel_add_workout")]
    ]
    await update.message.reply_text(
        f"Упражнение '<b>{html.escape(exercise_name)}</b>' добавлено!\n"
        "Что дальше?",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return CONFIRM_ADD_EXERCISE

async def confirm_add_exercise(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "add_another_exercise":
        await query.edit_message_text(
            "Отлично, давай добавим следующее упражнение. Как оно называется?",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Отмена", callback_data="cancel_add_workout")]])
        )
        return ADD_EXERCISE_NAME
    elif data == "finish_workout_creation":
        workout_name = context.user_data.get('current_workout_name', 'тренировка')
        exercises_count = context.user_data.get('exercises_count', 0)
        await query.edit_message_text(
            f"🎉 Создание тренировки '<b>{html.escape(workout_name)}</b>' завершено! Добавлено {exercises_count} упражнений.",
            parse_mode="HTML",
            reply_markup=get_workouts_inline_keyboard() # Возвращаем к основному меню тренировок
        )
        context.user_data.clear() # Очищаем данные пользователя после завершения
        return ConversationHandler.END
    elif data == "cancel_add_workout":
        # Если отмена на этом этапе, удаляем созданный шаблон тренировки
        workout_id = context.user_data.get('current_workout_id')
        if workout_id:
            delete_workout_template(workout_id, update.effective_user.id)
            logging.info(f"Отменено создание тренировки, удален шаблон workout_id={workout_id}")
        await cancel_conversation(update, context) # Используем общий обработчик отмены
        return ConversationHandler.END

# --- Handlers for "My Workouts" ---
async def show_my_workouts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in authorized_users:
        await update.effective_message.reply_text(AUTH_QUESTION)
        return

    workouts = get_workout_templates(user_id)
    keyboard = []
    text = "<b>💪 Ваши тренировки:</b>\n\n"

    if not workouts:
        text += "У вас пока нет сохраненных тренировок. Нажмите '➕ Добавить тренировку', чтобы создать первую!"
    else:
        for workout in workouts:
            keyboard.append([
                InlineKeyboardButton(workout['name'], callback_data=f"view_workout:{workout['workout_id']}"),
                InlineKeyboardButton("▶️ Запустить", callback_data=f"start_workout:{workout['workout_id']}"),
                InlineKeyboardButton("🗑️ Удалить", callback_data=f"confirm_delete_workout:{workout['workout_id']}")
            ])
        text += "Выберите тренировку для просмотра/запуска или удаления:"
    
    keyboard.append([InlineKeyboardButton("⬅️ Назад в меню тренировок", callback_data="workouts_menu")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.effective_message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")


async def view_workout_details(update: Update, context: ContextTypes.DEFAULT_TYPE, workout_id: int):
    user_id = update.effective_user.id
    workout = get_workout_template_by_id(workout_id, user_id)
    if not workout:
        error_text = "Тренировка не найдена или у вас нет к ней доступа."
        if update.callback_query:
            await update.callback_query.edit_message_text(error_text, reply_markup=get_workouts_inline_keyboard())
        else:
            await update.message.reply_text(error_text, reply_markup=get_workouts_inline_keyboard())
        return

    exercises = get_exercises_for_workout(workout_id)
    
    text = f"<b>🏋️‍♀️ {html.escape(workout['name'])}</b>\n\n"
    if not exercises:
        text += "В этой тренировке пока нет упражнений."
    else:
        for i, exercise in enumerate(exercises):
            text += f"<b>{i+1}. {html.escape(exercise['name'])}</b>\n"
            text += f"   Подходы: {exercise['planned_sets']} x Повторения: {exercise['planned_reps']}\n"
            if exercise['comment']:
                text += f"   <i>Комментарий: {html.escape(exercise['comment'])}</i>\n"
            text += "\n"
    
    keyboard = [
        [InlineKeyboardButton("✏️ Редактировать название", callback_data=f"edit_workout_name:{workout_id}")],
        [InlineKeyboardButton("➕ Добавить упражнение", callback_data=f"add_exercise_to_existing:{workout_id}")],
        [InlineKeyboardButton("📝 Редактировать упражнения", callback_data=f"edit_exercises:{workout_id}")],
        [InlineKeyboardButton("🗑️ Удалить тренировку", callback_data=f"confirm_delete_workout:{workout_id}")],
        [InlineKeyboardButton("⬅️ Назад к моим тренировкам", callback_data="workouts_my")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")

    context.user_data['current_viewed_workout_id'] = workout_id # Сохраняем ID для дальнейших действий


# --- Handlers for deleting a workout ---
async def confirm_delete_workout_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE, workout_id: int):
    workout = get_workout_template_by_id(workout_id, update.effective_user.id)
    if not workout:
        await update.callback_query.edit_message_text("Тренировка не найдена.", reply_markup=get_workouts_inline_keyboard())
        return

    text = f"Вы уверены, что хотите удалить тренировку '<b>{html.escape(workout['name'])}</b>' и все связанные с ней упражнения?\n" \
           "Это действие необратимо."
    keyboard = [
        [InlineKeyboardButton("✅ Да, удалить", callback_data=f"delete_workout_confirmed:{workout_id}")],
        [InlineKeyboardButton("❌ Нет, отмена", callback_data=f"cancel_delete_workout:{workout_id}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")

async def delete_workout_confirmed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    workout_id = int(query.data.split(":")[1])
    user_id = query.from_user.id

    delete_workout_template(workout_id, user_id)
    await query.edit_message_text("Тренировка успешно удалена.", reply_markup=get_workouts_inline_keyboard())
    # После удаления возвращаемся в список тренировок
    await show_my_workouts(update, context)
    return ConversationHandler.END # Завершаем диалог, если он был частью ConversationHandler

async def cancel_delete_workout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    workout_id = int(query.data.split(":")[1])
    await view_workout_details(update, context, workout_id) # Возвращаемся к деталям тренировки
    return ConversationHandler.END # Завершаем диалог, если он был частью ConversationHandler


# --- Handlers for editing workout name ---
async def start_edit_workout_name(update: Update, context: ContextTypes.DEFAULT_TYPE, workout_id: int) -> int:
    user_id = update.effective_user.id
    workout = get_workout_template_by_id(workout_id, user_id)
    if not workout:
        await update.callback_query.edit_message_text("Тренировка не найдена.", reply_markup=get_workouts_inline_keyboard())
        return ConversationHandler.END

    context.user_data['editing_workout_id'] = workout_id
    await update.callback_query.edit_message_text(
        f"Введите новое название для тренировки '<b>{html.escape(workout['name'])}</b>':",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Отмена", callback_data="cancel_edit_workout")]])
    )
    return EDIT_WORKOUT_NAME_STATE

async def process_edit_workout_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    new_name = update.message.text.strip()
    if not new_name:
        await update.message.reply_text("Название не может быть пустым. Введите новое название.")
        return EDIT_WORKOUT_NAME_STATE
    
    workout_id = context.user_data.get('editing_workout_id')
    user_id = update.effective_user.id
    
    update_workout_template_name(workout_id, user_id, new_name)
    await update.message.reply_text(f"Название тренировки изменено на '<b>{html.escape(new_name)}</b>'.", parse_mode="HTML")
    
    # Возвращаемся к деталям тренировки
    await view_workout_details(update, context, workout_id)
    
    context.user_data.pop('editing_workout_id', None) # Очищаем только этот ключ
    context.user_data.pop('editing_workout_name', None) # Очищаем только этот ключ
    return ConversationHandler.END

# --- Handlers for adding exercise to existing workout ---
async def start_add_exercise_to_existing(update: Update, context: ContextTypes.DEFAULT_TYPE, workout_id: int) -> int:
    user_id = update.effective_user.id
    workout = get_workout_template_by_id(workout_id, user_id)
    if not workout:
        await update.callback_query.edit_message_text("Тренировка не найдена.", reply_markup=get_workouts_inline_keyboard())
        return ConversationHandler.END

    context.user_data['current_workout_id'] = workout_id # Используем тот же ключ, что и при создании
    context.user_data['current_workout_name'] = workout['name']
    context.user_data['exercises_count'] = len(get_exercises_for_workout(workout_id)) # Обновляем счетчик
    
    await update.callback_query.edit_message_text(
        f"Добавляем упражнение в тренировку '<b>{html.escape(workout['name'])}</b>'.\nКак называется упражнение?",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Отмена", callback_data="cancel_add_workout")]])
    )
    return ADD_EXERCISE_NAME # Переходим в то же состояние, что и при создании

async def get_exercise_name_for_existing_workout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Повторно используем get_exercise_name, но с другим переходом
    return await get_exercise_name(update, context)

async def get_exercise_sets_for_existing_workout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await get_exercise_sets(update, context)

async def get_exercise_reps_for_existing_workout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await get_exercise_reps(update, context)

async def get_exercise_comment_for_existing_workout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await get_exercise_comment(update, context)

async def confirm_add_exercise_to_existing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "add_another_exercise":
        await query.edit_message_text(
            "Отлично, давай добавим следующее упражнение. Как оно называется?",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Отмена", callback_data="cancel_add_workout")]])
        )
        return ADD_EXERCISE_NAME
    elif data == "finish_workout_creation":
        workout_id = context.user_data.get('current_workout_id')
        await view_workout_details(update, context, workout_id) # Возвращаемся к деталям тренировки

        context.user_data.pop('current_workout_id', None)
        context.user_data.pop('current_workout_name', None)
        context.user_data.pop('exercises_count', None)
        context.user_data.pop('current_exercise_name', None)
        context.user_data.pop('current_exercise_sets', None)
        context.user_data.pop('current_exercise_reps', None)
        context.user_data.pop('current_exercise_comment', None)
        return ConversationHandler.END
    elif data == "cancel_add_workout":
        workout_id = context.user_data.get('current_workout_id')
        await view_workout_details(update, context, workout_id) # Возвращаемся к деталям тренировки
        context.user_data.pop('current_workout_id', None)
        context.user_data.pop('current_workout_name', None)
        context.user_data.pop('exercises_count', None)
        context.user_data.pop('current_exercise_name', None)
        context.user_data.pop('current_exercise_sets', None)
        context.user_data.pop('current_exercise_reps', None)
        context.user_data.pop('current_exercise_comment', None)
        return ConversationHandler.END

# --- Handlers for editing exercises list ---
async def start_edit_exercises_list(update: Update, context: ContextTypes.DEFAULT_TYPE, workout_id: int) -> int:
    user_id = update.effective_user.id
    workout = get_workout_template_by_id(workout_id, user_id)
    
    if not workout:
        error_text = "Тренировка не найдена."
        if update.callback_query:
            await update.callback_query.edit_message_text(error_text, reply_markup=get_workouts_inline_keyboard())
        else:
            await update.message.reply_text(error_text, reply_markup=get_workouts_inline_keyboard())
        return ConversationHandler.END

    exercises = get_exercises_for_workout(workout_id)
    context.user_data['editing_workout_id'] = workout_id
    
    text = f"<b>📝 Редактирование упражнений в '{html.escape(workout['name'])}'</b>\n\n"
    keyboard = []

    if not exercises:
        text += "В этой тренировке пока нет упражнений для редактирования."
    else:
        for exercise in exercises:
            keyboard.append([
                InlineKeyboardButton(f"✏️ {exercise['name']}", callback_data=f"edit_ex_select:{exercise['exercise_id']}"),
                InlineKeyboardButton(f"🗑️", callback_data=f"confirm_del_ex:{exercise['exercise_id']}")
            ])
    
    keyboard.append([InlineKeyboardButton("⬅️ Готово", callback_data="finish_edit_exercises")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        # This case happens when we return to the list after editing a field via text message
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")

    return EDIT_EXERCISE_SELECT

async def select_exercise_to_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    # Делегируем обработку на основе префикса callback_data
    if data.startswith("edit_field:"):
        return await edit_field_handler(update, context)
    elif data == "finish_edit_exercises":
        return await finish_edit_exercises(update, context)
    elif data.startswith("confirm_del_ex:"):
        return await confirm_delete_exercise_dialog(update, context)
    elif data.startswith("delete_exercise_confirmed:"):
        return await delete_exercise_confirmed(update, context)
    elif data.startswith("cancel_delete_exercise:"):
        return await cancel_delete_exercise(update, context)
    elif data.startswith("edit_ex_select:"):
        # Это фактический выбор упражнения для просмотра/редактирования его деталей
        exercise_id = int(data.split(":")[1])
        
        exercise = get_exercise_by_id(exercise_id)
        if not exercise:
            await query.edit_message_text("Упражнение не найдено.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="finish_edit_exercises")]]))
            return EDIT_EXERCISE_SELECT

        context.user_data['editing_exercise_id'] = exercise_id
        context.user_data['editing_exercise_data'] = exercise # Сохраняем текущие данные для удобства

        text = f"<b>✏️ Редактирование упражнения '{html.escape(exercise['name'])}'</b>\n\n" \
               f"Текущие данные:\n" \
               f"Название: {html.escape(exercise['name'])}\n" \
               f"Подходы: {exercise['planned_sets']}\n" \
               f"Повторения: {html.escape(str(exercise['planned_reps']))}\n" \
               f"Комментарий: {html.escape(exercise['comment'] or 'Нет')}\n\n" \
               f"Что вы хотите изменить?"
        
        keyboard = [
            [InlineKeyboardButton("Название", callback_data="edit_field:name")],
            [InlineKeyboardButton("Подходы", callback_data="edit_field:edit_sets")], # Изменено на edit_sets
            [InlineKeyboardButton("Повторения", callback_data="edit_field:edit_reps")], # Изменено на edit_reps
            [InlineKeyboardButton("Комментарий", callback_data="edit_field:edit_comment")], # Изменено на edit_comment
            [InlineKeyboardButton("⬅️ Назад к списку упражнений", callback_data="finish_edit_exercises")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")

        return EDIT_EXERCISE_SELECT # Остаемся в этом состоянии, пока не выберем поле для редактирования
    
    # Fallback для любых неожиданных callback_data, которые все еще соответствуют паттерну состояния
    logging.warning(f"Unexpected callback data in select_exercise_to_edit: {data}")
    # Попытаемся вернуться к списку упражнений, если возможно
    workout_id = context.user_data.get('editing_workout_id')
    if workout_id:
        await start_edit_exercises_list(update, context, workout_id)
    else:
        await show_my_workouts(update, context)
    return EDIT_EXERCISE_SELECT

async def edit_field_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    field = query.data.split(":")[1]
    
    exercise_data = context.user_data['editing_exercise_data']
    context.user_data['editing_field'] = field

    if field == "name":
        await query.edit_message_text(f"Введите новое название для упражнения '<b>{html.escape(exercise_data['name'])}</b>':", parse_mode="HTML")
        return EDIT_EXERCISE_NAME_STATE
    elif field == "edit_sets": # Изменено на edit_sets
        await query.edit_message_text(f"Введите новое количество подходов для '<b>{html.escape(exercise_data['name'])}</b>' (текущее: {exercise_data['planned_sets']}):", parse_mode="HTML")
        return EDIT_EXERCISE_SETS_STATE
    elif field == "edit_reps": # Изменено на edit_reps
        await query.edit_message_text(f"Введите новое количество повторений для '<b>{html.escape(exercise_data['name'])}</b>' (текущее: {html.escape(str(exercise_data['planned_reps']))}):", parse_mode="HTML")
        return EDIT_EXERCISE_REPS_STATE
    elif field == "edit_comment": # Изменено на edit_comment
        await query.edit_message_text(f"Введите новый комментарий для '<b>{html.escape(exercise_data['name'])}</b>' (текущий: {html.escape(exercise_data['comment'] or 'Нет')})\nИли '-' для удаления:", parse_mode="HTML")
        return EDIT_EXERCISE_COMMENT_STATE
    
    return EDIT_EXERCISE_SELECT # На всякий случай

async def process_edit_exercise_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    new_name = update.message.text.strip()
    if not new_name:
        await update.message.reply_text("Название не может быть пустым. Введите новое название.")
        return EDIT_EXERCISE_NAME_STATE
    
    exercise_id = context.user_data['editing_exercise_id']
    exercise_data = context.user_data['editing_exercise_data']
    
    update_exercise(exercise_id, new_name, exercise_data['planned_sets'], exercise_data['planned_reps'], exercise_data['comment'])
    await update.message.reply_text(f"Название упражнения изменено на '<b>{html.escape(new_name)}</b>'.", parse_mode="HTML")
    
    # Возвращаемся к выбору упражнения для редактирования
    workout_id = context.user_data['editing_workout_id']
    await start_edit_exercises_list(update, context, workout_id)
    context.user_data.pop('editing_field', None)
    context.user_data.pop('editing_exercise_id', None)
    context.user_data.pop('editing_exercise_data', None)
    return EDIT_EXERCISE_SELECT

async def process_edit_exercise_sets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        new_sets = int(update.message.text.strip())
        if new_sets <= 0:
            raise ValueError
        
        exercise_id = context.user_data['editing_exercise_id']
        exercise_data = context.user_data['editing_exercise_data']
        
        update_exercise(exercise_id, exercise_data['name'], new_sets, exercise_data['planned_reps'], exercise_data['comment'])
        await update.message.reply_text(f"Количество подходов для '<b>{html.escape(exercise_data['name'])}</b>' изменено на {new_sets}.", parse_mode="HTML")
        
        workout_id = context.user_data['editing_workout_id']
        await start_edit_exercises_list(update, context, workout_id)
        context.user_data.pop('editing_field', None)
        context.user_data.pop('editing_exercise_id', None)
        context.user_data.pop('editing_exercise_data', None)
        return EDIT_EXERCISE_SELECT
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите корректное число подходов (целое положительное число).")
        return EDIT_EXERCISE_SETS_STATE

async def process_edit_exercise_reps(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    new_reps = update.message.text.strip()
    if not new_reps:
        await update.message.reply_text("Количество повторений не может быть пустым. Введите новое количество повторений.")
        return EDIT_EXERCISE_REPS_STATE
    
    exercise_id = context.user_data['editing_exercise_id']
    exercise_data = context.user_data['editing_exercise_data']
    
    update_exercise(exercise_id, exercise_data['name'], exercise_data['planned_sets'], new_reps, exercise_data['comment'])
    await update.message.reply_text(f"Количество повторений для '<b>{html.escape(exercise_data['name'])}</b>' изменено на {html.escape(new_reps)}.", parse_mode="HTML")
    
    workout_id = context.user_data['editing_workout_id']
    await start_edit_exercises_list(update, context, workout_id)
    context.user_data.pop('editing_field', None)
    context.user_data.pop('editing_exercise_id', None)
    context.user_data.pop('editing_exercise_data', None)
    return EDIT_EXERCISE_SELECT

async def process_edit_exercise_comment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    new_comment = update.message.text.strip()
    new_comment = new_comment if new_comment != '-' else None
    
    exercise_id = context.user_data['editing_exercise_id']
    exercise_data = context.user_data['editing_exercise_data']
    
    update_exercise(exercise_id, exercise_data['name'], exercise_data['planned_sets'], exercise_data['planned_reps'], new_comment)
    await update.message.reply_text(f"Комментарий для '<b>{html.escape(exercise_data['name'])}</b>' обновлен.", parse_mode="HTML")
    
    workout_id = context.user_data['editing_workout_id']
    await start_edit_exercises_list(update, context, workout_id)
    context.user_data.pop('editing_field', None)
    context.user_data.pop('editing_exercise_id', None)
    context.user_data.pop('editing_exercise_data', None)
    return EDIT_EXERCISE_SELECT

async def confirm_delete_exercise_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    exercise_id = int(query.data.split(":")[1])
    
    exercise = get_exercise_by_id(exercise_id)
    if not exercise:
        await query.edit_message_text("Упражнение не найдено.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="finish_edit_exercises")]]))
        return EDIT_EXERCISE_SELECT

    text = f"Вы уверены, что хотите удалить упражнение '<b>{html.escape(exercise['name'])}</b>' из этой тренировки?"
    keyboard = [
        [InlineKeyboardButton("✅ Да, удалить", callback_data=f"delete_exercise_confirmed:{exercise_id}")],
        [InlineKeyboardButton("❌ Нет, отмена", callback_data=f"cancel_delete_exercise:{exercise_id}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    return CONFIRM_DELETE_EXERCISE

async def delete_exercise_confirmed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    exercise_id = int(query.data.split(":")[1])
    
    delete_exercise(exercise_id)
    await query.edit_message_text("Упражнение успешно удалено.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="finish_edit_exercises")]]))
    
    workout_id = context.user_data['editing_workout_id']
    await start_edit_exercises_list(update, context, workout_id) # Обновляем список упражнений
    return EDIT_EXERCISE_SELECT

async def cancel_delete_exercise(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    # Возвращаемся к списку упражнений для редактирования
    workout_id = context.user_data['editing_workout_id']
    await start_edit_exercises_list(update, context, workout_id)
    return EDIT_EXERCISE_SELECT

async def finish_edit_exercises(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    workout_id = context.user_data.get('editing_workout_id')
    if workout_id:
        await view_workout_details(update, context, workout_id)
    else:
        await show_my_workouts(update, context)
    context.user_data.pop('editing_workout_id', None) # Очищаем только этот ключ
    context.user_data.pop('editing_exercise_id', None) # Очищаем только этот ключ
    context.user_data.pop('editing_exercise_data', None) # Очищаем только этот ключ
    context.user_data.pop('editing_field', None) # Очищаем только этот ключ
    return ConversationHandler.END

# --- Handlers for starting a workout session ---
async def start_workout_session(update: Update, context: ContextTypes.DEFAULT_TYPE, workout_id: int) -> int:
    user_id = update.effective_user.id
    workout = get_workout_template_by_id(workout_id, user_id)
    if not workout:
        await update.callback_query.edit_message_text("Тренировка не найдена или у вас нет к ней доступа.", reply_markup=get_workouts_inline_keyboard())
        return ConversationHandler.END

    exercises = get_exercises_for_workout(workout_id)
    if not exercises:
        await update.callback_query.edit_message_text("В этой тренировке нет упражнений. Добавьте их перед запуском.", reply_markup=get_workouts_inline_keyboard())
        return ConversationHandler.END

    logged_workout_id = start_logged_workout(user_id, workout_id)
    
    context.user_data['active_workout'] = {
        'logged_workout_id': logged_workout_id,
        'workout_id': workout_id,
        'exercises': exercises,
        'current_exercise_index': 0,
        'current_set_number': 1,
        'current_exercise_data': None, # Будет заполнено при переходе к упражнению
        'current_weight': 0.0,
        'current_reps': 0
    }
    
    await update.callback_query.edit_message_text(f"Начинаем тренировку '<b>{html.escape(workout['name'])}</b>'!", parse_mode="HTML")
    return await next_set_or_exercise(update, context)

async def next_set_or_exercise(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    workout_state = context.user_data['active_workout']
    exercises = workout_state['exercises']
    current_exercise_index = workout_state['current_exercise_index']
    current_set_number = workout_state['current_set_number']

    if current_exercise_index >= len(exercises):
        # Все упражнения и подходы выполнены
        await end_workout_session(update, context)
        return ConversationHandler.END

    current_exercise = exercises[current_exercise_index]
    workout_state['current_exercise_data'] = current_exercise

    if current_set_number > current_exercise['planned_sets']:
        # Все подходы для текущего упражнения выполнены, переходим к следующему упражнению
        workout_state['current_exercise_index'] += 1
        workout_state['current_set_number'] = 1
        return await next_set_or_exercise(update, context) # Рекурсивный вызов для следующего упражнения

    
    # Отображаем текущий подход
    new_text = f"<b>🏋️‍♀️ {html.escape(current_exercise['name'])}</b>\n" \
               f"Подход {current_set_number} из {current_exercise['planned_sets']}\n"
    
    # Получаем данные из прошлой тренировки
    last_set_data = get_last_set_data_for_exercise(update.effective_user.id, current_exercise['exercise_id'])
    
    suggested_weight = last_set_data['weight'] if last_set_data else 0.0
    # Попытка извлечь число из planned_reps, если это строка типа "8-12"
    if isinstance(current_exercise['planned_reps'], str) and '-' in current_exercise['planned_reps']:
        try:
            suggested_reps = int(current_exercise['planned_reps'].split('-')[0])
        except ValueError:
            suggested_reps = 0 # Fallback
    else:
        try:
            suggested_reps = int(current_exercise['planned_reps'])
        except ValueError:
            suggested_reps = 0 # Fallback


    if last_set_data:
        new_text += f"<i>В прошлый раз: {last_set_data['weight']} кг на {last_set_data['reps_performed']} повторений.</i>\n"
    
    new_text += f"\nКакой вес и количество повторений сейчас?"

    # Инициализируем текущие значения для кнопок
    workout_state['current_weight'] = suggested_weight
    workout_state['current_reps'] = suggested_reps

    new_reply_markup = get_set_logging_keyboard(suggested_weight, suggested_reps)

    try:
        if update.callback_query:
            # Сравниваем с текущим сообщением, чтобы избежать "Message is not modified"
            current_message_text = update.callback_query.message.text_html
            current_message_reply_markup = update.callback_query.message.reply_markup

            if current_message_text == new_text and \
               (current_message_reply_markup and current_message_reply_markup.to_dict() == new_reply_markup.to_dict()):
                logging.debug("Message content is identical, skipping edit_message_text in next_set_or_exercise.")
                # query.answer() уже был вызван в log_set_confirm
                pass
            else:
                await update.callback_query.edit_message_text(new_text, reply_markup=new_reply_markup, parse_mode="HTML")
        else:
            # Это может быть первый вызов после start_workout_session, где update.callback_query.message уже отредактировано
            # или если мы пришли сюда из другого места (например, отмена)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=new_text, reply_markup=new_reply_markup, parse_mode="HTML")
    except BadRequest as e:
        if "Message is not modified" in str(e):
            logging.debug("Caught 'Message is not modified' error in next_set_or_exercise, ignoring.")
        else:
            raise # Перевыбрасываем другие BadRequest ошибки
    
    return LOG_SET_WEIGHT # Переходим в состояние ожидания ввода веса/повторений через кнопки

def get_set_logging_keyboard(current_weight: float, current_reps: int) -> InlineKeyboardMarkup:
    # Форматируем вес, чтобы избежать проблем с точностью чисел с плавающей запятой в строковом представлении
    # Например, 50.0 -> "50", 50.5 -> "50.5"
    formatted_weight = f"{current_weight:.1f}".rstrip('0').rstrip('.')
    if formatted_weight == "": # Если вес был 0.0, то rstrip может сделать его пустым
        formatted_weight = "0"

    weight_buttons = [
        InlineKeyboardButton(f"-5 кг", callback_data=f"adjust_weight:-5"),
        InlineKeyboardButton(f"-2.5 кг", callback_data=f"adjust_weight:-2.5"),
        InlineKeyboardButton(f"{formatted_weight} кг", callback_data=f"adjust_weight:0"), # Текущий вес
        InlineKeyboardButton(f"+2.5 кг", callback_data=f"adjust_weight:+2.5"),
        InlineKeyboardButton(f"+5 кг", callback_data=f"adjust_weight:+5")
    ]
    reps_buttons = [
        InlineKeyboardButton(f"-3 повт.", callback_data=f"adjust_reps:-3"),
        InlineKeyboardButton(f"-1 повт.", callback_data=f"adjust_reps:-1"),
        InlineKeyboardButton(f"{current_reps} повт.", callback_data=f"adjust_reps:0"), # Текущие повторения
        InlineKeyboardButton(f"+1 повт.", callback_data=f"adjust_reps:+1"),
        InlineKeyboardButton(f"+3 повт.", callback_data=f"adjust_reps:+3")
    ]
    action_buttons = [
        InlineKeyboardButton("✅ Выполнил подход", callback_data="log_set_confirm"),
        InlineKeyboardButton("⏭️ Пропустить подход", callback_data="skip_set")
    ]
    return InlineKeyboardMarkup([weight_buttons, reps_buttons, action_buttons])

async def adjust_set_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data
    
    workout_state = context.user_data['active_workout']
    old_weight = workout_state['current_weight']
    old_reps = workout_state['current_reps']
    current_exercise = workout_state['current_exercise_data']

    new_weight = old_weight # Инициализируем новыми значениями
    new_reps = old_reps

    if data.startswith("adjust_weight:"):
        change = float(data.split(":")[1])
        new_weight = max(0.0, old_weight + change) # Вес не может быть отрицательным
        workout_state['current_weight'] = new_weight
    elif data.startswith("adjust_reps:"):
        change = int(data.split(":")[1])
        new_reps = max(0, old_reps + change) # Повторения не могут быть отрицательными
        workout_state['current_reps'] = new_reps
    else:
        # Если callback_data не является adjust_weight или adjust_reps,
        # это может быть ошибка или нежелательное поведение, просто отвечаем и остаемся в состоянии
        return LOG_SET_WEIGHT

    # Генерируем новый текст и разметку
    new_text = f"<b>🏋️‍♀️ {html.escape(current_exercise['name'])}</b>\n" \
               f"Подход {workout_state['current_set_number']} из {current_exercise['planned_sets']}\n"
    
    last_set_data = get_last_set_data_for_exercise(update.effective_user.id, current_exercise['exercise_id'])
    if last_set_data:
        new_text += f"<i>В прошлый раз: {last_set_data['weight']} кг на {last_set_data['reps_performed']} повторений.</i>\n"
    
    new_text += f"\nКакой вес и количество повторений сейчас?"

    new_reply_markup = get_set_logging_keyboard(new_weight, new_reps)

    # Сравниваем с текущим сообщением, чтобы избежать "Message is not modified"
    current_message_text = query.message.text_html
    current_message_reply_markup = query.message.reply_markup

    if current_message_text == new_text and \
       (current_message_reply_markup and current_message_reply_markup.to_dict() == new_reply_markup.to_dict()):
        logging.debug("Message content is identical, skipping edit_message_text in adjust_set_data.")
        return LOG_SET_WEIGHT
    
    try:
        await query.edit_message_text(new_text, reply_markup=new_reply_markup, parse_mode="HTML")
    except BadRequest as e:
        if "Message is not modified" in str(e):
            logging.debug("Caught 'Message is not modified' error in adjust_set_data, ignoring.")
        else:
            raise # Перевыбрасываем другие BadRequest ошибки
    
    return LOG_SET_WEIGHT # Остаемся в этом состоянии

async def log_set_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    workout_state = context.user_data['active_workout']
    logged_workout_id = workout_state['logged_workout_id']
    current_exercise = workout_state['current_exercise_data']
    current_set_number = workout_state['current_set_number']
    current_weight = workout_state['current_weight']
    current_reps = workout_state['current_reps']

    add_logged_set(logged_workout_id, current_exercise['exercise_id'], current_set_number, current_weight, current_reps)
    
    workout_state['current_set_number'] += 1 # Переходим к следующему подходу
    
    return await next_set_or_exercise(update, context)

async def skip_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Skips the current set and moves to the next one."""
    query = update.callback_query
    await query.answer("Подход пропущен")
    
    workout_state = context.user_data['active_workout']
    workout_state['current_set_number'] += 1 # Просто увеличиваем счетчик подходов
    
    return await next_set_or_exercise(update, context)

async def end_workout_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    workout_state = context.user_data['active_workout']
    logged_workout_id = workout_state['logged_workout_id']
    end_logged_workout(logged_workout_id)
    
    # Определяем, куда отправлять сообщение о завершении
    if update.callback_query:
        await update.callback_query.edit_message_text("🎉 Тренировка завершена! Отличная работа!", reply_markup=get_workouts_inline_keyboard())
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="🎉 Тренировка завершена! Отличная работа!", reply_markup=get_workouts_inline_keyboard())
    
    context.user_data.clear()
    return ConversationHandler.END

async def cancel_workout_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    workout_state = context.user_data.get('active_workout')
    if workout_state:
        # Здесь можно удалить logged_workout_id, если тренировка не была завершена
        # Но для простоты пока оставим, чтобы не усложнять логику удаления незавершенных
        pass
    
    await query.edit_message_text("❌ Тренировка отменена.", reply_markup=get_workouts_inline_keyboard())
    context.user_data.clear()
    return ConversationHandler.END


# --- ConversationHandler for adding a workout ---
add_workout_conversation_handler = ConversationHandler(
    entry_points=[
        CallbackQueryHandler(start_add_workout, pattern='^workouts_add$')
    ],
    states={
        ADD_WORKOUT_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_workout_name)],
        ADD_EXERCISE_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_exercise_name)],
        ADD_EXERCISE_SETS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_exercise_sets)],
        ADD_EXERCISE_REPS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_exercise_reps)],
        ADD_EXERCISE_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_exercise_comment)],
        CONFIRM_ADD_EXERCISE: [CallbackQueryHandler(confirm_add_exercise, pattern='^(add_another_exercise|finish_workout_creation|cancel_add_workout)$')]
    },
    fallbacks=[
        CallbackQueryHandler(cancel_conversation, pattern='^cancel_add_workout$'),
        CommandHandler('cancel', cancel_conversation)
    ],
    map_to_parent={
        ConversationHandler.END: 0 # Возвращаемся в меню тренировок после завершения
    }
)

# --- ConversationHandler for editing a workout ---
edit_workout_conversation_handler = ConversationHandler(
    entry_points=[
        CallbackQueryHandler(lambda update, context: start_edit_workout_name(update, context, int(update.callback_query.data.split(":")[1])), pattern='^edit_workout_name:'),
        CallbackQueryHandler(lambda update, context: start_add_exercise_to_existing(update, context, int(update.callback_query.data.split(":")[1])), pattern='^add_exercise_to_existing:'),
        CallbackQueryHandler(lambda update, context: start_edit_exercises_list(update, context, int(update.callback_query.data.split(":")[1])), pattern='^edit_exercises:')
    ],
    states={
        EDIT_WORKOUT_NAME_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_edit_workout_name)],
        # Состояния для добавления упражнения к существующей тренировке
        ADD_EXERCISE_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_exercise_name_for_existing_workout)],
        ADD_EXERCISE_SETS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_exercise_sets_for_existing_workout)],
        ADD_EXERCISE_REPS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_exercise_reps_for_existing_workout)],
        ADD_EXERCISE_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_exercise_comment_for_existing_workout)],
        CONFIRM_ADD_EXERCISE: [CallbackQueryHandler(confirm_add_exercise_to_existing, pattern='^(add_another_exercise|finish_workout_creation|cancel_add_workout)$')],
        # Состояния для редактирования списка упражнений
        EDIT_EXERCISE_SELECT: [CallbackQueryHandler(select_exercise_to_edit, pattern='^edit_ex_select:(\d+)$|^edit_field:.*$|^finish_edit_exercises$|^confirm_del_ex:(\d+)$|^delete_exercise_confirmed:(\d+)$|^cancel_delete_exercise:(\d+)$')],
        EDIT_EXERCISE_NAME_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_edit_exercise_name)],
        EDIT_EXERCISE_SETS_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_edit_exercise_sets)],
        EDIT_EXERCISE_REPS_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_edit_exercise_reps)],
        EDIT_EXERCISE_COMMENT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_edit_exercise_comment)],
        CONFIRM_DELETE_EXERCISE: [CallbackQueryHandler(confirm_delete_exercise_dialog, pattern='^confirm_del_ex:(\d+)$|^delete_exercise_confirmed:(\d+)$|^cancel_delete_exercise:(\d+)$')]
    },
    fallbacks=[
        CallbackQueryHandler(cancel_conversation, pattern='^cancel_edit_workout$'),
        CommandHandler('cancel', cancel_conversation)
    ],
    map_to_parent={
        ConversationHandler.END: 0
    }
)

# --- ConversationHandler for running a workout ---
run_workout_conversation_handler = ConversationHandler(
    entry_points=[
        CallbackQueryHandler(lambda update, context: start_workout_session(update, context, int(update.callback_query.data.split(":")[1])), pattern='^start_workout:')
    ],
    states={
        LOG_SET_WEIGHT: [
            CallbackQueryHandler(adjust_set_data, pattern='^adjust_(weight|reps):.*$'),
            CallbackQueryHandler(log_set_confirm, pattern='^log_set_confirm$'),
            CallbackQueryHandler(skip_set, pattern='^skip_set$'),
            CallbackQueryHandler(cancel_workout_session, pattern='^cancel_workout_session$'),
        ],
        NEXT_SET_OR_EXERCISE: [CallbackQueryHandler(next_set_or_exercise)], # Для перехода между подходами/упражнениями
    },
    fallbacks=[
        CallbackQueryHandler(cancel_workout_session, pattern='^cancel_workout_session$'),
        CommandHandler('cancel', cancel_conversation)
    ],
    map_to_parent={
        ConversationHandler.END: 0
    }
)


async def handle_workouts_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Handles callbacks related to workouts menu."""
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id

    # Проверяем, не является ли callback частью активного ConversationHandler
    # Если да, то он будет обработан соответствующим ConversationHandler'ом
    # Здесь мы обрабатываем только те, которые запускают ConversationHandler или являются частью основного меню
    if data.startswith("workouts_add"):
        return False # Будет обработан add_workout_conversation_handler

    if data.startswith("edit_workout_name:") or \
       data.startswith("add_exercise_to_existing:") or \
       data.startswith("edit_exercises:"):
        return False # Будет обработан edit_workout_conversation_handler

    if data.startswith("start_workout:"):
        return False # Будет обработан run_workout_conversation_handler

    # Обработка остальных callback'ов
    if not data.startswith("workouts_") and data != "main_menu" \
       and not data.startswith("view_workout:") \
       and not data.startswith("confirm_delete_workout:") \
       and not data.startswith("delete_workout_confirmed:") \
       and not data.startswith("cancel_delete_workout:"):
        return False

    await query.answer()

    if data == "workouts_my":
        await show_my_workouts(update, context)
    elif data == "workouts_stats":
        await query.edit_message_text("Здесь будет статистика тренировок.") # Placeholder
    elif data == "main_menu" or data == "workouts_menu": # Добавляем обработку "workouts_menu" для возврата
        await show_workouts_menu(update, context)
    elif data.startswith("view_workout:"):
        workout_id = int(data.split(":")[1])
        await view_workout_details(update, context, workout_id)
    elif data.startswith("confirm_delete_workout:"):
        workout_id = int(data.split(":")[1])
        await confirm_delete_workout_dialog(update, context, workout_id)
    elif data.startswith("delete_workout_confirmed:"):
        await delete_workout_confirmed(update, context)
    elif data.startswith("cancel_delete_workout:"):
        await cancel_delete_workout(update, context)
    
    return True