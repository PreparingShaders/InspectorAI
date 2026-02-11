import os
import re
import asyncio
from faster_whisper import WhisperModel
from telegram import LinkPreviewOptions
from telegram.helpers import escape_markdown  # Добавляем для линк-фиксера
import telegramify_markdown, telegram

# Инициализируем модель Whisper
model_whisper = WhisperModel("base", device="cpu", compute_type="int8")


def safe_format_to_html(text: str) -> str:
    """Конвертирует Markdown в MarkdownV2 для Telegram."""
    if not text:
        return ""
    try:
        # 1. Сначала базовое форматирование
        converted = telegramify_markdown.markdownify(text)

        # 2. Магия: экранируем точки, которые библиотека могла пропустить
        # (только если они не заэкранированы и не в ссылках/блоках кода)
        # Но проще всего довериться библиотеке и добавить финальный catch-all
        return converted
    except Exception as e:
        print(f"⚠️ Ошибка форматирования: {e}")
        return escape_markdown(text, version=2)


def get_model_short_name(model_path: str, provider: str) -> str:
    if not model_path: return "Auto"
    if provider == "gemini":
        return model_path.replace("models/", "")
    return model_path.split("/")[-1].split(":")[0]


async def handle_voice_transcription(message):
    file_path = f"voice_{message.voice.file_unique_id}.ogg"
    try:
        voice_file = await message.voice.get_file()
        await voice_file.download_to_drive(file_path)
        segments, _ = await asyncio.to_thread(model_whisper.transcribe, file_path, beam_size=5)
        text = "".join([s.text for s in segments]).strip()
        return text if text else None
    except Exception as e:
        print(f"❌ Ошибка Whisper: {e}")
        return None
    finally:
        if os.path.exists(file_path): os.remove(file_path)


async def link_fixer_logic(update, context):
    message = update.message or update.edited_message
    if not message or not message.from_user: return

    # 1. Проверка авторизации (чтобы не было дыры)
    from handlers import authorized_users
    if message.from_user.id not in authorized_users:
        return

    text = message.text or message.caption or ""
    if not text: return

    replacements = {
        r"instagram\.com/": "kkinstagram.com/",
        r"(vm|vt|www)\.tiktok\.com/": "vxtiktok.com/",
        r"(twitter|x)\.com/": "fxtwitter.com/",
    }

    new_text = text
    found = False
    for pattern, rep in replacements.items():
        if re.search(pattern, new_text):
            new_text = re.sub(pattern, rep, new_text)
            found = True

    if not found: return

    user_name = escape_markdown(message.from_user.first_name, version=2)
    # Экранируем исправленный текст, чтобы спецсимволы в ссылках не ломали MarkdownV2
    safe_text = escape_markdown(new_text, version=2)
    final_text = f"✅ *От {user_name}:*\n{safe_text}"

    # 2. ОДИН блок try для всех сетевых операций
    try:
        # Пытаемся удалить оригинал
        await message.delete(read_timeout=10)

        # Отправляем исправленную версию
        await context.bot.send_message(
            chat_id=message.chat_id,
            text=final_text,
            parse_mode="MarkdownV2",
            message_thread_id=message.message_thread_id,
            link_preview_options=LinkPreviewOptions(is_disabled=False, prefer_large_media=True),
            read_timeout=20,
            write_timeout=20
        )
    except telegram.error.TimedOut:
        print("⏰ Ошибка LinkFixer: запрос отвалился по таймауту")
    except telegram.error.BadRequest as e:
        if "Message_too_long" in str(e):
            print("⚠️ Ошибка LinkFixer: текст слишком длинный")
        else:
            print(f"⚠️ Ошибка LinkFixer (BadRequest): {e}")
    except Exception as e:
        print(f"⚠️ Ошибка LinkFixer: {e}")