import os
import re
import asyncio
from faster_whisper import WhisperModel
from telegram import LinkPreviewOptions

# Инициализируем модель Whisper (base — оптимально для CPU)
model_whisper = WhisperModel("base", device="cpu", compute_type="int8")

# --- Регулярки для форматирования ---
bold_re = re.compile(r'(\*\*|__)(.*?)\1', re.DOTALL)
italic_re = re.compile(r'(\*|_)(.*?)\1', re.DOTALL)
code_re = re.compile(r'`(.*?)`')
pre_re = re.compile(r'```(?:.*?)\n?(.*?)```', re.DOTALL)


def escape_html(content: str) -> str:
    return content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def format_to_html(text: str) -> str:
    if not text: return ""
    text = bold_re.sub(lambda m: f'<b>{escape_html(m.group(2))}</b>', text)
    text = italic_re.sub(lambda m: f'<i>{escape_html(m.group(2))}</i>', text)
    text = pre_re.sub(lambda m: f'<pre>{escape_html(m.group(1))}</pre>', text)
    text = code_re.sub(lambda m: f'<code>{escape_html(m.group(1))}</code>', text)
    return text


def get_model_short_name(model_path: str, provider: str) -> str:
    if not model_path: return "Auto"
    if provider == "gemini":
        return model_path.replace("models/", "")
    # Для OpenRouter убираем автора и :free
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
    if not message:
        return

    text = message.text or message.caption or ""
    if not text:
        return

    replacements = {
        r"instagram\.com/": "kkinstagram.com/",
        r"(vm|vt|www)\.tiktok\.com/": "vxtiktok.com/",
        r"(twitter|x)\.com/": "fxtwitter.com/",
    }

    new_text = text
    found = False

    # Проверяем, есть ли в тексте наши "проблемные" ссылки
    for pattern, rep in replacements.items():
        if re.search(pattern, new_text):
            new_text = re.sub(pattern, rep, new_text)
            found = True

    # Если это НЕ ТикТок/Инста/Х — выходим сразу (ничего не делаем)
    if not found:
        return

    # --- ПУНКТ №4 МЫ УДАЛИЛИ ---
    # Теперь бот будет исправлять эти ссылки ВСЕГДА, даже без "Андрюхи"

    user_name = message.from_user.first_name
    url_match = re.search(r"https?://\S+", new_text)
    hidden_link = f'<a href="{url_match.group(0)}">\u200b</a>' if url_match else ""
    final_text = f"{hidden_link}✅ <b>От {user_name}:</b>\n{new_text}"

    try:
        # Пытаемся удалить оригинал
        await message.delete()

        # Отправляем исправленную
        await context.bot.send_message(
            chat_id=message.chat_id,
            text=final_text,
            parse_mode="HTML",
            message_thread_id=message.message_thread_id,
            link_preview_options=LinkPreviewOptions(is_disabled=False, prefer_large_media=True)
        )
    except Exception as e:
        # Если бот не админ, он не сможет удалить сообщение (ошибка 403/400)
        print(f"⚠️ Ошибка LinkFixer (проверь права админа!): {e}")
        # Если удалить не вышло, просто присылаем исправленную версию вдогонку
        await context.bot.send_message(
            chat_id=message.chat_id,
            text=f"Исправленная ссылка:\n{final_text}",
            parse_mode="HTML",
            message_thread_id=message.message_thread_id
        )