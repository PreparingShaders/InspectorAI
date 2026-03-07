import os
import re
import asyncio
import html
from faster_whisper import WhisperModel
from telegram import LinkPreviewOptions
import telegram

# Инициализируем модель Whisper
model_whisper = WhisperModel("base", device="cpu", compute_type="int8")

def to_html(text: str) -> str:
    """
    Преобразует Markdown от LLM в безопасный HTML для Telegram.
    """
    if not text:
        return ""

    # 1. Экранируем основные HTML-символы
    escaped_text = html.escape(text)

    # 2. Заменяем Markdown на HTML-теги
    # Жирный текст: **text** -> <b>text</b>
    escaped_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped_text)
    # Курсив: *text* -> <i>text</i>
    escaped_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', escaped_text)
    # Зачеркнутый: ~~text~~ -> <s>text</s>
    escaped_text = re.sub(r'~~(.*?)~~', r'<s>\1</s>', escaped_text)
    # Моноширинный (inline): `text` -> <code>text</code>
    escaped_text = re.sub(r'`(.*?)`', r'<code>\1</code>', escaped_text)
    
    # 3. Списки: преобразуем маркеры в •
    escaped_text = re.sub(r'^\s*[\*\-]\s+', '• ', escaped_text, flags=re.MULTILINE)

    # 4. Блоки кода: ```lang\ncode``` -> <pre><code class="language-lang">code</code></pre>
    def code_block_replacer(match):
        lang = match.group(1) or ""
        code = match.group(2)
        # lang может быть пустым, это нормально
        return f'<pre><code class="language-{lang}">{code}</code></pre>'
    
    escaped_text = re.sub(r'```(\w*)\n(.*?)\n```', code_block_replacer, escaped_text, flags=re.DOTALL)

    return escaped_text

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

    from handlers import authorized_users
    if message.from_user.id not in authorized_users:
        return

    text = message.text or message.caption or ""
    if not text: return

    replacements = {
        r"instagram\.com/": "ddinstagram.com/",
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

    user_name = html.escape(message.from_user.first_name)
    safe_text = html.escape(new_text)
    final_text = f"✅ <b>От {user_name}:</b>\n{safe_text}"

    try:
        await message.delete()
        await context.bot.send_message(
            chat_id=message.chat_id,
            text=final_text,
            parse_mode="HTML",
            message_thread_id=message.message_thread_id,
            link_preview_options=LinkPreviewOptions(is_disabled=False, prefer_large_media=True)
        )
    except Exception as e:
        print(f"⚠️ Ошибка LinkFixer: {e}")