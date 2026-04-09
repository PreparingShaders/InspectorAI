import os
import re
import asyncio
import html
from faster_whisper import WhisperModel
from telegram import LinkPreviewOptions
from telegram.ext import ApplicationHandlerStop
from handlers.state import authorized_users
import telegram

# Инициализируем модель Whisper
model_whisper = WhisperModel("base", device="cpu", compute_type="int8")

def to_html(text: str) -> str:
    """
    Преобразует Markdown от LLM в безопасный HTML для Telegram,
    правильно обрабатывая вложенность и экранирование.
    """
    if not text:
        return ""

    # Обрабатываем блочные элементы, такие как код, чтобы их содержимое не форматировалось
    code_blocks = {}
    def extract_code_blocks(m):
        key = f"__CODE_BLOCK_{len(code_blocks)}__"
        lang = html.escape(m.group(1) or "")
        code = html.escape(m.group(2))
        code_blocks[key] = f'<pre><code class="language-{lang}">{code}</code></pre>'
        return key

    processed_text = re.sub(r'```(\w*)\n(.*?)\n```', extract_code_blocks, text, flags=re.DOTALL)

    # Экранируем весь остальной текст, чтобы предотвратить HTML-инъекции
    processed_text = html.escape(processed_text)

    # Теперь применяем inline-форматирование к уже экранированному тексту.
    # Теги <b>, <i> и т.д. не будут экранированы, т.к. мы их добавляем после.
    replacements = {
        r'\*\*(.*?)\*\*': r'<b>\1</b>',
        r'\*(.*?)\*': r'<i>\1</i>',
        r'~~(.*?)~~': r'<s>\1</s>',
        r'`(.*?)`': r'<code>\1</code>',
    }
    for pattern, replacement in replacements.items():
        processed_text = re.sub(pattern, replacement, processed_text)

    # Списки (должны идти после inline, чтобы не конфликтовать с `*`)
    processed_text = re.sub(r'^\s*•\s+', '• ', processed_text, flags=re.MULTILINE) # Восстанавливаем, если было экранировано
    processed_text = re.sub(r'^\s*[\*\-]\s+', '• ', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'^\s*\d+\.\s+', '• ', processed_text, flags=re.MULTILINE)

    # Возвращаем на место блоки кода
    for key, block in code_blocks.items():
        processed_text = processed_text.replace(html.escape(key), block)

    return processed_text

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


    if message.from_user.id not in authorized_users:
        return

    text = message.text or message.caption or ""
    if not text: return

    replacements = {
        r"instagram\.com/": "kksave.com/",
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
    
    raise ApplicationHandlerStop
