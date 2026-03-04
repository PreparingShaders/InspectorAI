import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock

# 1. Настройка путей (чтобы видел config.py и llm_service.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Импорт БЕЗ приставки InspectorAI, так как мы добавили папку в sys.path
from llm_service import process_llm, current_free_or_models

@pytest.mark.asyncio
async def test_llm_fallback(mocker):
    context = MagicMock()
    context.bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    context.bot.edit_message_text = AsyncMock()

    update = MagicMock()
    update.effective_chat.id = 12345

    # Подготовка очереди
    current_free_or_models.clear()
    current_free_or_models.append("gemini-2.0-flash")

    # 3. МОКИ (убираем приставку InspectorAI. из путей)
    mocker.patch('llm_service.or_client.chat.completions.create',
                 side_effect=Exception("API Error"))

    mock_gemini = mocker.patch('llm_service.gemini_client.models.generate_content')
    mock_gemini.return_value.text = "Ответ от Gemini"

    # 4. ЗАПУСК
    await process_llm(
        update,
        context,
        "Привет",
        selected_model="arcee-ai/trinity-large-preview:free",
        selected_provider="openrouter",
        mode="chat"
    )

    assert mock_gemini.called