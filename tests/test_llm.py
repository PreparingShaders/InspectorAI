import pytest
from unittest.mock import AsyncMock, MagicMock
from InspectorAI.llm_service import process_llm

@pytest.mark.asyncio
async def test_llm_fallback(mocker):
    context = MagicMock()
    # Все методы, которые вызываются через await, ДОЛЖНЫ быть AsyncMock
    context.bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    context.bot.edit_message_text = AsyncMock()

    update = MagicMock()
    # Добавь это, чтобы не было ошибок доступа к атрибутам
    update.effective_chat.id = 12345
    mocker.patch('InspectorAI.llm_service.or_client.chat.completions.create',
                 side_effect=Exception("API Error"))

    # Мокаем успех Gemini
    mock_gemini = mocker.patch('InspectorAI.llm_service.gemini_client.models.generate_content')
    mock_gemini.return_value.text = "Ответ от Gemini"

    await process_llm(update, context, "Привет", mode="chat")

    # Проверяем, что бот переключился на Gemini после ошибки
    assert mock_gemini.called