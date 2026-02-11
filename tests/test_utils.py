from InspectorAI.utils import safe_format_to_html, get_model_short_name


def test_safe_format_to_html():
    # Проверяем, что жирный текст корректно конвертируется в MarkdownV2 (*)
    # В MarkdownV2 жирный это *текст*, а не <b>
    formatted = safe_format_to_html("**bold**")
    assert "*" in formatted
    assert "bold" in formatted

    # Проверяем моноширинный код
    formatted_code = safe_format_to_html("`code`")
    assert "`" in formatted_code
    assert "code" in formatted_code


def test_model_names():
    # Проверка для OpenRouter (убирает префикс автора и :free)
    assert get_model_short_name("openai/gpt-4o:free", "openrouter") == "gpt-4o"

    # Проверка для Gemini (убирает префикс models/)
    assert get_model_short_name("models/gemini-2.5-flash", "gemini") == "gemini-2.5-flash"
    assert get_model_short_name("gemini-2.0-flash", "gemini") == "gemini-2.0-flash"