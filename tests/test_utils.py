from InspectorAI.utils import format_to_html, get_model_short_name

def test_format_to_html():
    assert "<b>bold</b>" in format_to_html("**bold**")
    assert "<code>code</code>" in format_to_html("`code`")

def test_model_names():
    assert get_model_short_name("openai/gpt-4o:free", "openrouter") == "gpt-4o"
    assert get_model_short_name("models/gemini-1.5-pro", "gemini") == "gemini-1.5-pro"