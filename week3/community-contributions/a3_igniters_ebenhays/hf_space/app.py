import os
import tempfile
import gradio as gr
from openai import OpenAI
from langdetect import detect, DetectorFactory
from gtts import gTTS

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)

MODEL = "anthropic/claude-3.5-sonnet"
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024
MAX_WORD_COUNT = 500

SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi", "ru": "Russian"
}
LANGUAGE_TO_CODE = {v: k for k, v in SUPPORTED_LANGUAGES.items()}

SYSTEM_PROMPT = """You are a multilingual text summarizer and translator.
Given text in any language, provide a concise summary in the specified target language.
Keep the summary clear, informative, and under 150 words.
Always respond with ONLY the summary text, no additional commentary or labels."""


def validate_input(text_input=None, file_input=None):
    """Validate and extract text from either raw input or uploaded file."""
    if file_input is not None:
        try:
            if os.path.getsize(file_input) > MAX_FILE_SIZE_BYTES:
                return "", f"File exceeds 1MB limit."
            with open(file_input, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            return "", "Please upload a valid UTF-8 encoded text file."
        except Exception as e:
            return "", f"Error reading file: {e}"
    elif text_input and text_input.strip():
        text = text_input.strip()
    else:
        return "", "Please provide text input or upload a file."

    word_count = len(text.split())
    if word_count > MAX_WORD_COUNT:
        return "", f"Text has {word_count} words, exceeding the {MAX_WORD_COUNT} word limit."
    if word_count < 50:
        return "", "Text is too short. Please provide at least 50 words."

    return text, ""


def detect_language(text):
    """Detect the language of the input text."""
    DetectorFactory.seed = 0
    try:
        code = detect(text)
        name = SUPPORTED_LANGUAGES.get(code, f"Unknown ({code})")
        return code, name
    except Exception:
        return "en", "English (detection failed, defaulting)"


def summarize_text(text, target_language):
    """Stream a summary of the text in the target language."""
    user_message = f"Please summarize the following text in {target_language}:\n\n{text}"
    stream = openrouter.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        stream=True
    )
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response


def text_to_speech(text, lang_code):
    """Convert text to an MP3 audio file using gTTS."""
    gtts_lang_map = {"zh-cn": "zh-CN", "zh-tw": "zh-TW"}
    code = gtts_lang_map.get(lang_code, lang_code)
    try:
        tts = gTTS(text=text, lang=code, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"TTS error: {e}")
        return None


def process_text(text_input, file_input, target_language, enable_voice):
    """Orchestrate validation, summarization, and optional TTS."""
    if not openrouter_api_key:
        yield "Error: OPENROUTER_API_KEY secret is not set.", "", None, gr.update(visible=False), gr.update(interactive=True, value="Summarize")
        return

    text, error = validate_input(text_input, file_input)
    if error:
        yield error, "", None, gr.update(visible=False), gr.update(interactive=True, value="Summarize")
        return

    lang_code, lang_name = detect_language(text)
    detected_display = f"Detected: {lang_name} ({lang_code})"
    target_code = LANGUAGE_TO_CODE.get(target_language, "en")

    summary = ""
    for partial in summarize_text(text, target_language):
        summary = partial
        yield detected_display, summary, None, gr.update(visible=False), gr.update(interactive=False, value="Summarizing...")

    if enable_voice and summary:
        yield detected_display, summary, None, gr.update(visible=False), gr.update(interactive=False, value="Generating voice...")
        audio_path = text_to_speech(summary, target_code)
        yield detected_display, summary, audio_path, gr.update(visible=True, autoplay=True), gr.update(interactive=False, value="Playing audio...")
    else:
        yield detected_display, summary, None, gr.update(visible=False), gr.update(interactive=True, value="Summarize")


def on_audio_end():
    return gr.update(interactive=True, value="Summarize")


def on_file_upload(file):
    return "" if file is not None else gr.update()


with gr.Blocks(title="Multilingual Text Summarizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Multilingual Text Summarizer with Voice Output")
    gr.Markdown("Paste text or upload a `.txt` file to get a summary in your chosen language. Optionally listen to it read aloud.")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Enter text (max 500 words)", placeholder="Paste your text here...", lines=10)
            file_input = gr.File(label="Or upload a text file (max 1MB)", file_types=[".txt"], type="filepath")
            target_lang = gr.Dropdown(choices=list(SUPPORTED_LANGUAGES.values()), label="Summarize to language", value="English")
            voice_checkbox = gr.Checkbox(label="Read summary aloud", value=False)
            submit_btn = gr.Button("Summarize", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            detected_lang = gr.Textbox(label="Source Language", interactive=False)
            summary_output = gr.Markdown(label="Summary")
            audio_output = gr.Audio(label="Audio Summary", visible=False, autoplay=True)

    file_input.change(fn=on_file_upload, inputs=[file_input], outputs=[text_input])

    submit_btn.click(
        fn=process_text,
        inputs=[text_input, file_input, target_lang, voice_checkbox],
        outputs=[detected_lang, summary_output, audio_output, audio_output, submit_btn]
    )

    audio_output.stop(fn=on_audio_end, inputs=[], outputs=[submit_btn])

    clear_btn.click(
        fn=lambda: ("", None, "English", False, "", "", None, gr.update(interactive=True, value="Summarize")),
        inputs=[],
        outputs=[text_input, file_input, target_lang, voice_checkbox, detected_lang, summary_output, audio_output, submit_btn]
    )

demo.launch()
