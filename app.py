"""Gradio interface for the notebook's Swe-to-English translator."""

import spaces  # Must be imported FIRST
import gradio as gr
from pathlib import Path

from model import TranslationEngine


ROOT = Path(__file__).resolve().parent
engine = TranslationEngine(
    checkpoint_path=ROOT / "pico_model.pth",
    vocab_path=ROOT / "pico_translator.vocab",
)

@spaces.GPU(duration=60)
def translate_swedish(sentence: str) -> str:
    try:
        return engine.translate(sentence)
    except ValueError as error:
        raise gr.Error(str(error)) from error


with gr.Blocks(title="Swedish → English | Character Transformer") as demo:
    gr.Markdown(
        "# 🇸🇪 Swedish → English 🇬🇧\n"
        "Character-level Transformer trained for the Language Translation. "
    )
    with gr.Row():
        swedish = gr.Textbox(
            label="Swedish sentence",
            placeholder="Tack så mycket!",
            lines=3,
            max_lines=5,
        )
        english = gr.Textbox(
            label="English translation", lines=3, interactive=False
        )
    translate_button = gr.Button("Translate", variant="primary")
    gr.Examples(
        examples=[
        ["Tack så mycket!"],
        ["Jag älskar dig."],
        ["Jag förstår inte."],
        ["Kan du hjälpa mig?"],
        ["Var ligger tågstationen?"],
        ["Jag talar lite svenska."],
        ["Vad kostar det här?"],
        ["Jag vill ha en kopp kaffe."],
        ["Idag är vädret väldigt fint."],
        ["Det regnar mycket idag."],
        ["Jag bor i Sverige."],
        ["Jag tycker om att läsa böcker."],
        ["Vi ska åka till Stockholm imorgon."],
        ["Hon arbetar på ett stort företag."],
        ["Jag lär mig svenska eftersom jag bor i Sverige."],
        ],
        inputs=swedish,
    )
    gr.Markdown(
        f"**Limit:** {engine.max_characters} Swedish characters per request. "
        
    )
    translate_button.click(
        fn=translate_swedish, inputs=swedish, outputs=english
    )
    swedish.submit(fn=translate_swedish, inputs=swedish, outputs=english)

# Serial execution avoids several expensive autoregressive decodes at once on CPU.
demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == "__main__":
    demo.launch()
