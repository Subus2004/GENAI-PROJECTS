import torch
import gradio as gr
from transformers import pipeline

# Load Hugging Face model directly
model_path = "sshleifer/distilbart-cnn-12-6"

text_summary = pipeline(
    "summarization",
    model=model_path,
    torch_dtype=torch.bfloat16
)

def summary(input_text):
    output = text_summary(input_text)
    return output[0]["summary_text"]

gr.close_all()

demo = gr.Interface(
    fn=summary,
    inputs=[gr.Textbox(label="Input text to summarize", lines=6)],
    outputs=[gr.Textbox(label="Summarized text", lines=4)],
    title="GENAI Project 1 : TEXT_SUMMARIZER",
    description="This application helps summarize long text into a shorter version."
)

demo.launch()
