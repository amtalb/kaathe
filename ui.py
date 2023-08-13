import random
import gradio as gr

from model import QAModel

model = QAModel()


def chat(query, history):
    return model.ask(query, history)


demo = gr.ChatInterface(chat)

demo.launch()
