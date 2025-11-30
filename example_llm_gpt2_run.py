import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict
from hooks import gpt_manual_model_summary
from multiprocessing import Queue
from renderer import start_render
import random
import string


def generate_random_string(length):
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string

MODEL_NAME = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.to(device)
model.eval()


def run_inference(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        print('Running Model')
        _ = model(**inputs)
        render_queue.put(summary.copy())
        summary.clear()


if __name__ == "__main__":
    visualizer = {'visual': 'linear',
                  'camera_distance': 4}
    # visualizer = {'visual': 'ring',
    #               'camera_distance': 8}

    render_queue = Queue()
    p = start_render(render_queue, visualizer)

    summary = OrderedDict()
    inputs = tokenizer("Hello World", return_tensors="pt").to(device)
    gpt_manual_model_summary(model, inputs, summary, render_queue)

    while True:
        user_input = generate_random_string(random.randint(10, 100))
        if user_input.strip().lower() == "exit":
            break
        run_inference(user_input)
