import cv2
import gradio as gr
import torch
from PIL import Image

import llama


device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "/path/to/LLaMA/"

model, preprocess = llama.load("BIAS-7B", llama_dir, device)
model.half()
model.eval()

def multi_modal_generate(
    img_path: str,
    prompt: str,
    max_gen_len=256,
    temperature: float = 0.1,
    top_p: float = 0.75,
):
    try:
        img = Image.fromarray(cv2.imread(img_path))
    except:
        return ""

    img = preprocess(img).unsqueeze(0).half().to(device)
    prompt = llama.format_prompt(prompt)

    result = model.generate(img, [prompt], 
                            max_gen_len=max_gen_len, 
                            temperature=temperature, 
                            top_p=top_p)
    print(result[0])
    return result[0]


def create_multi_modal_demo():
    with gr.Blocks() as instruct_demo:
        with gr.Row():
            with gr.Column():
                img = gr.Image(label='Input', type='filepath')
                question = gr.Textbox(lines=2, label="Prompt")
                max_len = gr.Slider(minimum=1, maximum=512,
                                    value=256, label="Max length")
                with gr.Accordion(label='Advanced options', open=False):
                    temp = gr.Slider(minimum=0, maximum=1,
                                     value=0.1, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1,
                                      value=0.75, label="Top p")

                run_botton = gr.Button("Run")

            with gr.Column():
                outputs = gr.Textbox(lines=10, label="Output")

        inputs = [img, question, max_len, temp, top_p]

        examples = [
            ["../docs/logo_v1.png", "Please introduce this painting.", 256, 0.1, 0.75],
        ]

        gr.Examples(
            examples=examples,
            inputs=inputs,
            outputs=outputs,
            fn=multi_modal_generate,
            cache_examples=False
        )
        run_botton.click(fn=multi_modal_generate,
                         inputs=inputs, outputs=outputs)
    return instruct_demo


description = """
# LLaMA-Adapter V2🚀
The official demo for **LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model**.

Please refer to our [arXiv paper](https://arxiv.org/abs/2304.15010) and [github](https://github.com/ZrrSkywalker/LLaMA-Adapter) for more details.

The demo for **LLaMA-Adapter V1** is available at: [Huggingface Spaces](https://huggingface.co/spaces/csuhan/LLaMA-Adapter).
"""

with gr.Blocks(css="h1,p {text-align: center;}") as demo:
    gr.Markdown(description)
    with gr.TabItem("Multi-Modal Interaction"):
        create_multi_modal_demo()

demo.queue(api_open=True, concurrency_count=1).launch(share=True)
