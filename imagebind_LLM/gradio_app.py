import gradio as gr
import argparse

import ImageBind.data as data
import llama

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="checkpoints/7B.pth", type=str,
    help="Name of or path to LLaMAAdapter pretrained checkpoint",
)
args = parser.parse_args()

llama_dir = "/path/to/LLaMA/"
# model = llama.load("7B", llama_dir)
model = llama.load(args.model, llama_dir)
model.half()

def caption_generate(
    img: str,
    text: str,
    audio: str,
    video: str,
    prompt: str,
    max_gen_len=64,
    temperature: float = 0.1,
    top_p: float = 0.75,
):
    input = None
    input_type = None
    try:
        input = data.load_and_transform_vision_data([img], device='cuda')
        input_type = 'vision'
        print('image', input.shape)
    except:
        pass
    
    if text is not None and len(text) > 0:
        try:
            input = data.load_and_transform_text([text], device='cuda')
            input_type = 'text'
            print(input_type, input.shape)
        except:
            pass

    try:
        input = data.load_and_transform_audio_data([audio], device='cuda')
        input_type = 'audio'
        print(input_type, input.shape)
    except:
        pass

    try:
        input = data.load_and_transform_video_data([video], device='cuda')
        input_type = 'vision'
        print('video', input.shape)
    except:
        pass

    prompts = [llama.format_prompt(prompt)]

    results = model.generate(input, prompts, input_type, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
    result = results[0].strip()
    print(result)
    return result

def create_caption_demo():
    with gr.Blocks() as instruct_demo:
        with gr.Row():
            with gr.Column():
                img = gr.Image(label='Image', type='filepath')
                text = gr.Textbox(label='Text')
                audio = gr.Audio(label='Audio', type='filepath')
                video = gr.Video(label='Video', type='filepath')
                question = gr.Textbox(lines=2, label="Prompt")
                max_len = gr.Slider(minimum=1, maximum=512,
                                    value=128, label="Max length")
                with gr.Accordion(label='Advanced options', open=False):
                    temp = gr.Slider(minimum=0, maximum=1,
                                     value=0.1, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1,
                                      value=0.75, label="Top p")

                run_botton = gr.Button("Run")

            with gr.Column():
                outputs = gr.Textbox(lines=10, label="Output")

        inputs = [img, text, audio, video, question, max_len, temp, top_p]

        examples = [
            ["example_imgs/funny-photo.jpg", None, None, None, "Explain why this image is funny", 128, 0.1, 0.75],
            [None, None, None, "example_imgs/hitting_baseball.mp4", "Explain why this video is beautiful", 128, 0.1, 0.75],
            [None, None, "ImageBind/.assets/bird_audio.wav", None, "Describe this scene", 128, 0.1, 0.75],
        ]

        gr.Examples(
            examples=examples,
            inputs=inputs,
            outputs=outputs,
            fn=caption_generate,
            cache_examples=False
        )
        run_botton.click(fn=caption_generate, inputs=inputs, outputs=outputs)
    return instruct_demo


description = f"""
# ImageBind-LLM🚀
"""

with gr.Blocks(css="h1,p {text-align: center;}") as demo:
    gr.Markdown(description)
    with gr.TabItem("Multi-Modal Interaction"):
        create_caption_demo()

demo.queue(api_open=True, concurrency_count=1).launch(share=True)
