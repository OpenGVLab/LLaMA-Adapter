import ImageBind.data as data
import llama


llama_dir = "/path/to/LLaMA"

model = llama.load("7B-beta", llama_dir, knn=True)
model.eval()

inputs = {}
image = data.load_and_transform_vision_data(["examples/girl.jpg"], device='cuda')
inputs['Image'] = [image, 1]
audio = data.load_and_transform_audio_data(['examples/girl_bgm.wav'], device='cuda')
inputs['Audio'] = [audio, 1]

results = model.generate(
    inputs,
    [llama.format_prompt("Guess the girl's mood based on the background music and explain the reason?")],
    max_gen_len=256
)
result = results[0].strip()
print(result)