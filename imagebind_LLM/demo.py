import ImageBind.data as data
import llama

llama_dir = "/path/to/LLaMA/"

model = llama.load("checkpoints/7B-epoch0.pth", llama_dir)
input = data.load_and_transform_vision_data(["example_imgs/bad_tree.jpg"], device='cuda')
results = model.generate(input, [llama.format_prompt("Describe the image")], input_type="vision")
result = results[0].strip()
print(result)