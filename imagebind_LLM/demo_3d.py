import ImageBind.data as data
import llama


llama_dir = "/path/to/LLaMA"

model = llama.load("7B", llama_dir, knn=True)
model.eval()

inputs = {}
point = data.load_and_transform_point_cloud_data(["examples/airplane.pt"], device='cuda')
inputs['Point'] = [point, 1]

results = model.generate(
    inputs,
    [llama.format_prompt("Describe the 3D object in detail.")],
    max_gen_len=256
)
result = results[0].strip()
print(result)
