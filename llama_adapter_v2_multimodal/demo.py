import cv2
import llama
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "/path/to/LLaMA/"

# choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
model, preprocess = llama.load("BIAS-7B", llama_dir, device)
model.eval()

prompt = llama.format_prompt('Please introduce this painting.')
img = Image.fromarray(cv2.imread("../docs/logo_v1.png"))
img = preprocess(img).unsqueeze(0).to(device)

result = model.generate(img, [prompt])[0]

print(result)
