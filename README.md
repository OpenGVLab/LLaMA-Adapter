# LLaMA-Adapter: Efficient Fine-tuning of LLaMA 🚀

Official implementation of ['LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention'](https://arxiv.org/abs/2303.16199).

This repo proposes **LLaMA-Adapter**, a lightweight adaption method for fine-tuning instruction-following [LLaMA](https://github.com/facebookresearch/llama) models 🔥, using 52K data provided by [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).



## Overview
Efficiency Comparison:
|  Model | Parameters | Storage Space | Training Time  
| :-----: | :-----: |:-----:| :-----: |
|  [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 7B |13G| 3 Hours|
|  LLaMA-Adapter | 1.2M |4.7M| 1 Hour|

By inserting adapters into LLaMA's transformer, our method only introduces **1.2M** learnable parameters, and turns a LLaMA into an instruction-following model within **1 hour**. For stablizing training at early stages, we propose a novel **Zero-init Attention** with zero gating mechanism to adaptively incorporate the instructional signals. After fine-tuning, LLaMA-Adapter can generate high-quality instruction-following sentences, comparable to the fully fine-tuned [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [Alpaca-Lora](https://github.com/tloen/alpaca-lora).

<div align="center">
  <img src="pipeline.png"/>
</div>

Our approach can be simply extended to **Multi-modal Input Instructions**. The reasoning framework of image-conditioned LLaMA-Adapter for [ScienceQA](https://scienceqa.github.io/) is as follows, which is also shared by other modalities, such as audio and video.

<div align="center">
  <img src="multimodal.png"/>
</div>

## News
* **TODO**: training code, multi-modal LLaMA-Adapter, adapters for larger-scale LLaMA models
* [Paper](https://arxiv.org/pdf/2303.16199.pdf) is available on arXiv 📌. 
* The generation code of LLaMA-Adapter based on 7B LLaMA has been released.
* 🔥 We are **hiring** interns, postdocs and full-time researchers in **General Vision Group, Shanghai AI Lab**, aiming at multi-modality and vision foundation models. If you are interested, please contact gaopeng@pjlab.org.cn.


## Setup

Here is a from-scratch script.
```bash
conda create -n llama_adapter -y
conda activate llama_adapter

# install pytorch
conda install pytorch cudatoolkit -c pytorch -y

# install dependency and llama-adapter
pip install -r requirements.txt
pip install -e .
```

## Inference

Please request access to the pre-trained LLaMA from [this form](https://forms.gle/jk851eBVbX1m5TAv5) (official) or download the LLaMA-7B from [Hugging Face](https://huggingface.co/nyanko7/LLaMA-7B/tree/main) (unofficial). Then, obtain the weights of our LLaMA-Adapter from [here](https://github.com/ZrrSkywalker/LLaMA-Adapter/releases/download/v.1.0.0/llama_adapter_len10_layer30_release.pth). We denote the path to the downloaded weights of LLaMA and adapters as `TARGET_FOLDER` and `ADAPTER_PATH`.

Here is an example to generate instruction-following sentences with 7B LLaMA model and our LLaMA-Adapter:
```bash
torchrun --nproc_per_node 1 example.py \
         --ckpt_dir $TARGET_FOLDER/model_size\
         --tokenizer_path $TARGET_FOLDER/tokenizer.model \
         --adapter_path $ADAPTER_PATH
```

## Training

We will release the training code soon. We use 52K instruction-following [data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) for training, and adopt the hyper-parameters as following:

| hyper-parameter | value |
|-----------------|-------|
| batch Size      | 64    |
| learning rate   | 9e-3  |
| warmup epochs   | 2     |
| epochs          | 5     |
| weight decay    | 0.02  |


## Comparison with Other Methods

### Instruction 1: 
Tell me about alpacas.

* **LLaMA-Adapter**: Alpacas are members of the camelid family and are native to the Andes Mountains of South America. They are typically found in herds of 10-20 animals and are known for their gentle nature and their ability to thrive in harsh climates. Alpacas are typically white or grey in color and have long, silky coats. They are also known for their ability to produce high-quality wool, which is used in the production of clothing and other products.

<details><summary>OTHER METHODS</summary>

* **Alpaca-LoRA**: Alpacas are members of the camelid family and are native to the Andes Mountains of South America. They are known for their soft, luxurious fleece, which is used to make clothing, blankets, and other items. Alpacas are herbivores and graze on grasses and other plants. They are social animals and live in herds of up to 20 individuals.

* **Stanford Alpaca**: Alpacas are small, fluffy animals related to camels and llamas. They are native to Peru and Bolivia, and were first domesticated around 5,000 years ago. They are kept mainly for their fine, soft fleece, which is used to make knitwear and other garments. Alpacas are herd animals and live in small family groups, led by an older male. They are highly social animals and can form strong bonds with their owners.

* **text-davinci-003**: Alpacas are a domesticated species of South American camelid, similar to llamas. They are native to the Andes Mountains in South America and are kept as livestock for their soft, luxurious wool. Alpacas are social animals and live in herds of up to 20 individuals. They are typically kept in small herds of two to five animals, and are relatively easy to care for. Alpacas are herbivores and feed on grass, hay, and other vegetation. They are also known for their gentle and friendly nature, making them popular as pets.

</details>

---

### Instruction 2: 
Tell me about the president of Mexico in 2019.

* **LLaMA-Adapter**: The president of Mexico in 2019 was Andrés Manuel López Obrador, who took office on December 1st, 2018. He is a member of the left-wing party Morena and is known for his populist policies. He has focused on tackling corruption, reducing poverty, and improving the country's infrastructure. He has also been a vocal critic of the United States' immigration policies.

<details><summary>OTHER METHODS</summary>

* **Alpaca-LoRA**: The president of Mexico in 2019 was Andrés Manuel López Obrador, who took office on December 1, 2018. He is a member of the National Regeneration Movement (MORENA) political party and is the first left-wing president of Mexico since 1946. He is known for his anti-corruption and anti-neoliberal policies, as well as his commitment to improving the living conditions of the Mexican people.

* **Stanford Alpaca**: In 2019, Andres Manuel Lopez Obrador was sworn in as President of Mexico. He is the country’s 58th president and the first from the left-wing Morena party. He was elected in 2018 on a promise to combat corruption and reduce poverty and has faced criticism for his handling of the economy, crime and the country’s relationship with the United States. He is also known for his ambitious social welfare programs and has been praised for raising the minimum wage and providing aid to low-income families. He is seen as a champion of democracy and human rights in Mexico.

* **text-davinci-003**: The president of Mexico in 2019 is Andrés Manuel López Obrador, commonly known as AMLO. He is a Mexican politician who has been the President of Mexico since December 1, 2018. He is a member of the National Regeneration Movement (MORENA) party and previously served as the Head of Government of Mexico City from 2000 to 2005. He is a left-wing populist and has promised to reduce corruption and inequality in Mexico. He has also promised to increase public spending on social programs and infrastructure projects.

</details>

---

### Instruction 3: 
Tell me about the king of France in 2019.

* **LLaMA-Adapter**: The king of France in 2019 is King Louis XIV, who has been in power since 2015. He is the 35th monarch of the House of Bourbon and is the longest-serving monarch in French history. He is known for his lavish lifestyle and his ambitious building projects, including the Palace of Versailles. He is also credited with establishing the Code Noir, which granted slaves in the French colonies more rights than they had previously enjoyed.

<details><summary>OTHER METHODS</summary>

* **Alpaca-LoRA**: The king of France in 2019 is Emmanuel Macron, who was elected in 2017. He is the youngest president in the history of France, and has been credited with bringing about a number of reforms, including a new tax system and a crackdown on corruption. He has also been praised for his efforts to strengthen the country's relationship with the European Union.

* **Stanford Alpaca**: In 2019, the King of France was Emmanuel Macron. He was sworn in as President of France on May 14, 2017, and has been leading the country ever since. He is a member of the centrist La République En Marche! Party. He is known for his reformist agenda and has faced a number of protests from both the right and left of the political spectrum. He is the youngest French president in history.

* **text-davinci-003**: The King of France is a largely symbolic role, as France is a republic. The current President of France is Emmanuel Macron, who was elected in 2017.

</details>

---

### Instruction 4: 
List all Canadian provinces in alphabetical order.

* **LLaMA-Adapter**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan, Yukon.

<details><summary>OTHER METHODS</summary>

* **Alpaca-LoRA**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan

* **Stanford Alpaca**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan, Northwest Territories, Nunavut, Yukon.

* **text-davinci-003**:

1. Alberta
2. British Columbia
3. Manitoba
4. New Brunswick
5. Newfoundland and Labrador
6. Northwest Territories
7. Nova Scotia
8. Nunavut
9. Ontario
10. Prince Edward Island
11. Quebec
12. Saskatchewan
13. Yukon

</details>

---

### Instruction 5: 
Write a Python program that prints the first 10 Fibonacci numbers.

* **LLaMA-Adapter**:
```python
def fibonacci():
    a = 0
    b = 1
    for i in range(10):
        a, b = b, a + b
        print(a, b)
    print("Fibonacci numbers: ", a, b)

fibonacci()
```

<details><summary>OTHER METHODS</summary>

* **Alpaca-LoRA**:

```python
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
```

* **Stanford Alpaca**:

```python
numbers = [0, 1]

for i in range(2, 11):
    numbers.append(numbers[i-2] + numbers[i-1])

print(numbers)
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

* **text-davinci-003**:

```python
# Fibonacci numbers are the numbers in the following integer sequence, called the Fibonacci sequence, and characterized by the fact that every number after the first two is the sum of the two preceding ones.

# Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

# Program:

# Initialize the first two numbers of the Fibonacci sequence
a = 0
b = 1

# Print the first 10 numbers of the Fibonacci sequence
print(a)
print(b)

for i in range(2,10):
    c = a + b
    a = b
    b = c
    print(c)
```

</details>

---

### Instruction 6: 
Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.

* **LLaMA-Adapter**:

```python
def print_numbers(n):
    for i in range(1, 101):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)
```

<details><summary>OTHER METHODS</summary>

* **Alpaca-LoRA**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

* **Stanford Alpaca**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print('FizzBuzz')
    elif i % 3 == 0:
        print('Fizz')
    elif i % 5 == 0:
        print('Buzz')
    else:
        print(i)
```

* **text-davinci-003**:

```javascript
for (let i = 1; i <= 100; i++) {
  if (i % 3 === 0 && i % 5 === 0) {
    console.log("FizzBuzz");
  } else if (i % 3 === 0) {
    console.log("Fizz");
  } else if (i % 5 === 0) {
    console.log("Buzz");
  } else {
    console.log(i);
  }
}
```

</details>

---

### Instruction 7: 
Tell me five words that rhyme with 'shock'.

* **LLaMA-Adapter**: Flock, rock, lock, cook, and stock.

<details><summary>OTHER METHODS</summary>

* **Alpaca-LoRA**: Flock, clock, dock, stock, and sock.

* **Stanford Alpaca**: Five words that rhyme with shock are: rock, pop, shock, cook, and snock.

* **text-davinci-003**:

1. Block
2. Lock
3. Dock
4. Clock
5. Rock

</details>

---

### Instruction 8: 
  Translate the sentence 'I have no mouth but I must scream' into Spanish.

* **LLaMA-Adapter**: No tengo boca pero tengo que gritar.

<details><summary>OTHER METHODS</summary>

* **Alpaca-LoRA**: No tengo boca pero tengo que gritar.

* **Stanford Alpaca**: No tengo boca, pero debo gritar.

* **text-davinci-003**: No tengo boca pero debo gritar.

</details>


## More interesting examples


### Instruction 1: 
Write an introduction for LLaMA-Adapter, a lightweight and simple adapter for large language model LLaMA.

* **LLaMA-Adapter**: LLaMA-Adapter is a lightweight and simple adapter for large language models (LLaMA). It is designed to be used as a drop-in replacement for the standard LLaMA adapter, and is designed to be faster, more efficient, and more flexible. It is capable of handling a variety of different input and output formats, and can be easily configured to adapt to different models and datasets.

---

### Instruction 2: 
  Write a Python object detection inference program with Faster R-CNN algorithm.

* **LLaMA-Adapter**:

```python
import numpy as np
import os
import glob
import cv2
import tensorflow as tf

# load dataset
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
dataset_path = glob.glob(dataset_dir + '/*.jpg')

# define model
model = tf.keras.models.load_model('model.h5')

# input images
images = []
for image in dataset_path:
    images.append(cv2.imread(image))

# create a numpy array
images_np = np.array(images)

# create a batch of images
batch_size = 1
batches = np.reshape(images_np, (batch_size, 1, 3, 32, 32))

# create a dataset
dataset = tf.data.Dataset.from_tensor_slices(batches)

# define model
model.fit(dataset, steps=100)

# detect objects
detected_objects = model.predict(images_np)

# print detected objects
for obj in detected_objects:
    print(obj)
```

---

### Instruction 3: 
  Compose the lyrics of a song in the style of Jay Chou.

* **LLaMA-Adapter**:
```
I'm a lonely soul who's searching for a home,
Looking for someone to make me feel at home.

I'm a lonely soul who's searching for a place,
Looking for someone to make me feel at peace.

I'm a lonely soul who's searching for a love,
Looking for someone to make me feel alive.

I'm a lonely soul who's searching for a friend,
Looking for someone to make me feel complete.

I'm a lonely soul who's searching for a dream,
Looking for someone to make me feel free.

I'm a lonely soul who's searching for a life,
Looking for someone to make me feel alive.
```

---

### Instruction 4: 
  Please help me arrange a one week travel route in Bangkok.

* **LLaMA-Adapter**: 
```
Day 1: Explore the Grand Palace and Wat Pho, visit Chinatown and Khao San Road.

Day 2: Take a boat ride along the Chao Phraya River, visit the National Museum and the Jim Thompson House.

Day 3: Explore the Bangkok National Museum and the National Gallery, visit the Temple of the Reclining Buddha and the Dusit Zoo.

Day 4: Visit the Wat Arun, the Royal Barges Museum, and the Siam Paragon Shopping Mall.

Day 5: Explore the Chatuchak Weekend Market, the MBK Center, and the Suan Lum Ratcha Tep.

Day 6: Visit the Floating Market, the Royal Palace, and the Khao San Road.
```

---

### Instruction 5: 
  Explain why this poem is good.

**Input**: Life, thin and light-off time and time again. Frivolous tireless. I heard the echo, from the valleys and the heart. Open to the lonely soul of sickle harvesting. Repeat outrightly, but also repeat the well-being of eventually swaying in the desert oasis. I believe I am. Born as the bright summer flowers.

* **LLaMA-Adapter**: This poem is good because it uses simple language to convey a powerful message. It explores the themes of loneliness and isolation, and captures the feelings of alienation and despair that many people experience. The poem also uses imagery to create a vivid picture of the emotions being described.


## Contributors
[Peng Gao](https://scholar.google.com/citations?user=_go6DPsAAAAJ&hl=zh-CN), [Jiaming Han](https://csuhan.com), [Renrui Zhang](https://scholar.google.com/citations?user=YlL3xN4AAAAJ&hl=zh-CN)

## Citation
If you find our LLaMA-Adapter code and paper useful, please kindly cite:
```bash
@article{llamaadapter2023,
  title = {LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention},
  author={Zhang, Renrui and Han, Jiaming and Zhou, Aojun and Hu, Xiangfei and Yan, Shilin and Lu, Pan and Li, Hongsheng and Gao, Peng and Qiao Yu},
  journal={arXiv preprint arXiv:2303.16199},
  year={2023}
}
```

## Acknowledgement
This repo benefits from [LLaMA](https://github.com/facebookresearch/llama), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), and [Alpaca-Lora](https://github.com/tloen/alpaca-lora). Thanks for their wonderful works.
