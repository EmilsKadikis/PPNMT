import torch
import numpy as np
from transformers import MarianMTModel, MarianTokenizer, GenerationConfig

pretrained_model = "Helsinki-NLP/opus-mt-en-de"
seed = None
text = "The recipe table is very big."

if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)

# set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

model = MarianMTModel.from_pretrained(
    pretrained_model,
    output_hidden_states=True
)
model.to(device)
model.eval()

# load tokenizer
tokenizer = MarianTokenizer.from_pretrained(pretrained_model)

input_tensor = tokenizer(text, return_tensors="pt").input_ids.to(device)


print("Greedy decoding:")
config = GenerationConfig(num_beams=1, do_sample=False)
for i in range(10):
    result = model.generate(input_tensor, generation_config=config)
    print(tokenizer.decode(result[0]))

print("Beam search:")
config = GenerationConfig(num_beams=5, do_sample=False)
for i in range(5):
    result = model.generate(input_tensor, generation_config=config)
    print(tokenizer.decode(result[0]))

print("Beam search with sampling:")
config = GenerationConfig(num_beams=5, do_sample=True)
for i in range(5):
    result = model.generate(input_tensor, generation_config=config)
    print(tokenizer.decode(result[0]))

print("Contrastive search:")
config = GenerationConfig(top_k=5, penalty_alpha=0.5)
for i in range(5):
    result = model.generate(input_tensor, generation_config=config)
    print(tokenizer.decode(result[0]))

