import torch
import numpy as np
from transformers import MarianMTModel, MarianTokenizer, GenerationConfig

pretrained_model = "Helsinki-NLP/opus-mt-en-de"
seed = None
text = "This table is large."
text = "This data table is large."
text = "This database table is large."
text = "This table is large and holds a lot of documents."
text = "This table is large and holds a lot of dishes."
text = "This table is large and holds a lot of pictures."
text = "This table is large and holds a lot of images."
text = "This table is 5 gigabytes."
text = "5 gigabytes is the size of the table."
text = "This table is 2 times bigger than the user table."
text = "The user table is very big."
text = "The kitchen table is very big."
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

config = GenerationConfig(num_beams=1, do_sample=False)
result = model.generate(input_tensor, generation_config=config)
print(tokenizer.decode(result[0]))

