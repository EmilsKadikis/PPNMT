import torch
import numpy as np
from transformers import MarianMTModel, MarianTokenizer, GenerationConfig
from tqdm import tqdm

def make_predictions(source_texts, output_file_name="predictions.txt", model_name="Helsinki-NLP/opus-mt-en-de", device="cpu"):
    # set the device

    model = MarianMTModel.from_pretrained(
        model_name,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    predictions = []
    for text in tqdm(source_texts):
        input_tensor = tokenizer(text, return_tensors="pt").input_ids.to(device)
        config = GenerationConfig(num_beams=1, do_sample=False)
        prediction = model.generate(input_tensor, generation_config=config)
        prediction = tokenizer.decode(prediction[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predictions.append(prediction)

    # write predictions to file
    with open(output_file_name, "w") as f:
        for prediction in predictions:
            f.write(prediction + "\n")
    
    return predictions