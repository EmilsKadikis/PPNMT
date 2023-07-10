import os
from tqdm import tqdm
import random
from transformers import MarianTokenizer, GenerationConfig
from perturbable_marianmt_model import PerturbableMarianMTModel
from bag_of_words_processor import get_bag_of_words_vectors
from perturb_past import PerturbationArgs

def _get_bags_of_words(hyperparameters, device, tokenizer):
    positive_bags_of_words = hyperparameters.pop("bag_of_words", None)
    if type(positive_bags_of_words) is str:
        positive_bags_of_words_paths = positive_bags_of_words.split(";")
        positive_bags_of_words = None
    else:
        positive_bags_of_words_paths = None

    negative_bags_of_words = hyperparameters.pop("negative_bag_of_words", None)
    if type(negative_bags_of_words) is str:
        negative_bags_of_words_paths = negative_bags_of_words.split(";")
        negative_bags_of_words = None
    else:
        negative_bags_of_words_paths = None

    # set up perturbation args
    positive_bow = get_bag_of_words_vectors(
        tokenizer, 
        bag_of_words=positive_bags_of_words, 
        bag_of_words_paths=positive_bags_of_words_paths, 
        device=device
    ) if positive_bags_of_words is not None or positive_bags_of_words_paths is not None else None

    negative_bow = get_bag_of_words_vectors(
        tokenizer, 
        bag_of_words=negative_bags_of_words, 
        bag_of_words_paths=negative_bags_of_words_paths, 
        device=device
    ) if negative_bags_of_words is not None or negative_bags_of_words_paths is not None else None
    
    return positive_bow,negative_bow

def chunk(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def make_adapted_predictions(source_texts, hyperparameters, device="cpu"):
    print(f"Using device {device}")
    pretrained_model = hyperparameters["translation_model"]
    model = PerturbableMarianMTModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = MarianTokenizer.from_pretrained(pretrained_model)

    positive_bow, negative_bow = _get_bags_of_words(hyperparameters, device, tokenizer)
    args = PerturbationArgs(
        positive_bag_of_words=positive_bow,
        negative_bag_of_words=negative_bow,
        **hyperparameters
    )

    max_length = hyperparameters.pop("length", 100)
    batch_size = hyperparameters.pop("batch_size", 50)
    # top_k = hyperparameters.pop("top_k", 5)

    # sort source_texts by length for more efficient generation
    source_texts = [(i, text) for i, text in enumerate(source_texts)]
    source_texts.sort(key=lambda x: len(x[1]))

    predictions = []
    batches = list(chunk(source_texts, batch_size))
    for texts in tqdm(batches):
        indices, texts = zip(*texts)
        encoded_texts = tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = encoded_texts.input_ids.to(device) # [batch_size, max_seq_len]
        attention_mask = encoded_texts.attention_mask.to(device) # [batch_size, max_seq_len]
        generation_config = GenerationConfig(num_beams=1, do_sample=False, max_new_tokens=max_length-1)

        results = model.generate(args, input_ids, attention_mask=attention_mask, generation_config=generation_config)
        decoded_results = tokenizer.batch_decode(results, skip_special_tokens=True)

        decoded_results = [(i, text) for i, text in zip(indices, decoded_results)]
        predictions.extend(decoded_results)
        print(decoded_results)
    
    # sort predictions back into original order
    predictions.sort(key=lambda x: x[0])
    predictions = [text for _, text in predictions]
    return predictions