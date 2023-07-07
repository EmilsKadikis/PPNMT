from transformers import MarianTokenizer, GenerationConfig
import time
from typing import List
from perturbable_marianmt_model import PerturbableMarianMTModel
from bag_of_words_processor import get_bag_of_words_vectors
from perturb_past import PerturbationArgs

def translate(pretrained_model, 
              texts: List[str], 
              positive_bags_of_words_paths: List[str] = None, 
              negative_bags_of_words_paths: List[str] = None, 
              positive_bags_of_words: List[str] = None, 
              negative_bags_of_words: List[str] = None, 
              device='cpu'):
    model = PerturbableMarianMTModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = MarianTokenizer.from_pretrained(pretrained_model)

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

    args = PerturbationArgs(
        positive_bag_of_words=positive_bow,
        negative_bag_of_words=negative_bow,
    )

    start = time.time()
    encoded_texts = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = encoded_texts.input_ids.to(device) # [batch_size, max_seq_len]
    attention_mask = encoded_texts.attention_mask.to(device) # [batch_size, max_seq_len]

    generation_config = GenerationConfig(num_beams=1, do_sample=False, max_new_tokens=19)
    results = model.generate(args, inputs=input_ids, attention_mask=attention_mask, generation_config=generation_config)
    end = time.time()
    print("Time taken: ", end - start)
    
    decoded_results = tokenizer.batch_decode(results, skip_special_tokens=True)
    print(decoded_results)

if __name__ == '__main__':
    texts = ["Who is your favourite actor?", "I got a hundred colours in your city.", "Yeah, exactly. Okay now also have a look on my fashion stores where do you shop?"]
    positive_bows = ['test_bow']
    negative_bows = ['test_bow2']
    translate("Helsinki-NLP/opus-mt-en-de", texts, positive_bows, negative_bows, device='mps')
    # ['Wer ist dein Lieblingsdarsteller?', 'Ich habe hundert Farben in deiner Stadt.', 'Okay, jetzt schau auch in meinen Modegeschäften, wo kaufst du ein?']