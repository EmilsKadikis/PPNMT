from transformers import MarianMTModel, MarianTokenizer, GenerationConfig
from typing import List
from perturbable_marianmt_model import PerturbableMarianMTModel
from bag_of_words_processor import get_bag_of_words_vectors

def translate(pretrained_model, texts: List[str], positive_bags_of_words: List[str], device='cpu'):
    model = PerturbableMarianMTModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = MarianTokenizer.from_pretrained(pretrained_model)
    positive_bow = get_bag_of_words_vectors(positive_bags_of_words, tokenizer, device)
    model.set_positive_bag_of_words(positive_bow)

    encoded_texts = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = encoded_texts.input_ids.to(device) # [batch_size, max_seq_len]
    attention_mask = encoded_texts.attention_mask.to(device) # [batch_size, max_seq_len]

    generation_config = GenerationConfig(num_beams=1, do_sample=False, max_new_tokens=100)
    results = model.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config)
    
    decoded_results = tokenizer.batch_decode(results, skip_special_tokens=True)
    print(decoded_results)

if __name__ == '__main__':
    texts = ["Who is your favourite actor?", "I got a hundred colours in your city.", "Yeah, exactly. Okay now also have a look on my fashion stores where do you shop?"]
    positive_bows = ['rewrite/test_bow']
    translate("Helsinki-NLP/opus-mt-en-de", texts, positive_bows, device='cpu')
    # ['Wer ist dein Lieblingsdarsteller?', 'Ich habe hundert Farben in deiner Stadt.', 'Okay, jetzt schau auch in meinen Modegeschäften, wo kaufst du ein?']