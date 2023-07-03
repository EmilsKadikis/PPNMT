from typing import List
import torch
import os

def get_bag_of_words_vectors(bag_of_words_paths: List[str], tokenizer, device='cpu'):
    bags_of_words = _load_bags_of_words(bag_of_words_paths)
    # flatten bags_of_words
    bag_of_words = [word for bag in bags_of_words for word in bag]
    bow_indices = [tokenizer.encode(word.strip(), add_special_tokens=False) for word in bag_of_words]
    return _build_bows_one_hot_vectors(bow_indices, tokenizer, device)

""" Gets the given bags of words, tokenizes each word in it and returns a list of their indices. """
def _load_bags_of_words(bag_of_words_paths: List[str]) -> List[List[List[int]]]:
    words = []
    for path in bag_of_words_paths:
        filepath = './' + path
        if not os.path.exists(filepath):
            filepath = filepath + '.txt'
        with open(filepath, "r") as f:
            words.append(f.read().strip().split("\n"))
    return words


""" Builds a one-hot vector for each word in the bag of words. """
def _build_bows_one_hot_vectors(bow_indices, tokenizer, device='cpu'):
    if bow_indices is None:
        return None

    one_hot_vectors = []
    for word in bow_indices:
        one_hot = torch.zeros(tokenizer.vocab_size).to(device)
        one_hot.scatter_(0, torch.tensor(word), 1)
        one_hot = one_hot / len(word)
        one_hot_vectors.append(one_hot)
    return one_hot_vectors

if __name__ == '__main__':
    from transformers import MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
    bows = get_bag_of_words_vectors(['rewrite/test_bow', 'rewrite/test_bow2'], tokenizer)
    print(bows)