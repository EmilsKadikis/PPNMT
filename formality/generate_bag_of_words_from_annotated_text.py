from transformers import MarianTokenizer

def generate_bag_of_words(model, formal_file_path, informal_file_path, length = 10):
    tokenizer = MarianTokenizer.from_pretrained(model)

    import re
    pattern = r'\[F\](.*?)\[/F\]'

    texts = []
    with open(formal_file_path, "r") as f:
        for line in f:
            texts.append(line.strip())

    formal_token_counts = {}
    for text in texts:
        formal_words = " ".join(re.findall(pattern, text))
        with tokenizer.as_target_tokenizer():
            tokens = tokenizer(formal_words)["input_ids"]
        for token in tokens:
            if token in formal_token_counts:
                formal_token_counts[token] += 1
            else:
                formal_token_counts[token] = 1

    texts = []
    with open(informal_file_path, "r") as f:
        for line in f:
            texts.append(line.strip())

    informal_token_counts = {}
    for text in texts:
        informal_words = " ".join(re.findall(pattern, text))
        with tokenizer.as_target_tokenizer():
            tokens = tokenizer(informal_words)["input_ids"]
        for token in tokens:
            if token in informal_token_counts:
                informal_token_counts[token] += 1
            else:
                informal_token_counts[token] = 1

    formal_bow = []
    informal_bow = []

    for item in sorted(formal_token_counts.items(), key=lambda item: item[1], reverse=True):
        if item[0] not in informal_token_counts:
            formal_bow.append(tokenizer.decode(item[0]))
        if len(formal_bow) == length:
            break
            
    for item in sorted(informal_token_counts.items(), key=lambda item: item[1], reverse=True):
        if item[0] not in formal_token_counts:
            informal_bow.append(tokenizer.decode(item[0]))
        if len(informal_bow) == length:
            break

    return formal_bow, informal_bow
