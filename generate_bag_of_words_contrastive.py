from transformers import MarianTokenizer

def generate_bag_of_words(model, first_domain_sentences, second_domain_sentences, length = 10):
    tokenizer = MarianTokenizer.from_pretrained(model)

    first_domain_token_counts = {}
    for text in first_domain_sentences:
        if isinstance(text, list):
            text = text[0]
        with tokenizer.as_target_tokenizer():
            tokens = tokenizer(text)["input_ids"]
        for token in tokens:
            if token in first_domain_token_counts:
                first_domain_token_counts[token] += 1
            else:
                first_domain_token_counts[token] = 1

    second_domain_token_counts = {}
    for text in second_domain_sentences:
        if isinstance(text, list):
            text = text[0]
        with tokenizer.as_target_tokenizer():
            tokens = tokenizer(text)["input_ids"]
        for token in tokens:
            if token in second_domain_token_counts:
                second_domain_token_counts[token] += 1
            else:
                second_domain_token_counts[token] = 1

    first_domain_bow = []
    second_domain_bow = []

    for item in sorted(first_domain_token_counts.items(), key=lambda item: item[1], reverse=True):
        if item[0] not in second_domain_token_counts:
            first_domain_bow.append(tokenizer.decode(item[0]))
        if len(first_domain_bow) == length:
            break
            
    for item in sorted(second_domain_token_counts.items(), key=lambda item: item[1], reverse=True):
        if item[0] not in first_domain_token_counts:
            second_domain_bow.append(tokenizer.decode(item[0]))
        if len(second_domain_bow) == length:
            break

    return first_domain_bow, second_domain_bow

if __name__ == "__main__":
    from formality.data_loader import load_data as load_data_formality
    _, formal, _, _ = load_data_formality("en", "de", "train", "all", "formal")
    _, informal, _, _ = load_data_formality("en", "de", "train", "all", "informal")
    formal_bow, informal_bow = generate_bag_of_words("Helsinki-NLP/opus-mt-en-de", formal, informal, 30)
    print(formal_bow)
    print(informal_bow)

    print("-----------------------------")

    from gender.data_loader import load_data as load_data_gender
    _, male_texts, _, _ = load_data_gender("train", "M")
    _, female_texts, _, _ = load_data_gender("train", "F")
    male_bow, female_bow = generate_bag_of_words("Helsinki-NLP/opus-mt-en-ar", male_texts, female_texts, 30)
    print(male_bow)
    print(female_bow)

    print("-----------------------------")
    from scientific_literature.data_loader import load_data
    _, target_medicine, _, _ = load_data("en", "ja", "train", "medicine", None)
    _, target_physics, _, _ = load_data("en", "ja", "train", "physics", None)
    medicine_bow, physics_bow = generate_bag_of_words("Helsinki-NLP/opus-mt-en-jap", target_medicine, target_physics, 30)
    print(medicine_bow)
    print(physics_bow)


