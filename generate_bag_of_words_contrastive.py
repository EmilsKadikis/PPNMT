from transformers import MarianTokenizer

def generate_bag_of_words(model, length, all_domain_sentences):
    tokenizer = MarianTokenizer.from_pretrained(model)

    all_domain_token_counts = []
    for one_domain_sentences in all_domain_sentences:
        one_domain_token_counts = {}
        for text in one_domain_sentences:
            if isinstance(text, list):
                text = text[0]
            with tokenizer.as_target_tokenizer():
                tokens = tokenizer(text)["input_ids"]
            for token in tokens:
                if token in one_domain_token_counts:
                    one_domain_token_counts[token] += 1
                else:
                    one_domain_token_counts[token] = 1
        all_domain_token_counts.append(one_domain_token_counts)

    unique_tokens = set()
    seen_tokens = set()
    for one_domain_token_counts in all_domain_token_counts:
        for item in one_domain_token_counts.items():
            if item[1] < 2:
                continue
            if item[0] in seen_tokens:
                if item[0] in unique_tokens:
                    unique_tokens.remove(item[0])
            else:
                unique_tokens.add(item[0])
                seen_tokens.add(item[0])

    all_domain_token_counts = [dict([item for item in one_domain_token_counts.items() if item[0] in unique_tokens]) for one_domain_token_counts in all_domain_token_counts]

    all_domain_bows = []
    for one_domain_token_counts in all_domain_token_counts:
        one_domain_bow = []
        for item in sorted(one_domain_token_counts.items(), key=lambda item: item[1], reverse=True)[0:length]:
            one_domain_bow.append(tokenizer.decode(item[0]))
        all_domain_bows.append(one_domain_bow)
        
    return all_domain_bows

if __name__ == "__main__":
    from formality.data_loader import load_data as load_data_formality
    _, formal, _, _ = load_data_formality("en", "de", "train", "all", "formal")
    _, informal, _, _ = load_data_formality("en", "de", "train", "all", "informal")
    formal_bow, informal_bow = generate_bag_of_words("Helsinki-NLP/opus-mt-en-de", 30, [formal, informal])
    print(formal_bow)
    print(informal_bow)

    print("-----------------------------")

    from gender.data_loader import load_data as load_data_gender
    _, male_texts, _, _ = load_data_gender("train", "M")
    _, female_texts, _, _ = load_data_gender("train", "F")
    male_bow, female_bow = generate_bag_of_words("Helsinki-NLP/opus-mt-en-ar", 30, [male_texts, female_texts])
    print(male_bow)
    print(female_bow)

    print("-----------------------------")
    from scientific_literature.data_loader import load_data
    _, target_medicine, _, _ = load_data("ja", "en", "train", "medicine", None)
    _, target_physics, _, _ = load_data("ja", "en", "train", "physics", None)
    _, target_biology, _, _ = load_data("ja", "en", "train", "biology", None)
    _, target_mechanical_engineering, _, _ = load_data("ja", "en", "train", "mechanical engineering", None)
    medicine_bow, physics_bow, biology_bow, mechanical_engineering_bow = generate_bag_of_words("Helsinki-NLP/opus-mt-ja-en", 30, [target_medicine, target_physics, target_biology, target_mechanical_engineering])
    print(medicine_bow)
    print(physics_bow)
    print(biology_bow)
    print(mechanical_engineering_bow)


