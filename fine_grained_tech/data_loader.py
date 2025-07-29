possible_source_languages = ["en", "zh"]
possible_target_languages = ["en", "zh"]
possible_domains = ["auto", "education", "network", "phone", "auto-sentences", "education-sentences", "network-sentences", "phone-sentences"]
possible_bag_of_words_types = ["dict", "topic_modeling", "contrastive", "contrastive_dict"]
possible_splits = ["dev", "test"]

def load_data(source_language, target_language, split, domain, bag_of_words_type=None, use_negative_bags_of_words=False, bag_of_words_size=10):
    if source_language not in possible_source_languages:
        raise ValueError("Invalid source language: " + source_language)
    if target_language not in possible_target_languages:
        raise ValueError("Invalid target language: " + target_language)
    if domain not in possible_domains:
        raise ValueError("Invalid domain: " + domain)
    if bag_of_words_type is not None and bag_of_words_type not in possible_bag_of_words_types:
        raise ValueError("Invalid bag of words type: " + bag_of_words_type)
    if split not in possible_splits:
        raise ValueError("Invalid split: " + split)
    
    source_texts = _load_domain_texts(domain, split, source_language)
    target_texts = _load_domain_texts(domain, split, target_language)

    if bag_of_words_type is None:
        positive_bag_of_words = None
        negative_bag_of_words = None
    elif bag_of_words_type == "dict":
        positive_bag_of_words, negative_bag_of_words = _get_dict_bag_of_words(domain, target_language)
    elif bag_of_words_type == "topic_modeling":
        positive_bag_of_words, negative_bag_of_words = _get_topic_modeling_bag_of_words(domain, split, target_language)
    elif bag_of_words_type == "contrastive":
        positive_bag_of_words, negative_bag_of_words = _get_contrastive_bag_of_words(domain, split, target_language, bag_of_words_size)
    elif bag_of_words_type == "contrastive_dict":
        positive_bag_of_words, negative_bag_of_words = _get_contrastive_dict_bag_of_words(domain, split, target_language, bag_of_words_size)
    else:
        raise ValueError("Invalid bag of words type: " + bag_of_words_type)
    
    if not use_negative_bags_of_words:
        negative_bag_of_words = None

    return source_texts, target_texts, positive_bag_of_words, negative_bag_of_words

def _load_domain_texts(domain, split, language):
    base_path = f"./fine_grained_tech/FDMT3.0/{split}/" 
    base_domain_path = base_path + domain + f".{split}."
    texts = []
    with open(base_domain_path + language, "r") as f:
        for line in f:
            texts.append(line.strip())
    return texts


def _get_dict_bag_of_words(domain, target_language):
    if "sentences" in domain:
        domain = domain.split("-")[0]
    positive_bag_of_words = f"fine_grained_tech/FDMT3.0/dict/{domain}.dict.{target_language}"
    negative_bag_of_words = ""
    for other_domain in possible_domains:
        if "sentences" in other_domain:
            continue
        if other_domain != domain:
            negative_bag_of_words += f"fine_grained_tech/FDMT3.0/dict/{other_domain}.dict.{target_language};"
    negative_bag_of_words = negative_bag_of_words[:-1]
    return positive_bag_of_words, negative_bag_of_words


def _get_topic_modeling_bag_of_words(domain, split, target_language):
    from topic_modelling_bag_of_words_generator import generate_bags_of_words
    domain_texts = [_load_domain_texts(domain, split, target_language)]

    for other_domain in possible_domains:
        if "sentences" not in domain and "sentences" in other_domain:
            continue
        elif "sentences" in domain and "sentences" not in other_domain:
            continue
        if other_domain != domain:
            domain_texts.append(_load_domain_texts(other_domain, split, target_language))
            
    bags_of_words = generate_bags_of_words(domain_texts, target_language)
    negative_bags_of_words = bags_of_words[1:]
    negative_bag_of_words = [word for bag_of_words in negative_bags_of_words for word in bag_of_words]
    return bags_of_words[0], negative_bag_of_words

def _get_contrastive_bag_of_words(domain, split, target_language, max_bag_size=10):
    from generate_bag_of_words_contrastive import generate_bag_of_words as generate_bag_of_words_contrastive
    
    domain_texts = [_load_domain_texts(domain, split, target_language)]
    for other_domain in possible_domains:
        if "sentences" not in domain and "sentences" in other_domain:
            continue
        elif "sentences" in domain and "sentences" not in other_domain:
            continue
        if other_domain != domain:
            domain_texts.append(_load_domain_texts(other_domain, split, target_language))

    tokenizer_model = "Helsinki-NLP/opus-mt-en-zh" if target_language == "zh" else "Helsinki-NLP/opus-mt-zh-en"
    bags_of_words = generate_bag_of_words_contrastive(tokenizer_model, max_bag_size, domain_texts)
    negative_bags_of_words = bags_of_words[1:]
    negative_bag_of_words = [word for bag_of_words in negative_bags_of_words for word in bag_of_words]
    return bags_of_words[0], negative_bag_of_words

def _get_contrastive_dict_bag_of_words(domain, split, target_language, max_bag_size=10):
    if "sentences" in domain:
        domain = domain.split("-")[0]
    from generate_bag_of_words_contrastive import generate_bag_of_words as generate_bag_of_words_contrastive
    positive_dict_path = f"fine_grained_tech/FDMT3.0/dict/{domain}.dict.{target_language}"
    positive_words = []
    with open(positive_dict_path, "r") as f:
        for line in f:
            positive_words.append(line.strip())

    domain_texts = [positive_words]
    for other_domain in possible_domains:
        if "sentences" in other_domain:
            continue
        if other_domain != domain:
            negative_words = []
            negative_dict_path = f"fine_grained_tech/FDMT3.0/dict/{other_domain}.dict.{target_language}"
            with open(negative_dict_path, "r") as f:
                for line in f:
                    negative_words.append(line.strip())

            domain_texts.append(negative_words)

    tokenizer_model = "Helsinki-NLP/opus-mt-en-zh" if target_language == "zh" else "Helsinki-NLP/opus-mt-zh-en"
    bags_of_words = generate_bag_of_words_contrastive(tokenizer_model, max_bag_size, domain_texts)
    negative_bags_of_words = bags_of_words[1:]
    negative_bag_of_words = [word for bag_of_words in negative_bags_of_words for word in bag_of_words]
    return bags_of_words[0], negative_bag_of_words


if __name__ == "__main__":
    # examples
    source_texts, target_texts, positive_bag_of_words, negative_bag_of_words = load_data("en", "zh", "dev", "phone", "dict", True)
    print(source_texts[0])
    print(target_texts[0])
    print(positive_bag_of_words)
    print(negative_bag_of_words)

    print("------------------")

    source_texts, target_texts, positive_bag_of_words, negative_bag_of_words = load_data("zh", "en", "dev", "education", "topic_modeling", True)
    print(source_texts[0])
    print(target_texts[0])
    print(positive_bag_of_words)
    print(negative_bag_of_words)    
    
    print("------------------")

    source_texts, target_texts, positive_bag_of_words, negative_bag_of_words = load_data("zh", "en", "dev", "education", "contrastive_dict", True)
    print(source_texts[0])
    print(target_texts[0])
    print(positive_bag_of_words)
    print(negative_bag_of_words)