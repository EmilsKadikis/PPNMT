possible_domains = ["auto", "education", "network", "phone"]
possible_bag_of_words_types = ["dict", "topic_modeling"]
possible_splits = ["dev", "test"]

def load_data(source_language, target_language, split, domain, bag_of_words_type, use_negative_bags_of_words=False):
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
    else:
        raise ValueError("Invalid bag of words type: " + bag_of_words_type)
    
    if not use_negative_bags_of_words:
        negative_bag_of_words = None

    return source_texts, target_texts, positive_bag_of_words, negative_bag_of_words

def _load_domain_texts(domain, split, language):
    base_path = f"./FGraDA/FDMT3.0/{split}/" 
    base_domain_path = base_path + domain + f".{split}."
    texts = []
    with open(base_domain_path + language, "r") as f:
        for line in f:
            texts.append(line.strip())
    return texts


def _get_dict_bag_of_words(domain, target_language):
    positive_bag_of_words = f"FGraDA/FDMT3.0/dict/{domain}.dict.{target_language}"
    negative_bag_of_words = ""
    for other_domain in possible_domains:
        if other_domain != domain:
            negative_bag_of_words += f"FGraDA/FDMT3.0/dict/{other_domain}.dict.{target_language};"
    negative_bag_of_words = negative_bag_of_words[:-1]
    return positive_bag_of_words, negative_bag_of_words


def _get_topic_modeling_bag_of_words(domain, split, target_language):
    from topic_modelling_bag_of_words_generator import generate_bags_of_words
    domain_texts = [_load_domain_texts(domain, split, target_language)]

    for other_domain in possible_domains:
        if other_domain != domain:
            domain_texts.append(_load_domain_texts(other_domain, split, target_language))
            
    bags_of_words = generate_bags_of_words(domain_texts)
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