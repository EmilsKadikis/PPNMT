possible_bag_of_words_types = ["topic_modeling"]
possible_splits = ["train", "dev", "devtest", "test"]
possible_source_languages = ["en", "ja"]
possible_target_languages = ["en", "ja"]
possible_domains = {
    "general science": "A",
    "physics": "B",
    "chemistry": "C",
    "space/earth science": "D",
    "biology": "E",
    "agriculture": "F",
    "medicine": "G",
    "engineering": "H",
    "systems Engineering": "I",
    "computer Science": "J",
    "industrial Engineering": "K",
    "energy Science": "L",
    "nuclear Science": "M",
    "electronic Engineering": "N",
    "thermodynamics": "P",
    "mechanical Engineering": "Q",
    "construction": "R",
    "environmental Science": "S",
    "transportation Engineering": "T",
    "mining Engineering": "U",
    "metal Engineering": "W",
    "chemical Engineering": "X",
    "chemical Manufacturing": "Y",
    "other": "Z"
}

def load_data(source_language, target_language, split, domain, bag_of_words_type, count=200, count_for_bag_of_words=200, use_negative_bags_of_words=False):
    if domain not in possible_domains:
        raise ValueError("Invalid domain: " + domain)
    if bag_of_words_type is not None and bag_of_words_type not in possible_bag_of_words_types:
        raise ValueError("Invalid bag of words type: " + bag_of_words_type)
    if split not in possible_splits:
        raise ValueError("Invalid split: " + split)
    if source_language not in possible_source_languages:
        raise ValueError("Invalid source language: " + source_language)
    if target_language not in possible_target_languages:
        raise ValueError("Invalid target language: " + target_language)
    if source_language == target_language:
        raise ValueError("Source and target language must be different")
    if source_language == "zh" and target_language == "en" or source_language == "en" and target_language == "zh":
        raise ValueError("Chinese-English translation is not supported")
    domain_code = possible_domains[domain]
    japanese_texts, english_texts = _load_all_domain_texts(split)

    if source_language == "ja":
        source_texts = japanese_texts[domain_code][0:count]
        target_texts = english_texts[domain_code][0:count]
    else:
        source_texts = english_texts[domain_code][0:count]
        target_texts = japanese_texts[domain_code][0:count]

    if bag_of_words_type is None:
        positive_bag_of_words = None
        negative_bag_of_words = None
    elif bag_of_words_type == "topic_modeling":
        if target_language == "ja":
            positive_bag_of_words, negative_bag_of_words = _get_topic_modeling_bag_of_words(japanese_texts, domain_code, count_for_bag_of_words)
        else:
            positive_bag_of_words, negative_bag_of_words = _get_topic_modeling_bag_of_words(english_texts, domain_code, count_for_bag_of_words)
    else:
        raise ValueError("Invalid bag of words type: " + bag_of_words_type)
    
    if not use_negative_bags_of_words:
        negative_bag_of_words = None

    return source_texts[0:count], target_texts[0:count], positive_bag_of_words, negative_bag_of_words

def _load_all_domain_texts(split, count = None): 
    file_name = f"train-1.txt" if split == "train" else f"{split}.txt"
    file_path = f"./ASPEC/ASPEC/ASPEC-JE/{split}/{file_name}"

    japanese_texts = {}
    english_texts = {}
    with open(file_path, "r") as f:
        for line in f:
            split_line = line.split("|||")
            if split == "train":
                domain_code = split_line[1].strip()[0]
                japanese_text = split_line[3].strip()
                english_text = split_line[4].strip()
            else:
                domain_code = split_line[0].strip()[0]
                japanese_text = split_line[2].strip()
                english_text = split_line[3].strip()
            if domain_code not in japanese_texts:
                japanese_texts[domain_code] = []
                english_texts[domain_code] = []
            if count is not None and len(japanese_texts[domain_code]) >= count:
                continue
            japanese_texts[domain_code].append(japanese_text)
            english_texts[domain_code].append(english_text)
    return japanese_texts, english_texts


def _get_topic_modeling_bag_of_words(target_texts, domain_code, count):
    from topic_modelling_bag_of_words_generator import generate_bags_of_words
    domain_texts = [target_texts[domain_code][0:count]]
    other_domain_texts = []
    for other_domain in possible_domains.keys():
        other_domain_code = possible_domains[other_domain]
        if other_domain_code != domain_code:
            other_domain_texts.extend(target_texts[other_domain_code][0:count])
    domain_texts.append(other_domain_texts)
            
    bags_of_words = generate_bags_of_words(domain_texts)
    negative_bags_of_words = bags_of_words[1:]
    negative_bag_of_words = [word for bag_of_words in negative_bags_of_words for word in bag_of_words]
    return bags_of_words[0], negative_bag_of_words

if __name__ == "__main__":
    # examples
    source_texts, target_texts, positive_bag_of_words, negative_bag_of_words = load_data("ja", "en", "train", "medicine", "topic_modeling", count=200, count_for_bag_of_words=200, use_negative_bags_of_words=True)
    print(source_texts[0])
    print(target_texts[0])
    print(positive_bag_of_words)
    print(negative_bag_of_words)
    print(len(positive_bag_of_words))
    print(len(negative_bag_of_words))