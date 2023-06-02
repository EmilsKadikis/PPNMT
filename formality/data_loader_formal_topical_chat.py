import os

def load_data(source_language, target_language):
    language_pair = source_language + "-" + target_language
    base_path = "./formality/CoCoA-MT/train/" + language_pair
    
    target_texts_formal = []
    with open(base_path + "/formality-control.train.topical_chat." + language_pair + ".formal." + target_language, "r") as f:
        for line in f:
            target_texts_formal.append(line.strip())

    feminine_texts_file = base_path + "/formality-control.train.topical_chat." + language_pair + ".formal.feminine." + target_language
    if os.path.exists(feminine_texts_file): 
        target_texts_formal_feminine = []
        with open(base_path + "/formality-control.train.topical_chat." + language_pair + ".formal.feminine." + target_language, "r") as f:
            for line in f:
                target_texts_formal_feminine.append(line.strip())

        target_texts = list(map(list, zip(target_texts_formal, target_texts_formal_feminine)))
    else:
        target_texts = target_texts_formal

    source_texts = []
    with open(base_path + "/formality-control.train.topical_chat." + language_pair + "." + source_language, "r") as f:
        for line in f:
            source_texts.append(line.strip())

    return source_texts, target_texts