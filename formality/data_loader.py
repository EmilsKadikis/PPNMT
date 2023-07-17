import os

allowed_source_languages = ["en"]
allowed_target_languages = ["de", "es", "fr", "hi", "it", "ja"]
allowed_splits = ["train", "test"]
allowed_domains = ["telephony", "topical_chat", "call_center", "all"]
allowed_target_formalities = ["formal", "informal"]

def load_data(source_language, target_language, split, domain, target_formality):
    if source_language not in allowed_source_languages:
        raise ValueError("Invalid source language: " + source_language)
    if target_language not in allowed_target_languages:
        raise ValueError("Invalid target language: " + target_language)
    if split not in allowed_splits:
        raise ValueError("Invalid split: " + split)
    if domain not in allowed_domains:
        raise ValueError("Invalid domain: " + domain)
    if target_formality not in allowed_target_formalities:
        raise ValueError("Invalid target formality: " + target_formality)
    if domain == "call_center" and split == "train":
        raise ValueError(f"Invalid domain and split combination: {domain} {split}")

    language_pair = source_language + "-" + target_language
    base_path = f"./formality/CoCoA-MT/{split}/{language_pair}/"
    
    if domain == "all":
        if split == "train":
            domains = ["telephony", "topical_chat"]
        elif split == "test":
            domains = ["telephony", "topical_chat", "call_center"]
    else:
        domains = [domain]

    source_files, target_files, target_files_feminine = [], [],[]
    for domain in domains:
        source_suffix = f".{source_language}" if split == "train" else ""
        target_suffix = f".{target_language}" if split == "train" else ""
        source_files.append(f"formality-control.{split}.{domain}.{language_pair}{source_suffix}")
        target_files.append(f"formality-control.{split}.{domain}.{language_pair}.{target_formality}{target_suffix}")
        if target_language != "ja":
            target_files_feminine.append(f"formality-control.{split}.{domain}.{language_pair}.{target_formality}.feminine{target_suffix}")

    source_texts, target_texts, target_texts_feminine = [], [], []
    for file in source_files:
        with open(base_path + file, "r") as f:
            for line in f:
                source_texts.append(line.strip())
    for file in target_files:
        with open(base_path + file, "r") as f:
            for line in f:
                target_texts.append(line.strip())
    for file in target_files_feminine:
        with open(base_path + file, "r") as f:
            for line in f:
                target_texts_feminine.append(line.strip())

    if target_texts_feminine != []:
        target_texts = list(map(list, zip(target_texts, target_texts_feminine)))

    return source_texts, target_texts, None, None


if __name__ == "__main__":
    source_texts, target_texts, _, _ = load_data("en", "de", "train", "all", "formal")
    print(len(source_texts))
    print(len(target_texts))
    print(source_texts[200])
    print(target_texts[200])

    print("=======================")

    source_texts, target_texts, _, _ = load_data("en", "hi", "test", "all", "formal")
    print(len(source_texts))
    print(len(target_texts))
    print(source_texts[200])
    print(target_texts[200])

    print("=======================")

    source_texts, target_texts, _, _ = load_data("en", "ja", "test", "call_center", "informal")
    print(len(source_texts))
    print(len(target_texts))
    print(source_texts[0])
    print(target_texts[0])