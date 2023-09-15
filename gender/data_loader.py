from generate_bag_of_words_contrastive import generate_bag_of_words

possible_splits = ["train", "dev", "test"]
possible_target_genders = ["M", "F"]
possible_automatic_bag_of_words = ["contrastive"]


def load_data(split, target_gender, allow_ambiguous_instances=False, count=None, automatic_bag_of_words=None, automatic_bag_of_words_model=None, automatic_bag_of_words_length=None):
    if split not in possible_splits:
        raise ValueError("Invalid split: " + split)
    if target_gender not in possible_target_genders:
        raise ValueError("Invalid target gender")
    
    source_texts, target_texts = _load_texts(split, target_gender, allow_ambiguous_instances, count)

    male_bow, female_bow = None, None
    if automatic_bag_of_words == "contrastive":
        assert automatic_bag_of_words_model is not None and automatic_bag_of_words_length is not None
        _, male_texts = _load_texts(split, "M", allow_ambiguous_instances, count)
        _, female_texts = _load_texts(split, "F", allow_ambiguous_instances, count)
        male_bow, female_bow = generate_bag_of_words(automatic_bag_of_words_model, automatic_bag_of_words_length, [male_texts, female_texts])
    if target_gender == "M":
        return source_texts, target_texts, male_bow, female_bow
    else:
        return source_texts, target_texts, female_bow, male_bow

def _load_texts(split, target_gender, allow_ambiguous_instances=False, count=None):
    base_path = f"./gender/Arabic-parallel-gender-corpus-v-2.0/data/{split}/" 
    if target_gender == "M":
        gender_suffix = "MM"
    elif target_gender == "F":
        gender_suffix = "FF"
    else:
        raise ValueError("Invalid target gender: " + target_gender)   

    file_name_en = f"{split}.ar.{gender_suffix}.en"
    file_name_ar = f"{split}.ar.{gender_suffix}"
    file_name_labels = f"{split}.ar.{gender_suffix}.label"

    source_texts, target_texts = [], []
    with open(base_path + file_name_en, "r") as f_en, open(base_path + file_name_ar, "r") as f_ar, open(base_path + file_name_labels, "r") as f_labels:
        for i, (line_en, line_ar, line_labels) in enumerate(zip(f_en, f_ar, f_labels)):
            if not allow_ambiguous_instances and "B" in line_labels:
                continue
            elif allow_ambiguous_instances and "BB" in line_labels:
                continue
            text_en = line_en.replace("<s>", "").replace("</s>", "").strip()
            text_ar = line_ar.replace("<s>", "").replace("</s>", "").strip()
            if source_texts != [] and source_texts[-1] == text_en: # quick and dirty way of removing duplicates
                continue
            source_texts.append(text_en)
            target_texts.append(text_ar)
            if count is not None and len(source_texts) >= count:
                break
    return source_texts, target_texts

if __name__ == "__main__":
    # examples
    source_texts, target_texts, positive_bag_of_words, negative_bag_of_words = load_data("train", "F", allow_ambiguous_instances=False, count=None)
    print(source_texts[0])
    print(source_texts[1])
    print(target_texts[0])
    print(target_texts[1])
    print(len(source_texts))
    print(len(target_texts))
