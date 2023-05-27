def load_data():
    target_texts_informal = []
    with open("./formality/CoCoA-MT/train/en-de/formality-control.train.telephony.en-de.informal.de", "r") as f:
        for line in f:
            target_texts_informal.append(line.strip())

    target_texts_informal_feminine = []
    with open("./formality/CoCoA-MT/train/en-de/formality-control.train.telephony.en-de.informal.feminine.de", "r") as f:
        for line in f:
            target_texts_informal_feminine.append(line.strip())


    target_texts = list(map(list, zip(target_texts_informal, target_texts_informal_feminine)))


    source_texts = []
    with open("./formality/CoCoA-MT/train/en-de/formality-control.train.telephony.en-de.en", "r") as f:
        for line in f:
            source_texts.append(line.strip())

    return source_texts, target_texts