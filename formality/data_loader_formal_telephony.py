def load_data():
    target_texts_formal = []
    with open("./formality/CoCoA-MT/train/en-de/formality-control.train.telephony.en-de.formal.de", "r") as f:
        for line in f:
            target_texts_formal.append(line.strip())

    target_texts_formal_feminine = []
    with open("./formality/CoCoA-MT/train/en-de/formality-control.train.telephony.en-de.formal.feminine.de", "r") as f:
        for line in f:
            target_texts_formal_feminine.append(line.strip())


    target_texts = list(map(list, zip(target_texts_formal, target_texts_formal_feminine)))


    source_texts = []
    with open("./formality/CoCoA-MT/train/en-de/formality-control.train.telephony.en-de.en", "r") as f:
        for line in f:
            source_texts.append(line.strip())

    return source_texts, target_texts