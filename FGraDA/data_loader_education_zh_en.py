def load_data():
    target_texts = []
    with open("./FGraDA/FDMT3.0/dev/education.dev.en", "r") as f:
        for line in f:
            target_texts.append(line.strip())

    source_texts = []
    with open("./FGraDA/FDMT3.0/dev/education.dev.zh", "r") as f:
        for line in f:
            source_texts.append(line.strip())

    return source_texts, target_texts