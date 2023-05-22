import formality.data_loader as data_loader

def load_data():
    source_texts, target_texts = data_loader.load_data()
    indices = [12, 16, 17, 30, 78, 141, 197, 198]
    return [source_texts[i] for i in indices], [target_texts[i] for i in indices]