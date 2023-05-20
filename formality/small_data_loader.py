import formality.data_loader as data_loader

def load_data():
    source_texts, target_texts = data_loader.load_data()
    return source_texts[:3], target_texts[:3]