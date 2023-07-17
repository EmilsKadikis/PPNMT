import formality.data_loader as data_loader_formal_telephony

def load_data():
    source_texts, target_texts, _, _ = data_loader_formal_telephony.load_data("en", "de", "train", "telephony", "formal")
    indices = [173, 182, 190, 186, 196, 188, 176, 189, 177, 12, 16]
    return [source_texts[i] for i in indices], [target_texts[i] for i in indices], None, None