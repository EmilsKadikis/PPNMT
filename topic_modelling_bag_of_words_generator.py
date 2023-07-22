from bertopic import BERTopic

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def _preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(tokens)

def generate_bags_of_words(domain_texts, score_threshold = None):
    all_texts = [text for domain in domain_texts for text in domain]        
    processed_data = [_preprocess_text(sentence) for sentence in all_texts]

    labels = [i for i in range(len(domain_texts)) for _ in range(len(domain_texts[i]))]
    topic_model = BERTopic()
    topic_model.fit_transform(processed_data, y=labels)

    bags_of_words = []
    for i in topic_model.get_topics():
        topic = topic_model.get_topic(i)
        bag_of_words = []
        for (word, score) in topic:
            if word != '' and (score_threshold is None or score >= score_threshold):
                bag_of_words.append(word)
        bags_of_words.append(bag_of_words)

    return bags_of_words


if __name__ == "__main__":
    # Example
    from fine_grained_tech.data_loader import load_data

    _, target_automotive, _, _ = load_data("zh", "en", "dev", "auto", None)
    _, target_education, _, _ = load_data("zh", "en", "dev", "education", None)
    _, target_network, _, _ = load_data("zh", "en", "dev", "network", None)
    _, target_phone, _, _ = load_data("zh", "en", "dev", "phone", None)

    domain_texts = [target_automotive, target_education, target_network, target_phone]
    bags_of_words = generate_bags_of_words(domain_texts)
    print(bags_of_words)
