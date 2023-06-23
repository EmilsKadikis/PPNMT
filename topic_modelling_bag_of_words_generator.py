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

def generate_bags_of_words(domain_texts):
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
            bag_of_words.append(word)
        bags_of_words.append(bag_of_words)

    return bags_of_words


if __name__ == "__main__":
    # Example
    from FGraDA.data_loader_automotive_zh_en import load_data as load_data_automotive
    from FGraDA.data_loader_education_zh_en import load_data as load_data_education
    from FGraDA.data_loader_network_zh_en import load_data as load_data_network
    from FGraDA.data_loader_phone_zh_en import load_data as load_data_phone

    _, target_automotive = load_data_automotive()
    _, target_education = load_data_education()
    _, target_network = load_data_network()
    _, target_phone = load_data_phone()

    domain_texts = [target_automotive, target_education, target_network, target_phone]
    bags_of_words = generate_bags_of_words(domain_texts)
    print(bags_of_words)
