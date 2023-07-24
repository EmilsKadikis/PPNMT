from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression

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
    empty_dimensionality_model = BaseDimensionalityReduction()
    clf = LogisticRegression()
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Create a fully supervised BERTopic instance
    topic_model= BERTopic(
            umap_model=empty_dimensionality_model,
            hdbscan_model=clf,
            ctfidf_model=ctfidf_model
    )
    topic_model.fit_transform(processed_data, y=labels)

    bags_of_words = []
    mappings = topic_model.topic_mapper_.get_mappings()
    for i in topic_model.get_topics():
        topic = topic_model.get_topic(i)
        bag_of_words = []
        for (word, score) in topic:
            if word != '' and (score_threshold is None or score >= score_threshold):
                bag_of_words.append(word)
        bags_of_words.append(bag_of_words)

    return [bags_of_words[mappings[i]] for i in mappings]


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
