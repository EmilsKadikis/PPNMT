import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import models
from gensim.corpora import Dictionary

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

def generate_bag_of_words(documents, labels):
    # Step 1: Data Preparation
    # Assuming you have a list of labeled documents
    labeled_documents = list(zip(documents, labels))

    # Step 2: Preprocess Text and Create Corpus
    processed_documents = []
    for document in documents:
        tokens = preprocess_text(document)
        processed_documents.append(tokens)

    # Step 3: Create Dictionary and Corpus
    dictionary = Dictionary(processed_documents)
    corpus = [dictionary.doc2bow(doc) for doc in processed_documents]

    # Step 5: Train Labeled LDA Model
    num_topics = len(set(labels))  # Number of unique topic labels
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, alpha='auto', random_state=42)

    # Step 6: Describe Topics
    topic_descriptions = {}
    for topic_id in range(num_topics):
        top_words = [word for word, _ in lda_model.show_topic(topic_id)]
        topic_descriptions[topic_id] = top_words

    # Step 7: Return topic descriptions
    topics = {}
    for topic_id, description in topic_descriptions.items():
        topics[labels[topic_id]] = description

    return topics


if __name__ == "__main__":
    from scientific_literature.data_loader import load_data
    _, target_texts_medicine, _, _ = load_data("ja", "en", "train", "medicine", None, count=2000)
    _, target_texts_physics, _, _ = load_data("ja", "en", "train", "physics", None, count=2000)
    _, target_texts_chemistry, _, _ = load_data("ja", "en", "train", "chemistry", None, count=2000)
    _, target_texts_biology, _, _ = load_data("ja", "en", "train", "biology", None, count=2000)
    texts = target_texts_medicine + target_texts_physics + target_texts_chemistry + target_texts_biology
    labels = ["medicine"] * len(target_texts_medicine) + ["physics"] * len(target_texts_physics) + ["chemistry"] * len(target_texts_chemistry) + ["biology"] * len(target_texts_biology)
    result = generate_bag_of_words(texts, labels)
    print(result)
