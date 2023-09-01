# import random

# def run_ppnmt_experiment(**experiment_definition):
#     rand_val = random.random()
#     print("Bag of words: ", experiment_definition['hyperparameters']['bag_of_words'], ":", rand_val)
#     return None, None, {"average_percent_change": rand_val}

from ppnmt_experiment import run as run_ppnmt_experiment

def _extract_bag_of_words(hyperparameters):
    return hyperparameters['hyperparameters'].pop('bag_of_words', None)

def run(**experiment_definition):
    bag_of_words = _extract_bag_of_words(experiment_definition)
    num_beams = experiment_definition['num_beams']

    word_values = {}
    for word in bag_of_words:
        print("Evaluating word: ", word)
        experiment_definition['hyperparameters']['bag_of_words'] = [word]
        _, _, evaluation_summary = run_ppnmt_experiment(**experiment_definition)
        average_percent_change_of_metrics = evaluation_summary['average_percent_change']
        word_values[word] = average_percent_change_of_metrics


    top_k_words = sorted(word_values.keys(), key=lambda word: word_values[word], reverse=True)[:num_beams]
    beam = []
    for word in top_k_words:
        hypothesis = ([word], word_values[word])
        remaining_words = [word for word in bag_of_words if word != hypothesis[0][0]]
        hypothesis = hypothesis + (remaining_words,)
        beam.append(hypothesis)
    explored_combinations = set()  # Track explored word combinations

    best_bag_of_words = None
    best_score = 0
    while True:
        new_beam = []
        for hypothesis in beam:
            bag_of_words, score, remaining_words = hypothesis
            for next_word in remaining_words:
                new_bag_of_words = sorted(bag_of_words + [next_word])  # Sort the hypothesis to ignore word order
                if tuple(new_bag_of_words) in explored_combinations:
                    continue
                
                print("Evaluating bag of words: ", new_bag_of_words)
                explored_combinations.add(tuple(new_bag_of_words))
                experiment_definition['hyperparameters']['bag_of_words'] = list(new_bag_of_words)
                _, _, evaluation_summary = run_ppnmt_experiment(**experiment_definition)
                average_percent_change_of_metrics = evaluation_summary['average_percent_change']

                new_hypothesis = (new_bag_of_words, average_percent_change_of_metrics, [word for word in remaining_words if word != next_word])
                new_beam.append(new_hypothesis)

                if average_percent_change_of_metrics > best_score:
                    print("Found better bag of words: ", new_bag_of_words, " with score: ", average_percent_change_of_metrics)
                    best_score = average_percent_change_of_metrics
                    best_bag_of_words = new_bag_of_words.copy()

        if not new_beam:
            break
        
        # sort new_beam by score and take the top k
        beam = sorted(new_beam, key=lambda hypothesis: hypothesis[1], reverse=True)[:num_beams]

    print("Best bag of words: ", best_bag_of_words, " with score: ", best_score)

if __name__ == "__main__":
    run(**{"hyperparameters": {"bag_of_words": ["hello", "world", "test", "okay", "bye"]}, "num_beams": 3})