from run_pplm import *
import random
from debug_log_processing import *

def run_stepsize_experiment():
    args = dict()
    args["bag_of_words"] = "technology_de"
    args["num_samples"] = 1
    args["sample"] = False
    args["decay"] = True
    args["cond_text"] = "The website has a lot of tables."

    args["pretrained_model"] = "Helsinki-NLP/opus-mt-en-de"
    args["length"] = 25
    args["seed"] = random.randint(0, 100000000)
    args["colorama"] = True
    args["no_cuda"] = True
    args["verbosity"] = "quiet"
    args["top_k"] = 5

    args["gamma"] = 1
    args["num_iterations"] = 6
    args["stepsize"] = 0.1
    args["window_length"] = 5
    args["kl_scale"] = 0.1
    args["gm_scale"] = 0.95
    args["num_iterations"] = 6

    interesting_tokens = ["Tabellen", "Tische"]

    tokenizer = MarianTokenizer.from_pretrained(args["pretrained_model"])

    token_probabilities = dict()
    for token in interesting_tokens:
        token_probabilities[token] = []
    for stepsize in [0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.5]:
        args["stepsize"] = stepsize
        results, debug_log = run_pplm_example(**args)
        print("Stepsize:", stepsize)
        print(results[0][2]) # unperturbed translation
        print(tokenizer.tokenize(results[0][2])) # unperturbed translation
        print()
        for result in results:
            print(result[1])
            print(tokenizer.tokenize(result[1]))

        for token in interesting_tokens:
            if token_probabilities[token] == []:
                token_probabilities[token].append(get_total_unperturbed_probability_of_token(debug_log, token))
            token_probabilities[token].append(get_total_perturbed_probability_of_token(debug_log, token))
        print()
        print_perturbed_and_unperturbed_word_probabilities(tokenizer, debug_log, interesting_tokens)
        print()
        print()
        save_token_probabilities_as_csv(debug_log, "tabelle_"+str(stepsize))
    print(token_probabilities)

if __name__ == "__main__":
    run_stepsize_experiment()
