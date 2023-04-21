from run_pplm import *
import random

args = dict()
args["bag_of_words"] = "technology"
args["num_samples"] = 5
args["sample"] = True
args["decay"] = True
args["cond_text"] = "Dies ist ein Test der Domänenanpassung für neuronische maschinelle Übersetzung."
# args["cond_text"] = "I know that the table is large."
# args["cond_text"] = "I know that the table with the data is large."

args["pretrained_model"] = "Helsinki-NLP/opus-mt-de-en"
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

for stepsize in [0.001, 0.01, 0.05, 0.1, 0.125, 0.15]:
    args["stepsize"] = stepsize
    results = run_pplm_example(**args)
    print("Stepsize:", stepsize)
    print(results[0][2]) # unperturbed translation
    print()
    for result in results:
        print(result[1])
    print()
