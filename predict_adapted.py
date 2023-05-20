from run_pplm import *
import os
from debug_log_processing import *
from tqdm import tqdm

def make_adapted_predictions(source_texts, hyperparameters, verbosity="quiet", device="cpu"):
    bag_of_words = hyperparameters["bag_of_words"]

    # if the bag of words is passed in directly, save it to a file. Easiest way to not have to change the code too much, which expects a file
    if type(bag_of_words) is list:
        with open("tmp_bag_of_words.txt", "w") as f:
            for word in bag_of_words:
                f.write(word + "\n")
        bag_of_words = "tmp_bag_of_words"


    args = dict()
    args["bag_of_words"] = bag_of_words
    args["num_samples"] = hyperparameters.get("num_samples", 1)
    args["sample"] = hyperparameters.get("sample", False)
    args["decay"] = hyperparameters.get("decay", False)

    args["pretrained_model"] = hyperparameters["translation_model"]
    args["length"] = hyperparameters.get("length", 100)
    args["colorama"] = False
    args["no_cuda"] = device == "cpu"
    args["verbosity"] = verbosity
    args["top_k"] = hyperparameters.get("top_k", 5)

    args["gamma"] = hyperparameters.get("gamma", 1)
    args["num_iterations"] = hyperparameters.get("num_iterations", 6)
    args["stepsize"] = hyperparameters.get("stepsize", 0.1)
    args["window_length"] = hyperparameters.get("window_length", 5)
    args["kl_scale"] = hyperparameters.get("kl_scale", 0.1)
    args["gm_scale"] = hyperparameters.get("gm_scale", 0.95)    
    args["generate_unperturbed"] = False

    predictions = []
    for text in tqdm(source_texts):
        args["cond_text"] = text
        results, debug_log = run_pplm_example(**args)
        predictions.append(results[0][1])
    
    if os.path.exists("tmp_bag_of_words.txt"):
        os.remove("tmp_bag_of_words.txt")

    return predictions