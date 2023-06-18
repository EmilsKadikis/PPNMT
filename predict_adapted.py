from run_pplm import *
import os
from debug_log_processing import *
from tqdm import tqdm
import random

def make_adapted_predictions(source_texts, hyperparameters, target_texts=None, verbosity="quiet", device="cpu", generate_unperturbed_predictions=False):
    bag_of_words = hyperparameters.get("bag_of_words", None)
    bag_of_words_file_name = None
    # if the bag of words is passed in directly, save it to a file. Easiest way to not have to change the code too much, which expects a file
    if type(bag_of_words) is list:
        bag_of_words_file_name = "tmp_bag_of_words" + str(random.randint(0, 1000000)) + ".txt"
        with open(bag_of_words_file_name, "w") as f:
            for word in bag_of_words:
                f.write(word + "\n")
        bag_of_words = bag_of_words_file_name

    negative_bag_of_words = hyperparameters.get("negative_bag_of_words", None)
    negative_bag_of_words_file_name = None
    # if the bag of words is passed in directly, save it to a file. Easiest way to not have to change the code too much, which expects a file
    if type(negative_bag_of_words) is list:
        negative_bag_of_words_file_name = "tmp_bag_of_words" + str(random.randint(0, 1000000)) + ".txt"
        with open(negative_bag_of_words_file_name, "w") as f:
            for word in negative_bag_of_words:
                f.write(word + "\n")
        negative_bag_of_words = negative_bag_of_words_file_name


    args = dict()
    args["bag_of_words"] = bag_of_words
    args["negative_bag_of_words"] = negative_bag_of_words
    args["num_samples"] = hyperparameters.get("num_samples", 1)
    args["sample"] = hyperparameters.get("sample", False)
    args["decay"] = hyperparameters.get("decay", False)

    args["pretrained_model"] = hyperparameters["translation_model"]
    args["length"] = hyperparameters.get("length", 100)
    args["colorama"] = False
    args["no_cuda"] = device == "cpu"
    args["verbosity"] = "quiet"
    args["top_k"] = hyperparameters.get("top_k", 5)

    args["gamma"] = hyperparameters.get("gamma", 1)
    args["num_iterations"] = hyperparameters.get("num_iterations", 6)
    args["stepsize"] = hyperparameters.get("stepsize", 0.1)
    args["window_length"] = hyperparameters.get("window_length", 5)
    args["kl_scale"] = hyperparameters.get("kl_scale", 0.1)
    args["gm_scale"] = hyperparameters.get("gm_scale", 0.95)    
    args["temperature"] = hyperparameters.get("temperature", 1)    
    args["grad_length"] = hyperparameters.get("grad_length", 10000)    
    args["stepsize_decay"] = hyperparameters.get("stepsize_decay", None)    
    args["generate_unperturbed"] = generate_unperturbed_predictions

    predictions_unperturbed = []
    predictions = []
    for i, text in enumerate(tqdm(source_texts)):
        args["cond_text"] = text
        results, debug_log = run_pplm_example(**args)
        if verbosity != "quiet":
            print(results[0][0])
            if target_texts is not None:
                print("Target:", target_texts[i])
            if generate_unperturbed_predictions:
                print("Unperturbed:", results[0][2])
            print("Perturbed:  ", results[0][1])
            print()
        predictions.append(results[0][1])
        if generate_unperturbed_predictions:
            predictions_unperturbed.append(results[0][2])
    
    if bag_of_words_file_name is not None and os.path.exists(bag_of_words_file_name):
        os.remove(bag_of_words_file_name)
    if negative_bag_of_words_file_name is not None and os.path.exists(negative_bag_of_words_file_name):
        os.remove(negative_bag_of_words_file_name)

    return predictions, predictions_unperturbed