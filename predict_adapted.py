from run_pplm import *
import random
from debug_log_processing import *
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

def make_adapted_predictions(source_texts, bag_of_words=None, output_file_name="predictions_adapted.txt", model_name="Helsinki-NLP/opus-mt-en-de", device="cpu"):
    args = dict()
    args["bag_of_words"] = bag_of_words
    args["num_samples"] = 1
    args["sample"] = False
    args["decay"] = True

    args["pretrained_model"] = "Helsinki-NLP/opus-mt-en-de"
    args["length"] = 100
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
    args["generate_unperturbed"] = False

    predictions = []
    for text in tqdm(source_texts):
        args["cond_text"] = text
        results, debug_log = run_pplm_example(**args)
        predictions.append(results[0][1])

    # write predictions to file
    with open(output_file_name, "w") as f:
        for prediction in predictions:
            f.write(prediction + "\n")
    
    return predictions