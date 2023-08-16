from predict import make_predictions
from predict_adapted import make_adapted_predictions
from experiment_helpers import expand_experiments
from evaluation import evaluate_with_metric, extract_score
import wandb

import argparse
import json
import importlib
import time
import os
# setting up command line interface
parser = argparse.ArgumentParser(description='Machine translation domain adaptation experiments using a plug-and-play approach.')
parser.add_argument('--infile', nargs=1,
                    help="JSON file containing details about the experiment to run.",
                    type=argparse.FileType('r'))
args = parser.parse_args()

def determine_target_language(hyperparameters):
    if "target_language" in hyperparameters:
        return hyperparameters["target_language"]
    else:
        data_loader = hyperparameters["data_loader"]
        if isinstance(data_loader, dict):
            return data_loader["args"]["target_language"]
        
    raise Exception("Could not determine target language from hyperparameters.")

def load_data_from_data_loader(data_loader_definition): 
    # if data_loader_definition is a string, import the module and call load_data
    if isinstance(data_loader_definition, str):
        data_loader = importlib.import_module(data_loader_definition)
        return data_loader.load_data()
    # if data_loader_definition is a dictionary, call load_data with the arguments in the dictionary
    elif isinstance(data_loader_definition, dict):
        data_loader = importlib.import_module(data_loader_definition['name'])
        return data_loader.load_data(**data_loader_definition['args'])


if __name__ == "__main__":
    if args.infile is not None and args.infile[0] is not None:
        all_experiments = json.load(args.infile[0])['experiments']
    else:
        print("No .json file defining the experiment passed in, running 'experiment_definitions/formality_small.json' by default.")
        all_experiments = json.load(open("experiment_definitions/formality_small.json", "r"))['experiments']

    all_experiments = expand_experiments(all_experiments)
    for experiment_definition in all_experiments:
        print("=====================================================")
        print("Running experiment:")
        print(experiment_definition)

        experiment_entry_point = experiment_definition.pop("experiment_entry_point")
        experiment_entry_point = importlib.import_module(experiment_entry_point)
        experiment_entry_point.run(**experiment_definition)
        print("=====================================================")