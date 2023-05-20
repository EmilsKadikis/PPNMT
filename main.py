from formality import data_loader as formality_data_loader
from predict import make_predictions
from predict_adapted import make_adapted_predictions
from experiment_helpers import expand_experiments
from evaluation import evaluate_with_metric, extract_score

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

def save_results(experiment_definition, unadapted_predictions, adapted_predictions, unadapted_evaluation_results, adapted_evaluation_results):
    experiment_name = experiment_definition['experiment_name']
    timestamp = time.strftime("%m_%d__%H_%M_%S", time.gmtime())
    base_path = os.path.join('experiment_results', experiment_name + '_' + timestamp)
    os.mkdir(base_path)

    # save hyperparameters
    with open(os.path.join(base_path, "experiment.json"), "w") as f:
        f.writelines(json.dumps(experiment_definition, indent=4))

    # write unadapted predictions to file
    with open(os.path.join(base_path, "unadapted_predictions.txt"), "w") as f:
        for prediction in unadapted_predictions:
            f.write(prediction + "\n")

    # write adapted predictions to file
    with open(os.path.join(base_path, "adapted_predictions.txt"), "w") as f:
        for prediction in adapted_predictions:
            f.write(prediction + "\n")

    with open(os.path.join(base_path, "adapted_evaluation_results.json"), "w") as f:
        f.writelines(json.dumps(adapted_evaluation_results, indent=4))


    with open(os.path.join(base_path, "unadapted_evaluation_results.json"), "w") as f:
        f.writelines(json.dumps(unadapted_evaluation_results, indent=4))

    evaluation_summary = {}
    for metric in unadapted_evaluation_results.keys():
        evaluation_summary[metric] = {"unadapted": extract_score(metric, unadapted_evaluation_results[metric]), "adapted": extract_score(metric, adapted_evaluation_results[metric])}
    
    with open(os.path.join(base_path, "evaluation_summary.json"), "w") as f:
        f.writelines(json.dumps(evaluation_summary, indent=4))



if __name__ == "__main__":
    args.infile = [open('formality_adaptation.json', 'r')] # uncomment this to run an experiment without having to pass it in the command line
    if args.infile is not None and args.infile[0] is not None:
        all_experiments = json.load(args.infile[0])['experiments']
        all_experiments = expand_experiments(all_experiments)
        for experiment_definition in all_experiments:
            experiment_name = experiment_definition['experiment_name']
            hyperparameters = experiment_definition['hyperparameters']

            data_loader = importlib.import_module(hyperparameters["data_loader"])
            source_texts, target_texts = data_loader.load_data()

            device = experiment_definition.get("device", "cpu")
            model_name = hyperparameters["translation_model"]
            predictions = make_predictions(source_texts, output_file_name="predictions.txt", model_name=model_name, device=device)
            
            metrics = [("bleu", None),
                    ("google_bleu", None), 
                    ("sacrebleu", None), 
                    ("meteor", None), 
                    ("chrf", None), 
                    ("bertscore", {"lang":"de"})]

            unadapted_evaluation_results = {}
            for (metric_name, kwargs) in metrics:
                unadapted_evaluation_results[metric_name] = evaluate_with_metric(predictions, target_texts, metric_name, kwargs)

            adapted_predictions = make_adapted_predictions(source_texts, hyperparameters=hyperparameters)

            adapted_evaluation_results = {}
            for (metric_name, kwargs) in metrics:
                adapted_evaluation_results[metric_name] = evaluate_with_metric(adapted_predictions, source_texts, metric_name, kwargs)
                 
            save_results(experiment_definition, predictions, adapted_predictions, unadapted_evaluation_results, adapted_evaluation_results)
    else:
        print("Please pass in a .json file defining the experiment to run with --infile <file>")
