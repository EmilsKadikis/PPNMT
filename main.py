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

def initialize_experiment(experiment_definition):
    wandb.init(
        # set the wandb project where this run will be logged
        project="ppnmt",
        group=experiment_definition['experiment_name'],
        tags=experiment_definition.get('tags', None),
        notes=experiment_definition.get('notes', None),
        # track hyperparameters and run metadata
        config=experiment_definition['hyperparameters'],
    )

def log_results_in_wandb(experiment_definition, data_loader, predictions, adapted_predictions, unadapted_evaluation_results, adapted_evaluation_results, extra_evaluation_results):
    data_loader = importlib.import_module(experiment_definition['hyperparameters']['data_loader'])
    source_texts, target_texts = data_loader.load_data()

    table = wandb.Table(columns = ["source", "target", "unadapted_translation", "adapted_translation"])
    [table.add_data(source, target, pred, adapted_pred) for source, target, pred, adapted_pred in zip(source_texts, target_texts, predictions, adapted_predictions)]
    wandb.log({"translations": table})

    evaluation_summary = {}
    for metric in unadapted_evaluation_results.keys():
        evaluation_summary[metric] = {"unadapted": extract_score(metric, unadapted_evaluation_results[metric]), "adapted": extract_score(metric, adapted_evaluation_results[metric])}

    evaluation_summary = {**evaluation_summary, **extra_evaluation_results}
    wandb.log(evaluation_summary)

def save_results(experiment_definition, unadapted_predictions, adapted_predictions, unadapted_evaluation_results, adapted_evaluation_results):
    evaluation_summary = {}
    for metric in unadapted_evaluation_results.keys():
        evaluation_summary[metric] = {"unadapted": extract_score(metric, unadapted_evaluation_results[metric]), "adapted": extract_score(metric, adapted_evaluation_results[metric])}

    experiment_name = experiment_definition['experiment_name']
    timestamp = time.strftime("%m_%d__%H_%M_%S", time.gmtime())
    details = str(evaluation_summary['bertscore']['adapted']) + '-' + str(evaluation_summary['chrf']['adapted'])
    base_path = os.path.join('experiment_results', experiment_name + '_' + details + '_' + timestamp)
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
    
    with open(os.path.join(base_path, "evaluation_summary.json"), "w") as f:
        f.writelines(json.dumps(evaluation_summary, indent=4))

    return base_path

if __name__ == "__main__":
    if args.infile is not None and args.infile[0] is not None:
        all_experiments = json.load(args.infile[0])['experiments']
    else:
        print("No .json file defining the experiment passed in, running 'automotive_domain.json' by default.")
        all_experiments = json.load(open("automotive_domain.json", "r"))['experiments']

    all_experiments = expand_experiments(all_experiments)
    for experiment_definition in all_experiments:
        initialize_experiment(experiment_definition)
        print("=====================================================")
        print("Running experiment:")
        print(experiment_definition)

        experiment_name = experiment_definition['experiment_name']
        hyperparameters = experiment_definition['hyperparameters']

        data_loader = importlib.import_module(hyperparameters["data_loader"])
        source_texts, target_texts = data_loader.load_data()

        device = experiment_definition.get("device", "cpu")
        model_name = hyperparameters["translation_model"]
        generate_unperturbed_predictions = hyperparameters.get("generate_unperturbed_predictions", False)
        if not generate_unperturbed_predictions:
            predictions = make_predictions(source_texts, max_length=hyperparameters.get("length", 100), output_file_name="predictions.txt", model_name=model_name, device=device)
        
        metrics = [("bleu", None),
                ("google_bleu", None), 
                ("sacrebleu", None), 
                ("meteor", None), 
                ("chrf", None), 
                ("bertscore", {"lang":"de"})]


        adapted_predictions = []
        adapted_predictions, unperturbed_predictions = make_adapted_predictions(source_texts, verbosity=experiment_definition.get("verbosity", "quiet"), hyperparameters=hyperparameters, target_texts=target_texts, generate_unperturbed_predictions=generate_unperturbed_predictions)

        if generate_unperturbed_predictions:
            predictions = unperturbed_predictions
        unadapted_evaluation_results = {}
        for (metric_name, kwargs) in metrics:
            unadapted_evaluation_results[metric_name] = evaluate_with_metric(predictions, target_texts, metric_name, kwargs)

        adapted_evaluation_results = {}
        for (metric_name, kwargs) in metrics:
            adapted_evaluation_results[metric_name] = evaluate_with_metric(adapted_predictions, target_texts, metric_name, kwargs)

        extra_evaluation_results = {}
        extra_evaluation = experiment_definition.get("extra_evaluation", None)      
        if extra_evaluation is not None:
            evaluation = importlib.import_module(extra_evaluation["name"])
            extra_evaluation_results = evaluation.evaluate(adapted_predictions, predictions, extra_evaluation.get("args", {}))

        save_results(experiment_definition, predictions, adapted_predictions, unadapted_evaluation_results, adapted_evaluation_results)
        log_results_in_wandb(experiment_definition, data_loader, predictions, adapted_predictions, unadapted_evaluation_results, adapted_evaluation_results, extra_evaluation_results)
        wandb.finish()