from predict import make_predictions
from predict_adapted import make_adapted_predictions
from evaluation import evaluate_with_metric, extract_score
import os
import importlib
import time
import wandb
import json

def _get_evaluation_summary(unadapted_evaluation_results, adapted_evaluation_results):
    evaluation_summary = {}
    total_percent_change = 0
    amount_of_metrics_used = 0
    for metric in unadapted_evaluation_results.keys():
        unadapted_score = extract_score(metric, unadapted_evaluation_results[metric])
        adapted_score = extract_score(metric, adapted_evaluation_results[metric])
        if unadapted_score == 0 or adapted_score == 0:
            print("Warning: unadapted score or adapted score for metric " + metric + " is 0, ignoring it for evaluation summary.")
            evaluation_summary[metric] = {
                    "unadapted": unadapted_score, 
                    "adapted": adapted_score,
                    "percent_change": 0
                }
        else:
            evaluation_summary[metric] = {
                "unadapted": unadapted_score, 
                "adapted": adapted_score,
                "percent_change": (adapted_score - unadapted_score) / unadapted_score
            }
            amount_of_metrics_used += 1
            total_percent_change += evaluation_summary[metric]["percent_change"]
    evaluation_summary["average_percent_change"] = total_percent_change / amount_of_metrics_used
    return evaluation_summary

def _log_results_in_wandb(experiment_definition, predictions, adapted_predictions, evaluation_summary):
    source_texts, target_texts, positive_bag_of_words, negative_bag_of_words = _load_data_from_data_loader(experiment_definition["hyperparameters"]["data_loader"])

    if positive_bag_of_words is not None:
        wandb.log({"positive_bag_of_words": positive_bag_of_words})
    if negative_bag_of_words is not None:
        wandb.log({"negative_bag_of_words": negative_bag_of_words})

    table = wandb.Table(columns = ["source", "target", "unadapted_translation", "adapted_translation"])
    [table.add_data(source, target, pred, adapted_pred) for source, target, pred, adapted_pred in zip(source_texts, target_texts, predictions, adapted_predictions)]
    wandb.log({"translations": table})
    wandb.log(evaluation_summary)

def _save_results(experiment_definition, unadapted_predictions, adapted_predictions, unadapted_evaluation_results, adapted_evaluation_results, evaluation_summary):
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

def _determine_target_language(hyperparameters):
    if "target_language" in hyperparameters:
        return hyperparameters["target_language"]
    else:
        data_loader = hyperparameters["data_loader"]
        if isinstance(data_loader, dict):
            return data_loader["args"]["target_language"]
        
    raise Exception("Could not determine target language from hyperparameters.")

def _load_data_from_data_loader(data_loader_definition): 
    # if data_loader_definition is a string, import the module and call load_data
    if isinstance(data_loader_definition, str):
        data_loader = importlib.import_module(data_loader_definition)
        return data_loader.load_data()
    # if data_loader_definition is a dictionary, call load_data with the arguments in the dictionary
    elif isinstance(data_loader_definition, dict):
        data_loader = importlib.import_module(data_loader_definition['name'])
        return data_loader.load_data(**data_loader_definition['args'])

def run(**experiment_definition):
    hyperparameters = experiment_definition['hyperparameters']

    source_texts, target_texts, positive_bag_of_words, negative_bag_of_words = _load_data_from_data_loader(hyperparameters["data_loader"])

    device = experiment_definition.get("device", "mps")
    model_name = hyperparameters["translation_model"]
    if positive_bag_of_words is not None:
        hyperparameters["bag_of_words"] = positive_bag_of_words
    if negative_bag_of_words is not None:
        hyperparameters["negative_bag_of_words"] = negative_bag_of_words

    predictions = make_predictions(source_texts, max_length=hyperparameters.get("length", 100), model_name=model_name, device=device)
    
    metrics = [("bleu", None),
            ("google_bleu", None), 
            ("sacrebleu", None), 
            ("meteor", None), 
            ("chrf", None), 
            ("bertscore", {"lang":_determine_target_language(hyperparameters)})]

    batch_size = experiment_definition.pop("batch_size", 50)
    worker_count = experiment_definition.pop("worker_count", 4)

    adapted_predictions = []
    adapted_predictions = make_adapted_predictions(source_texts, hyperparameters=hyperparameters, batch_size=batch_size, worker_count=worker_count, device=device)

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

    evaluation_summary = _get_evaluation_summary(unadapted_evaluation_results, adapted_evaluation_results)
    evaluation_summary = {**evaluation_summary, **extra_evaluation_results}

    _save_results(experiment_definition, predictions, adapted_predictions, unadapted_evaluation_results, adapted_evaluation_results, evaluation_summary)
    _log_results_in_wandb(experiment_definition, predictions, adapted_predictions, evaluation_summary)
    return adapted_evaluation_results, unadapted_evaluation_results