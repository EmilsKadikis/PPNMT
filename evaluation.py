import evaluate
from statistics import mean

def evaluate_with_metric(predictions, target_texts, metric_name, kwargs):
    metric = evaluate.load(metric_name)
    if kwargs is None:
        metric_result = metric.compute(predictions=predictions, references=target_texts)
    else:
        metric_result = metric.compute(predictions=predictions, references=target_texts, **kwargs)
    return metric_result

def extract_score(metric_name, metric_result):
    if metric_name == "bleu":
        return metric_result['bleu']
    elif metric_name == "google_bleu":
        return metric_result['google_bleu']
    elif metric_name == "sacrebleu":
        return metric_result['score']
    elif metric_name == "meteor":
        return metric_result['meteor']
    elif metric_name == "chrf":
        return metric_result['score']
    elif metric_name == "bertscore":
        return mean(metric_result['f1'])
    else:
        raise ValueError("Unknown metric name")
