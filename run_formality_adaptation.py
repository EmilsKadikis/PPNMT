from formality import data_loader as formality_data_loader
from predict import make_predictions
from predict_adapted import make_adapted_predictions
import evaluate

def evaluate_with_metric(predictions, target_texts, metric_name, kwargs):
    print(metric_name)
    metric = evaluate.load(metric_name)
    if kwargs is None:
        metric_result = metric.compute(predictions=predictions, references=target_texts)
    else:
        metric_result = metric.compute(predictions=predictions, references=target_texts, **kwargs)
    print(metric_result)
    print()

if __name__ == "__main__":
    device = "cpu"
    source_texts, target_texts = formality_data_loader.load_data()
    model_name = "Helsinki-NLP/opus-mt-en-de"
    predictions = make_predictions(source_texts, output_file_name="predictions.txt", model_name=model_name, device=device)
    
    metrics = [("bleu", None),
            ("google_bleu", None), 
            ("sacrebleu", None), 
            ("meteor", None), 
            ("chrf", None), 
            ("bertscore", {"lang":"de"})]

    for (metric_name, kwargs) in metrics:
        evaluate_with_metric(predictions, target_texts, metric_name, kwargs)

    adapted_predictions = make_adapted_predictions(source_texts, bag_of_words="./formality/formal", output_file_name="adapted_predictions.txt", model_name=model_name, device=device)
    for (metric_name, kwargs) in metrics:
        evaluate_with_metric(adapted_predictions, source_texts, metric_name, kwargs)
