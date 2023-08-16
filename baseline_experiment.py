from transformers import MarianMTModel, MarianTokenizer
from sklearn.model_selection import KFold
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import wandb
from fine_grained_tech.data_loader import load_data
from datasets import Dataset
import numpy as np
import evaluate
from evaluation import evaluate_with_metric, extract_score
import random
import string
from experiment_helpers import load_data_from_data_loader, determine_target_language, determine_source_language

def run(**experiment_definition):
    # Get hyperparameters and other experiment parameters
    random_group_name = ''.join(random.choices(string.ascii_lowercase, k=5))
    wandb_project = experiment_definition["wandb_project"]
    device = experiment_definition.get("device", "cpu")

    hyperparameters = experiment_definition["hyperparameters"]

    model_name = hyperparameters["translation_model"]
    data_loader = hyperparameters["data_loader"]
    num_folds = hyperparameters.get("num_folds", 5)
    num_train_epochs = hyperparameters.get("num_train_epochs", 3)
    batch_size = hyperparameters.get("batch_size", 16)

    target_language = determine_target_language(hyperparameters)
    source_language = determine_source_language(hyperparameters)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Load the data
    source_texts, target_texts, _, _ = load_data_from_data_loader(data_loader)
    if isinstance(target_texts[0], list):
        texts = [(source, target[0]) for source, target in zip(source_texts, target_texts)]
    else:
        texts = [(source, target) for source, target in zip(source_texts, target_texts)]
    
    # Create the k-fold cross-validator
    random_state = random.randint(0, 100000000) 
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        metrics = [("bleu", None),
            ("chrf", None), 
            ("bertscore", {"lang":target_language})]

        results = {}
        for (metric_name, args) in metrics:
            result = evaluate_with_metric(decoded_preds, decoded_labels, metric_name, args)
            result = { metric_name: extract_score(metric_name, result) }
            results.update(result)
        return results

    # Perform k-fold cross-validation
    for fold, (train_index, dev_index) in enumerate(kf.split(texts)):
        model = MarianMTModel.from_pretrained(model_name).to(device)
        print(f"Fold {fold + 1}")
        wandb.init(
            project=wandb_project, 
            group=random_group_name,
            tags=experiment_definition.get('tags', None),
            notes=experiment_definition.get('notes', None),
        )

        wandb.log({"fold": fold + 1, "random_state": random_state}) # Log the random state used to create the folds for reproducibility

        # Split the data into training and validation sets for this fold
        train_dataset = Dataset.from_list([{"id": i, "translation": {source_language: texts[i][0], target_language: texts[i][1]}} for i in train_index])
        dev_dataset = Dataset.from_list([{"id": i, "translation": {source_language: texts[i][0], target_language: texts[i][1]}} for i in dev_index])

        def preprocess_function(examples):
            inputs = [ex[source_language] for ex in examples["translation"]]
            targets = [ex[target_language] for ex in examples["translation"]]
            model_inputs = tokenizer(inputs, truncation=True)
            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        dev_dataset = dev_dataset.map(preprocess_function, batched=True)

        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"output_fold_{fold + 1}",
            num_train_epochs=num_train_epochs,
            evaluation_strategy="steps",
            logging_strategy="steps",
            eval_steps=10,
            logging_steps=10,
            per_device_train_batch_size=batch_size,
            overwrite_output_dir=True,
            predict_with_generate = True,
            # Wandb integration
            report_to="wandb"
        )

        # Initialize the Seq2SeqTrainer with wandb integration
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            compute_metrics=compute_metrics
        )

        # Fine-tuning
        print("Fine-tuning...")
        trainer.train()

        print("Fold evaluation complete\n")
        wandb.finish()

    print("Cross-validation complete")

