from transformers import MarianMTModel, MarianTokenizer
from sklearn.model_selection import KFold
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import wandb
from fine_grained_tech.data_loader import load_data
from datasets import Dataset
import numpy as np
import time
from evaluation import evaluate_with_metric, extract_score
import random
import string
from experiment_helpers import load_data_from_data_loader, determine_target_language, determine_source_language
from predict import make_predictions

def run(**experiment_definition):
    # Get hyperparameters and other experiment parameters
    random_group_name = ''.join(random.choices(string.ascii_lowercase, k=5))
    wandb_project = experiment_definition["wandb_project"]
    device = experiment_definition.get("device", "cpu")

    hyperparameters = experiment_definition["hyperparameters"]

    model_name = hyperparameters["translation_model"]
    data_loader = hyperparameters["data_loader"]
    validation_data_loader = hyperparameters.get("validation_data_loader", None)
    num_folds = hyperparameters.get("num_folds", 1)
    num_train_epochs = hyperparameters.get("num_train_epochs", 3)
    batch_size = hyperparameters.get("batch_size", 16)

    target_language = determine_target_language(hyperparameters)
    source_language = determine_source_language(hyperparameters)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Load the data
    if validation_data_loader is None:
        source_texts, target_texts, _, _ = load_data_from_data_loader(data_loader)
        if isinstance(target_texts[0], list):
            all_texts = [(source, target[0]) for source, target in zip(source_texts, target_texts)]
        else:
            all_texts = [(source, target) for source, target in zip(source_texts, target_texts)]
    else:
        if num_folds > 1:
            print("Disabling cross-validation because a validation data loader is specified.")
            num_folds = 1
        
        source_texts, target_texts, _, _ = load_data_from_data_loader(data_loader)
        source_texts_val, target_texts_val, _, _ = load_data_from_data_loader(validation_data_loader)
        if isinstance(target_texts[0], list):
            training_texts = [(source, target[0]) for source, target in zip(source_texts, target_texts)]
            validation_texts = [(source, target[0]) for source, target in zip(source_texts_val, target_texts_val)]
        else:
            training_texts = [(source, target) for source, target in zip(source_texts, target_texts)]
            validation_texts = [(source, target) for source, target in zip(source_texts_val, target_texts_val)]

        initial_train_predictions = make_predictions(source_texts, model_name=model_name, device=device)
        initial_validation_predictions = make_predictions(source_texts_val, model_name=model_name, device=device)
    
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


    # Split the data into training and validation sets for this fold
    timestamp = time.strftime("%m_%d__%H_%M_%S", time.gmtime())
    if num_folds > 1:
        # Create the k-fold cross-validator
        random_state = random.randint(0, 100000000) 
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

        for fold, (train_index, dev_index) in enumerate(kf.split(all_texts)):
            wandb.init(
                project=wandb_project, 
                group=random_group_name,
                tags=experiment_definition.get('tags', None),
                notes=experiment_definition.get('notes', None),
                config=experiment_definition
            )

            print(f"Fold {fold + 1}")
            wandb.log({"fold": fold + 1, "random_state": random_state}) # Log the random state used to create the folds for reproducibility
            output_dir = f"models/output_{timestamp}_fold_{fold + 1}"
            train_dataset, dev_dataset = create_datasets_for_crossfold_validation(target_language, source_language, tokenizer, all_texts, train_index, dev_index)
            perform_training(model_name, train_dataset, dev_dataset, num_train_epochs, batch_size, output_dir, tokenizer, compute_metrics, device)

            wandb.finish()
    else:

        wandb.init(
            project=wandb_project, 
            group=random_group_name,
            tags=experiment_definition.get('tags', None),
            notes=experiment_definition.get('notes', None),
            config=experiment_definition
        )

        output_dir = f"models/output_{timestamp}"
        train_dataset, dev_dataset = create_datasets(target_language, source_language, tokenizer, training_texts, validation_texts)

        perform_training(model_name, train_dataset, dev_dataset, num_train_epochs, batch_size, output_dir, tokenizer, compute_metrics, device)
        
        final_train_predictions = make_predictions(source_texts, model_name=output_dir, tokenizer_name=model_name, device=device)
        final_validation_predictions = make_predictions(source_texts_val, model_name=output_dir, tokenizer_name=model_name, device=device)
        log_predictions_to_wandb("train_", source_texts, target_texts, initial_train_predictions, final_train_predictions)
        log_predictions_to_wandb("validation_", source_texts_val, target_texts_val, initial_validation_predictions, final_validation_predictions)

        wandb.finish()



    if num_folds > 1:
        print("Fold evaluation complete\n")

    wandb.finish()

    print("Training complete")

def perform_training(model_name, train_dataset, dev_dataset, num_train_epochs, batch_size, output_dir, tokenizer, compute_metrics, device):
    model = MarianMTModel.from_pretrained(model_name).to(device)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        save_strategy="epoch",
        logging_steps=10,
        logging_first_step=True,
        per_device_train_batch_size=batch_size,
        overwrite_output_dir=True,
        predict_with_generate = True,
        generation_num_beams=1,
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
    initial_evaluation = trainer.evaluate()
    print(f"Initial evaluation: {initial_evaluation}")
    wandb.log(initial_evaluation)
    trainer.train()
    trainer.save_model(output_dir)

def create_datasets_for_crossfold_validation(target_language, source_language, tokenizer, all_texts, train_index, dev_index):
    train_dataset = Dataset.from_list([{"id": i, "translation": {source_language: all_texts[i][0], target_language: all_texts[i][1]}} for i in train_index])
    dev_dataset = Dataset.from_list([{"id": i, "translation": {source_language: all_texts[i][0], target_language: all_texts[i][1]}} for i in dev_index])

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
    return train_dataset,dev_dataset

def create_datasets(target_language, source_language, tokenizer, train_texts, validation_texts):
    train_dataset = Dataset.from_list([{"id": i, "translation": {source_language: train_texts[i][0], target_language: train_texts[i][1]}} for i in range(len(train_texts))])
    validation_dataset = Dataset.from_list([{"id": i, "translation": {source_language: validation_texts[i][0], target_language: validation_texts[i][1]}} for i in range(len(validation_texts))])

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
    validation_dataset = validation_dataset.map(preprocess_function, batched=True)
    return train_dataset, validation_dataset

def log_predictions_to_wandb(prefix, source_texts, target_texts, predictions, adapted_predictions): 
    table = wandb.Table(columns = ["source", "target", "unadapted_translation", "adapted_translation"])
    [table.add_data(source, target, pred, adapted_pred) for source, target, pred, adapted_pred in zip(source_texts, target_texts, predictions, adapted_predictions)]
    wandb.log({prefix + "translations": table})
