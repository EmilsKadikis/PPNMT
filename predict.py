from transformers import MarianMTModel, MarianTokenizer, GenerationConfig

def make_predictions(source_texts, max_length=100, output_file_name=None, model_name="Helsinki-NLP/opus-mt-en-de", tokenizer_name=None, device="cpu"):
    if tokenizer_name is None:
        tokenizer_name = model_name

    model = MarianMTModel.from_pretrained(
        model_name,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)

    predictions = []
    tokenizer_output = tokenizer(source_texts, return_tensors="pt", padding=True).to(device)
    config = GenerationConfig(num_beams=1, do_sample=False, max_length=max_length)
    predictions = model.generate(tokenizer_output.input_ids, attention_mask=tokenizer_output.attention_mask, generation_config=config)
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # write predictions to file
    if output_file_name is not None:
        with open(output_file_name, "w") as f:
            for prediction in predictions:
                f.write(prediction + "\n")
    
    return predictions


if __name__ == '__main__':
    from formality.data_loader_formal_telephony import load_data
    source_texts, target_texts, _, _ = load_data("en", "de")
    predictions = make_predictions(source_texts, model_name="Helsinki-NLP/opus-mt-en-de")
    print(len(predictions))
    print(predictions[0])
    print(predictions[1])
    print(predictions[2])
    print(predictions[3])
    print(predictions[4])