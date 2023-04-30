from transformers import MarianMTModel, MarianTokenizer, GenerationConfig, AddedToken
import torch

pretrained_model = "Helsinki-NLP/opus-mt-en-de"

# set the device
device = "mps"
device = "cpu"

class MyTokenizer(MarianTokenizer):
    pass

# load pretrained model
model = MarianMTModel.from_pretrained(
    pretrained_model,
    output_hidden_states=True
)
model.to(device)
model.eval()

# load tokenizer
tokenizer = MarianTokenizer.from_pretrained(pretrained_model)

# new tokens
new_tokens = [AddedToken("Tabelle", single_word=True)]

# check if the tokens are already in the vocabulary
# new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())

print(new_tokens)
# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(new_tokens)
tokenizer.unique_no_split_tokens.extend([str(token) for token in new_tokens])
# tokenizer.save_pretrained('./updated_tokenizer')
# tokenizer = MarianTokenizer.from_pretrained('./updated_tokenizer')

tokenizer._switch_to_input_mode()
print(tokenizer.tokenize("Die Tabelle ist sehr ein großartig."))

# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))
