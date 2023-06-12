def get_probabilites_of_token(logs, token_to_find):
    result = []
    for entry in logs:
        found = False
        for (word, probability) in entry[2]:
            if word == token_to_find:
                result.append(probability)
                found = True
                break
        if not found:
            result.append(0)
    return result

def print_perturbed_and_unperturbed_word_probabilities(log, words_to_find): 
    unperturbed = [entry for entry in log if entry[0] == "unperturbed"]
    perturbed = [entry for entry in log if entry[0] == "perturbed"]
    
    all_tokens_in_unperturbed = set([word[0] for entry in unperturbed for word in entry[2]])
    all_tokens_in_perturbed = set([word[0] for entry in perturbed for word in entry[2]])

    for word in words_to_find: 
        print(word)
        print("    Unperturbed")
        print("        ", get_probabilites_of_token(unperturbed, word))
        print("    Perturbed")
        print("        ", get_probabilites_of_token(perturbed, word))

def get_total_perturbed_probability_of_token(log, token):
    perturbed = [entry for entry in log if entry[0] == "perturbed"]
    return sum(get_probabilites_of_token(perturbed, token))

def get_total_unperturbed_probability_of_token(log, token):
    unperturbed = [entry for entry in log if entry[0] == "unperturbed"]
    return sum(get_probabilites_of_token(unperturbed, token))




def save_token_probabilities_as_csv(log, filename):
    unperturbed = [entry for entry in log if entry[0] == "unperturbed"]
    perturbed = [entry for entry in log if entry[0] == "perturbed"]
    
    all_tokens_in_unperturbed = set([word[0] for entry in unperturbed for word in entry[2]])
    all_tokens_in_perturbed = set([word[0] for entry in perturbed for word in entry[2]])

    with open(filename+"_unperturbed.csv", "w") as file:
        file.write("token,probability\n")
        for token in all_tokens_in_unperturbed.union(all_tokens_in_perturbed):
            unperturbed_probabilities = get_probabilites_of_token(unperturbed, token)
            if token == ",":
                token = '","'
            file.write(token + ",")
            file.write(",".join([str(probability) for probability in unperturbed_probabilities]))
            file.write("\n")
    with open(filename+"_perturbed.csv", "w") as file:
        file.write("token,probability\n")
        for token in all_tokens_in_unperturbed.union(all_tokens_in_perturbed):
            perturbed_probabilities = get_probabilites_of_token(perturbed, token)
            if token == ",":
                token = '","'
            file.write(token + ",")
            file.write(",".join([str(probability) for probability in perturbed_probabilities]))
            file.write("\n")
