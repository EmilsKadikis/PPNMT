import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('language', type=str, choices=['de', 'es', 'fr', 'hi', 'it', 'ja'])
parser.add_argument('domain', type=str, choices=['telephony', 'topical_chat'])
parser.add_argument('formality', type=str, choices=['formal', 'informal'])

args = parser.parse_args()

if __name__ == "__main__":
    language_pair = "en-" + args.language
    base_path = "./formality/CoCoA-MT/train/" + language_pair + "/"
    file_name = "formality-control.train." + args.domain + "." + language_pair + "." + args.formality + ".annotated." + args.language
    texts = []

    with open(base_path + file_name, "r") as f:
        for line in f:
            # extract the text that's between [F] and [/F] tags with regex
            result = re.findall(r'\[F\](.*?)\[\/F\]', line)
            texts.extend(result)
    print(list(set(texts)))