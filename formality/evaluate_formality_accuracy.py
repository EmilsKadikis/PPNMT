import os
import random
from formality.scorer import compute_score

def evaluate(adapted_hypotheses, unadapted_hypotheses, parameters):
        formal = parameters["annotated_formal_texts"]
        informal = parameters["annotated_informal_texts"]
        unadapted_hypotheses_path = os.path.join(str(random.randint(0, 1000000)) + "_tmp_unadapted_predictions.txt")
        adapted_hypotheses_path = os.path.join(str(random.randint(0, 1000000)) + "_tmp_adapted_predictions.txt")

        with open(unadapted_hypotheses_path, "w") as f:
            f.write("\n".join(unadapted_hypotheses))

        with open(adapted_hypotheses_path, "w") as f:
            f.write("\n".join(adapted_hypotheses))

        formal_acc, informal_acc = compute_score(
            unadapted_hypotheses_path, 
            formal, 
            informal, 
            tok_split=True
        ) 

        adapted_formal_acc, adapted_informal_acc = compute_score(
            adapted_hypotheses_path, 
            formal, 
            informal, 
            tok_split=True
        ) 

        os.remove(unadapted_hypotheses_path)
        os.remove(adapted_hypotheses_path)

        return {
            "formality accuracy": {
                    "adapted": adapted_formal_acc,
                    "unadapted": formal_acc
            },
            "informality accuracy": {
                    "adapted": adapted_informal_acc,
                    "unadapted": informal_acc
            }
        }