import glob, os
import traceback
from scorer import compute_score

if __name__ == "__main__":
    for experiment_result_dir in glob.iglob('./experiment_results/formality_adaptation_**', recursive=True):
        score_file_name = os.path.join(experiment_result_dir, "formality_scores.txt")
        if os.path.exists(score_file_name):
            continue
        unadapted_hypotheses = os.path.join(experiment_result_dir, "unadapted_predictions.txt")
        adapted_hypotheses = os.path.join(experiment_result_dir, "adapted_predictions.txt")

        try:
            formal_acc, informal_acc = compute_score(
                unadapted_hypotheses, 
                "./formality/CoCoA-MT/train/en-de/formality-control.train.telephony.en-de.formal.annotated.de", 
                "./formality/CoCoA-MT/train/en-de/formality-control.train.telephony.en-de.informal.annotated.de", 
                tok_split=True
            ) 

            adapted_formal_acc, adapted_informal_acc = compute_score(
                adapted_hypotheses, 
                "./formality/CoCoA-MT/train/en-de/formality-control.train.telephony.en-de.formal.annotated.de", 
                "./formality/CoCoA-MT/train/en-de/formality-control.train.telephony.en-de.informal.annotated.de", 
                tok_split=True
            ) 

            with open(score_file_name, "w") as score_file:
                score_file.write(f"Formal Accuracy: {formal_acc}  Informal Accuracy: {informal_acc}")
                score_file.write(f"\n")
                score_file.write(f"Adapted Formal Accuracy: {adapted_formal_acc}  Adapted Informal Accuracy: {adapted_informal_acc}")
        except Exception as e:
            traceback.print_exc()

