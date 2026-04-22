import os
import csv
from collections import defaultdict
from tqdm.auto import tqdm
import re


def levenshtein_distance(s1, s2):
    """Calculates the minimum edit distance between two strings."""
    s1 = s1.replace(" ", "")
    s2 = s2.replace(" ", "")
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calculate_smart_average_ld(output_dir):
    """
    Calculates the slot-averaged norm. Levenshtein distance for each language.
    """
    lang_metrics = {}

    # 1. Iterate over language folders
    for lang_iso in tqdm(os.listdir(output_dir)):
        lang_path = os.path.join(output_dir, lang_iso)

        # Skip if it's not a directory or if it's completely empty
        if not os.path.isdir(lang_path) or not os.listdir(lang_path):
            continue

        slot_distances = defaultdict(list)

        # 2. Parse all CSV files for the current language
        for file in os.listdir(lang_path):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(lang_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=",")
                header = next(reader, None)  # Skip header

                if not header:
                    continue

                for row in reader:
                    # Ensure the row has the expected 4 columns (lemma, paradigm_slot, target, prediction)
                    if len(row) < 4:
                        continue

                    features = row[1]        # paradigm_slot is now at index 1
                    target_phon = row[2]     # target is now at index 2
                    predicted_phon = row[3]  # prediction is now at index 3

                    # 3. Handle decode failures (kept as a safeguard)
                    if predicted_phon == "DECODE_FAILED":
                        # Penalize failure by treating the prediction as an empty string
                        dist = 1.0
                    else:
                        dist = levenshtein_distance(target_phon, predicted_phon)
                        # normalize it
                        max_len = max(len(target_phon), len(predicted_phon))
                        dist = dist / max_len if max_len > 0 else 0.0

                    slot_distances[features].append(dist)

        if not slot_distances:
            continue

        # 4. Calculate the smart average
        slot_averages = []
        for features, dists in slot_distances.items():
            slot_mean = sum(dists) / len(dists)
            slot_averages.append(slot_mean)

        # Macro-average across all unique paradigm slots
        smart_macro_avg = sum(slot_averages) / len(slot_averages)
        lang_metrics[lang_iso] = smart_macro_avg

    return lang_metrics


##### Detailed summary #######

def calculate_detailed_smart_average_ld(output_dir):
    """
    Calculates the slot-averaged norm. Levenshtein distance per individual run
    (sample_seed, split_seed, fold) for each language.
    """
    detailed_metrics = []

    # Regex to extract seeds and fold from the filename
    # Assumes format: preds_samp{sample_seed}_split{split_seed}_fold{fold}.csv
    pattern = re.compile(r"samp(\d+)_split(\d+)_fold(\d+)\.csv$")

    # 1. Iterate over language folders
    for lang_iso in tqdm(os.listdir(output_dir)):
        lang_path = os.path.join(output_dir, lang_iso)

        if not os.path.isdir(lang_path) or not os.listdir(lang_path):
            continue

        # 2. Parse EACH CSV file separately
        for file in os.listdir(lang_path):
            if not file.endswith(".csv"):
                continue

            match = pattern.search(file)
            if not match:
                continue

            samp_seed, split_seed, fold = match.groups()
            file_path = os.path.join(lang_path, file)

            # Reset the slot tracker for EACH file/run
            slot_distances = defaultdict(list)

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=",")
                header = next(reader, None)

                if not header:
                    continue

                for row in reader:
                    if len(row) < 4:
                        continue

                    features = row[1]        # paradigm_slot is now at index 1
                    target_phon = row[2]     # target is now at index 2
                    predicted_phon = row[3]  # prediction is now at index 3

                    # 3. Handle distance calculation
                    if predicted_phon == "DECODE_FAILED":
                        dist = 1.0
                    else:
                        dist_raw = levenshtein_distance(target_phon, predicted_phon)
                        max_len = max(len(target_phon), len(predicted_phon))
                        # Protect against zero-division if both strings are empty
                        dist = dist_raw / max_len if max_len > 0 else 0.0

                    slot_distances[features].append(dist)

            if not slot_distances:
                continue

            # 4. Calculate the smart average for THIS specific file
            slot_averages = []
            for features, dists in slot_distances.items():
                slot_mean = sum(dists) / len(dists)
                slot_averages.append(slot_mean)

            if slot_averages:
                smart_macro_avg = sum(slot_averages) / len(slot_averages)

                # Append the detailed record
                detailed_metrics.append(
                    {
                        "iso": lang_iso,
                        "sample_seed": int(samp_seed),
                        "split_seed": int(split_seed),
                        "fold": int(fold),
                        "score": smart_macro_avg,
                    }
                )

    return detailed_metrics


# --- Execution ---
OUTPUT_DIR = "judiling_output/"

detailed_results = calculate_detailed_smart_average_ld(OUTPUT_DIR)

# Sort chronologically by language, then sample seed, split seed, and fold
detailed_results.sort(
    key=lambda x: (x["iso"], x["sample_seed"], x["split_seed"], x["fold"])
)

# Output summary filenames adjusted to reflect the new pipeline
with open("judiling_detailed_results.tsv", "w", encoding="utf-8") as f:
    f.write("iso\tsample_seed\tsplit_seed\tfold\tSmart_Avg_Levenshtein\n")
    for res in detailed_results:
        f.write(
            f"{res['iso']}\t{res['sample_seed']}\t{res['split_seed']}\t{res['fold']}\t{res['score']:.4f}\n"
        )


results = calculate_smart_average_ld(OUTPUT_DIR)

with open("judiling_results.tsv", "w", encoding="utf-8") as f:
    f.write("iso\tAvg_Levenshtein\n")
    for iso, average in results.items():
        f.write(f"{iso}\t{average:.4f}\n")