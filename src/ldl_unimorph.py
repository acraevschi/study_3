import numpy as np
import csv
from tqdm import tqdm
import os
import random
from collections import defaultdict


def get_next_ngram(s, ngram_types):
    """get all ngrams that start with the last two characters of the input ngram"""
    return [s_ for s_ in ngram_types if s_[:2] == s[1:]]


K = 10
TARGET_LEMMAS = 80
DATA_DIR = "unimorph/"
OUTPUT_DIR = "ldl_results/"
CELLS_LEMMA = 4.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 0. DATA FILTERING & SELECTION ---
langs_to_process = []
lang_data = {}

for file in tqdm(os.listdir(DATA_DIR), desc="Filtering languages"):
    if not file.endswith(".csv"):
        continue

    lang = file.replace(".csv", "")
    output_dir_lang = OUTPUT_DIR + lang + "/"
    os.makedirs(output_dir_lang, exist_ok=True)

    # Read the data, accounting for the new format: lemma (0), form (1), slot (2)
    lines = [
        l.strip().split(",")
        for l in open(os.path.join(DATA_DIR, file), "r", encoding="utf-8")
        if l.strip()
    ]

    # Map lemmas to their rows to check our conditions
    lemma_dict = defaultdict(list)
    for row in lines:
        if len(row) == 3:
            lemma_dict[row[0]].append(row)

    num_lemmas = len(lemma_dict)

    if num_lemmas == 0:
        continue

    avg_cells = sum(len(forms) for forms in lemma_dict.values()) / num_lemmas

    # Condition: >= 80 lemmas AND >= 4 cells per lemma on average
    if num_lemmas >= TARGET_LEMMAS and avg_cells >= CELLS_LEMMA:
        langs_to_process.append(lang)
        lang_data[lang] = lemma_dict

langs_to_process = sorted(langs_to_process)
print(f"Filtered down to {len(langs_to_process)} languages.")

# --- 1. PROCESSING ---
limit = 5
count = 0
for lang in tqdm(langs_to_process, desc="Processing langs"):

    output_dir_lang = OUTPUT_DIR + lang + "/"
    os.makedirs(output_dir_lang, exist_ok=True)

    if count == limit:
        break
    for sample_seed in range(3):
        for split_seed in range(3):

            # --- Sample exactly 80 lemmas to control vocabulary size ---
            random.seed(sample_seed)
            all_lemmas = list(lang_data[lang].keys())
            sampled_lemmas = random.sample(all_lemmas, TARGET_LEMMAS)

            # Flatten back into a text list for the sampled lemmas only
            text = []
            for lemma in sampled_lemmas:
                text.extend(lang_data[lang][lemma])

            K_ngram = 3
            ngrams = []
            for l in text:
                l_ = []
                # Form is now at index 1
                w = ["#"] + list(l[1]) + ["$"]
                for i in range(len(w) - (K_ngram - 1)):
                    l_.append(tuple(w[i : i + K_ngram]))
                ngrams.append(l_)

            ngram_types = sorted(set([s for row in ngrams for s in row]))
            ngram_to_idx = {s: i for i, s in enumerate(ngram_types)}

            # Construct Form Matrix (C)
            ngram_bin = np.zeros([len(text), len(ngram_types)])
            for i, row in enumerate(ngrams):
                for s in row:
                    ngram_bin[i, ngram_to_idx[s]] += 1

            # --- 2. LDL CONTINUOUS SEMANTIC SIMULATION ---
            unique_lemmas = sorted(list(set([l[0] for l in text])))
            # Features are now at index 2
            unique_features = sorted(
                list(set([f for l in text for f in l[2].split(";")]))
            )

            vec_dim = len(ngram_types)

            # Seed for consistent vector generation across runs
            np.random.seed(97)

            lemma_vectors = {
                lemma: np.random.normal(0.0, 4.0, vec_dim) for lemma in unique_lemmas
            }
            feature_vectors = {
                feat: np.random.normal(0.0, 4.0, vec_dim) for feat in unique_features
            }

            # Construct Semantic Matrix (S)
            sem_continuous = np.zeros([len(text), vec_dim])
            for i, l in enumerate(text):
                lemma = l[0]
                features = l[2].split(";")  # Features at index 2

                base_vec = np.copy(lemma_vectors[lemma])
                for feat in features:
                    if feat in feature_vectors:
                        base_vec += feature_vectors[feat]

                noise = np.random.normal(0.0, 1.0, vec_dim)
                sem_continuous[i, :] = base_vec + noise

            # --- 3. FOLD PROCESSING ---
            batch_inds = np.arange(len(text))
            np.random.seed(split_seed)
            np.random.shuffle(batch_inds)

            fold_inds = list(range(0, len(batch_inds), int(len(batch_inds) / K))) + [
                len(batch_inds)
            ]

            for fold in tqdm(range(K), desc=f"Folds for {lang}", leave=False):
                test_inds = list(batch_inds[fold_inds[fold] : fold_inds[fold + 1]])
                train_inds = [i for i in batch_inds if i not in test_inds]

                ngram_bin_train = ngram_bin[np.array(train_inds),]
                sem_train = sem_continuous[np.array(train_inds),]

                ngram_bin_test = ngram_bin[np.array(test_inds),]
                sem_test = sem_continuous[np.array(test_inds),]

                # Compute Production Matrix (G): Mapping semantics to forms
                W = np.linalg.lstsq(sem_train, ngram_bin_train, rcond=None)

                output_filename = os.path.join(
                    output_dir_lang,
                    f"ldl_{lang}_samp{sample_seed}_split{split_seed}_fold{fold}.tsv",
                )

                with open(output_filename, "w", encoding="utf-8") as f:
                    print("lemma\tform\tfeatures\tpredicted_form", file=f)

                    for i in range(sem_test.shape[0]):
                        try:
                            w = np.dot(sem_test[i, :], W[0])
                            weights = {s: w[idx] for s, idx in ngram_to_idx.items()}

                            start = sorted(
                                [s for s in ngram_types if s[0] == "#"],
                                key=lambda x: weights[x],
                            )[-1]

                            output = [start]
                            curr_ngram = start

                            # Greedy decoding
                            while True:
                                candidates = [
                                    s
                                    for s in ngram_types
                                    if s[:2] == curr_ngram[1:] and s not in output
                                ]

                                if not candidates:
                                    break

                                weights_cand = [w[ngram_to_idx[s]] for s in candidates]
                                winner = candidates[np.argmax(weights_cand)]
                                output.append(winner)
                                curr_ngram = winner

                                if winner[-1] == "$":
                                    break

                            # text[test_inds[i]] contains [lemma, form, features]
                            predicted = "".join([s[1] for s in output])
                            line_to_print = text[test_inds[i]] + [predicted]
                            print("\t".join(line_to_print), file=f)

                        except Exception:
                            # Safely handle dead-ends in greedy decoding
                            line_to_print = text[test_inds[i]] + ["DECODE_FAILED"]
                            print("\t".join(line_to_print), file=f)
    count += 1


def levenshtein_distance(s1, s2):
    """Calculates the minimum edit distance between two strings."""
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
    Calculates the slot-averaged Levenshtein distance for each language.
    """
    lang_metrics = {}

    # 1. Iterate over language folders
    for lang_iso in os.listdir(output_dir):
        lang_path = os.path.join(output_dir, lang_iso)

        # Skip if it's not a directory or if it's completely empty
        if not os.path.isdir(lang_path) or not os.listdir(lang_path):
            continue

        slot_distances = defaultdict(list)

        # 2. Parse all TSV files for the current language
        for file in os.listdir(lang_path):
            if not file.endswith(".tsv"):
                continue

            file_path = os.path.join(lang_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader, None)  # Skip header

                if not header:
                    continue

                for row in reader:
                    # Ensure the row has lemma, form, features, and predicted_form
                    if len(row) < 4:
                        continue

                    target_form = row[1]
                    features = row[2]
                    predicted_form = row[3]

                    # 3. Handle decode failures
                    if predicted_form == "DECODE_FAILED":
                        # Penalize failure by treating the prediction as an empty string
                        dist = 1.0
                    else:
                        dist = levenshtein_distance(target_form, predicted_form)
                        # normalize it
                        dist = dist / max(len(target_form), len(predicted_form))

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


# --- Execution Example ---
OUTPUT_DIR = "ldl-cont_sem-results/"
results = calculate_smart_average_ld(OUTPUT_DIR)

with open("ldl_summary_results.tsv", "w", encoding="utf-8") as f:
    f.write("Language\tSmart_Avg_Levenshtein\n")
    for lang, score in sorted(results.items(), key=lambda item: item[1]):
        f.write(f"{lang}\t{score:.4f}\n")
