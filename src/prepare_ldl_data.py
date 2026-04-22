import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def main():
    pop_df = pd.read_csv("ethnologue_population_data.csv")
    pop_df.dropna(subset=["Population", "L2prop"], inplace=True)

    K = 5
    NUM_SEEDS = 2
    TARGET_LEMMAS = 80
    CELLS_PER_LEMMA = 4
    DATA_DIR = "unimorph/processed/"
    OUTPUT_DIR = "judiling_input/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    langs_to_process = []
    lang_data = {}

    # --- 1. DATA LOADING AND FILTERING ---
    for file in tqdm(os.listdir(DATA_DIR), desc="Loading languages"):
        if not file.endswith(".csv"):
            continue

        lang = file.replace(".csv", "")
        if lang not in pop_df["ISO"].to_list():
            continue

        lines = [l.strip().split(",") for l in open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") if l.strip()]
        
        if lines and lines[0][0] == "lemma":
            lines = lines[1:]  # Skip header

        lemma_dict = defaultdict(list)
        for row in lines:
            if len(row) >= 6:
                lemma_dict[row[0]].append(row)

        if not lemma_dict:
            continue

        # NEW LOGIC: Calculate average cells per lemma for the entire language
        total_cells = sum(len(forms) for forms in lemma_dict.values())
        avg_cells_per_lemma = total_cells / len(lemma_dict)

        # FILTER: Keep language only if avg cells >= 4 AND it meets the minimum lemma count
        if avg_cells_per_lemma < CELLS_PER_LEMMA or len(lemma_dict) < TARGET_LEMMAS:
            continue

        langs_to_process.append(lang)
        lang_data[lang] = lemma_dict

    langs_to_process = sorted(langs_to_process)
    print(f"Loaded {len(langs_to_process)} valid languages.")

    # --- 2. SAMPLING & FOLD GENERATION ---
    for lang in tqdm(langs_to_process, desc="Processing lang splits"):
        output_dir_lang = os.path.join(OUTPUT_DIR, lang)
        os.makedirs(output_dir_lang, exist_ok=True)

        all_lemmas = list(lang_data[lang].keys())

        for sample_seed in range(NUM_SEEDS):
            random.seed(sample_seed)
            # Sample up to TARGET_LEMMAS
            sampled_lemmas = random.sample(all_lemmas, min(TARGET_LEMMAS, len(all_lemmas)))

            for split_seed in range(NUM_SEEDS):
                text = []
                for lemma in sampled_lemmas:
                    text.extend(lang_data[lang][lemma])

                batch_inds = np.arange(len(text))
                if len(batch_inds) < 2:
                    continue

                np.random.seed(split_seed)
                np.random.shuffle(batch_inds)
                folds = np.array_split(batch_inds, K)

                fold_assignments = {idx: fold_idx for fold_idx, fold_inds in enumerate(folds) for idx in fold_inds}

                # --- 3. UNPACKING SEMANTICS FOR JUDILING ---
                out_rows = []
                for i, row in enumerate(text):
                    lemma, form, paradigm_slot, form_freq, lemma_freq, phonemic_form = row[:6]
                    fold = fold_assignments[i]

                    # Split UniMorph tags (e.g. "V;PRS;3;SG" -> ["V", "PRS", "3", "SG"])
                    tags = paradigm_slot.split(";")
                    if len(tags) > 10:
                        print(f"Warning: More than 10 tags for {lang} {lemma} {form}. Truncating to 10.")
                        tags = tags[:10]

                    # Pad tags to standard 10 columns so Julia has a fixed Semantic format
                    padded_tags = (tags + [""] * 10)[:10]

                    out_row = [lemma, form, paradigm_slot, form_freq, lemma_freq, phonemic_form, fold] + padded_tags
                    out_rows.append(out_row)

                cols = ["lemma", "form", "paradigm_slot", "form_freq", "lemma_freq", "phonemic_form", "Fold",
                        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
                
                df_out = pd.DataFrame(out_rows, columns=cols)
                out_filename = os.path.join(output_dir_lang, f"data_samp{sample_seed}_split{split_seed}.csv")
                df_out.to_csv(out_filename, index=False)

if __name__ == "__main__":
    main()