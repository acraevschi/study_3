import os
import csv
import tarfile
import re
import glob
from collections import defaultdict


def get_corpus_size(filename):
    """Extracts corpus size from the filename (e.g., 100K -> 100000, 1M -> 1000000)."""
    match = re.search(r"(\d+)([KM])", filename, re.IGNORECASE)
    if match:
        value = int(match.group(1))
        unit = match.group(2).upper()
        if unit == "K":
            return value * 1_000
        elif unit == "M":
            return value * 1_000_000
    return None


def process_corpora(unimorph_dir="unimorph/raw", lcc_dir="lcc_corpora"):
    # Iterate over all csv files in the unimorph directory
    for um_file in glob.glob(os.path.join(unimorph_dir, "*.csv")):
        lang_code = os.path.basename(um_file).replace(".csv", "")
        lcc_lang_dir = os.path.join(lcc_dir, lang_code)

        # 1. Validation: Check if LCC directory and tar file exist
        if not os.path.exists(lcc_lang_dir):
            print(f"Skipping {lang_code}: LCC directory not found.")
            continue

        tar_files = glob.glob(os.path.join(lcc_lang_dir, "*.tar.gz"))
        if not tar_files:
            print(f"Skipping {lang_code}: No .tar.gz file found.")
            continue

        tar_path = tar_files[0]
        corpus_size = get_corpus_size(os.path.basename(tar_path))

        if not corpus_size:
            print(f"Skipping {lang_code}: Could not infer corpus size from {tar_path}")
            continue

        # 2. Extract frequencies from the LCC archive
        form_normalized_freqs = {}
        with tarfile.open(tar_path, "r:gz") as tar:
            # Find the file ending in '-words.txt'
            words_member = next(
                (m for m in tar.getmembers() if m.name.endswith("-words.txt")), None
            )
            if not words_member:
                print(
                    f"Skipping {lang_code}: No '-words.txt' found inside the archive."
                )
                continue

            # Read line by line directly from the archive
            with tar.extractfile(words_member) as f:
                for line in f:
                    parts = line.decode("utf-8", errors="ignore").strip().split("\t")
                    if len(parts) >= 3:
                        form = parts[1]
                        absolute_freq = int(parts[2])
                        # Normalize by corpus size
                        form_normalized_freqs[form] = absolute_freq / corpus_size

        # 3. Process UniMorph CSV and map frequencies
        rows = []
        lemma_form_freqs = defaultdict(list)

        with open(um_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            for row in reader:
                # Check if the word is a verb (assuming paradigm slot starts with 'V')
                if row.get("paradigm_slot", "").startswith("V"):
                    form = row["form"]
                    lemma = row["lemma"]

                    # Get frequency (default to 0.0 if form is missing in corpus)
                    freq = form_normalized_freqs.get(form, 0.0)
                    row["form_freq"] = freq
                    lemma_form_freqs[lemma].append(freq)
                else:
                    # Non-verbs get blank/zero values to maintain CSV structure
                    row["form_freq"] = ""

                rows.append(row)

        # 4. Calculate the average frequency per lemma
        lemma_avg_freq = {
            lemma: sum(freqs) / len(freqs)
            for lemma, freqs in lemma_form_freqs.items()
            if freqs
        }

        # 5. Write back to the UniMorph file
        new_fieldnames = fieldnames + ["form_freq", "lemma_freq"]
        temp_file = um_file + ".tmp"

        with open(temp_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()

            for row in rows:
                lemma = row.get("lemma")
                if (
                    row.get("paradigm_slot", "").startswith("V")
                    and lemma in lemma_avg_freq
                ):
                    row["lemma_freq"] = lemma_avg_freq[lemma]
                else:
                    row["lemma_freq"] = ""

                writer.writerow(row)

        # Safely overwrite the original file with the new data
        os.replace(temp_file, um_file)
        print(f"Successfully processed {lang_code}.")


if __name__ == "__main__":
    # You can customize the paths here if your script is in a different directory
    process_corpora(unimorph_dir="unimorph/raw", lcc_dir="lcc_corpora")
