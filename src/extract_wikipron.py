import os
import pandas as pd
import requests

# --- Configuration ---
UNIMORPH_SUMMARY = "unimorph/summary.csv"
WIKIPRON_SUMMARY = "wikipron/summary.tsv"
UNIMORPH_RAW_DIR = "unimorph/raw"
OUTPUT_DIR = "unimorph/phon"
WIKIPRON_BASE_URL = "https://raw.githubusercontent.com/CUNY-CL/wikipron/refs/heads/master/data/scrape/tsv/{}"


def load_summaries():
    """Loads and filters the summary files."""
    # Load Unimorph and filter by criteria
    unimorph_df = pd.read_csv(UNIMORPH_SUMMARY)
    unimorph_filtered = unimorph_df[
        (unimorph_df["num_lemmas"] >= 80) & (unimorph_df["avg_cells_per_lemma"] >= 4)
    ]
    valid_isos = set(unimorph_filtered["iso"].unique())

    # Load Wikipron summary
    # Expected columns based on the prompt description
    wikipron_cols = [
        "file_name",
        "iso",
        "iso_lang",
        "name",
        "wiktionary_name",
        "script",
        "dialect",
        "filtered",
        "broad_narrow",
        "num_entries",
    ]
    wikipron_df = pd.read_csv(
        WIKIPRON_SUMMARY, sep="\t", header=None, names=wikipron_cols
    )

    return valid_isos, wikipron_df


def fetch_wikipron_dict(file_name):
    """Downloads a Wikipron TSV and returns a dictionary of {word: pronunciation}."""
    url = WIKIPRON_BASE_URL.format(file_name)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        pron_dict = {}
        for line in response.text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                word, pron = parts[0].strip(), parts[1].strip()
                # If a word has multiple pronunciations, we keep the first one
                if word not in pron_dict:
                    pron_dict[word] = pron
        return pron_dict
    except requests.RequestException as e:
        print(f"  [!] Failed to download {file_name}: {e}")
        return {}


def process_language(iso, wikipron_df):
    """Processes a single language: loads Unimorph, finds the best dictionary, and maps IPAs."""
    unimorph_file = os.path.join(UNIMORPH_RAW_DIR, f"{iso}.csv")
    output_file = os.path.join(OUTPUT_DIR, f"{iso}.csv")

    if not os.path.exists(unimorph_file):
        print(f"[-] Unimorph file missing for {iso}, skipping.")
        return

    # Load Unimorph data
    # Assuming columns: lemma, form, paradigm_slot
    df = pd.read_csv(unimorph_file)
    if "form" not in df.columns:
        print(f"[-] 'form' column missing in {iso}.csv, skipping.")
        return

    unique_forms = set(df["form"].dropna().astype(str))

    # Find candidate Wikipron files for this ISO
    candidates = wikipron_df[wikipron_df["iso"] == iso]["file_name"].tolist()

    if not candidates:
        print(f"[-] No Wikipron candidates found for {iso}. Saving empty column.")
        return

    print(f"[+] Evaluating {len(candidates)} Wikipron dictionaries for {iso}...")

    best_overlap = -1
    best_dict = {}
    best_file = None

    # Heuristic: Download all candidates and check which yields the highest intersection
    for file_name in candidates:
        candidate_dict = fetch_wikipron_dict(file_name)
        if not candidate_dict:
            continue

        overlap = len(unique_forms.intersection(candidate_dict.keys()))

        if overlap > best_overlap:
            best_overlap = overlap
            best_dict = candidate_dict
            best_file = file_name

    if best_dict and best_overlap > 0:
        print(f"  -> Selected '{best_file}' with {best_overlap} matches.")
        # Map forms to transcriptions, leaving missing ones as empty strings
        df["phonemic_form"] = df["form"].astype(str).map(best_dict).fillna("")
    else:
        print(f"  -> No overlap found in any scripts for {iso}. Saving empty column.")
        return

    df.to_csv(output_file, index=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading summaries...")
    valid_isos, wikipron_df = load_summaries()

    print(f"Found {len(valid_isos)} valid languages in Unimorph.")

    for count, iso in enumerate(valid_isos, 1):
        print(f"\n({count}/{len(valid_isos)}) Processing {iso}...")
        process_language(iso, wikipron_df)

    print("\nPipeline complete. Results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
