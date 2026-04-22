import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import pandas as pd
import requests
import io
from tqdm import tqdm

WIKIPRON_SUMMARY_URL = "https://raw.githubusercontent.com/CUNY-CL/wikipron/refs/heads/master/data/scrape/summary.tsv"
WIKIPRON_COLS = [
    "file_name",
    "iso",
    "iso_lang",
    "name",
    "script",
    "dialect",
    "filtered",
    "broad_narrow",
    "num_entries",
]


def load_trained_g2p_model(model_path):
    """
    Loads the fine-tuned ByT5 model and tokenizer from a local directory.

    Args:
        model_path (str): Path to the directory containing pytorch_model.bin/model.safetensors
                          and tokenizer configs.

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"--- Loading model from {model_path} ---")

    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load the model.
        # Using torch_dtype="auto" will attempt to load it in the precision it was saved (e.g., bf16)
        # device_map="auto" efficiently handles multi-GPU or CPU offloading
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )

        print("Successfully loaded model and tokenizer.")
        return model, tokenizer

    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return None, None


# --- 1. Check which languages the model was trained on ---
def get_trained_languages(summary_url, cols):
    """
    Fetches the Wikipron summary to extract the exact set of ISO codes
    used during the 'Broad' transcription training phase.
    """
    summary_resp = requests.get(summary_url)
    summary_resp.raise_for_status()

    df = pd.read_csv(io.StringIO(summary_resp.text), sep="\t", names=cols, header=None)
    # Match the exact filtering logic from training
    df_broad = df[df["broad_narrow"] == "Broad"]
    trained_isos = set(df_broad["iso"].unique())

    return trained_isos


# --- 2. Filter UniMorph data based on frequency/lemma count ---
def filter_unimorph_data(df):
    """
    Filters the DataFrame for the top 80 lemmas by frequency.
    If frequency data is missing, allows the file if total unique lemmas < 200.
    """
    # Drop rows where lemma or form is missing to prevent downstream errors
    df = df.dropna(subset=["lemma", "form"])

    has_freq_data = "lemma_freq" in df.columns and not df["lemma_freq"].isna().all()

    if has_freq_data:
        # Extract unique lemmas and their frequencies, sort, and grab top 80
        lemma_freqs = df[["lemma", "lemma_freq"]].drop_duplicates(subset=["lemma"])
        # Handle cases where frequencies might be strings/objects by casting to float
        lemma_freqs["lemma_freq"] = lemma_freqs["lemma_freq"].astype(float)
        top_80_lemmas = lemma_freqs.nlargest(80, "lemma_freq")["lemma"]

        # Filter original dataframe to only keep rows containing these top 80 lemmas
        return df[df["lemma"].isin(top_80_lemmas)].copy()

    else:
        # Fallback for languages missing frequency data
        unique_lemma_count = df["lemma"].nunique()
        if unique_lemma_count < 200:
            return df.copy()
        else:
            # Exceeds 200 lemmas and has no freq data -> discard
            return pd.DataFrame()


# --- 3. Run Inference ---
def generate_phonemes(forms, iso, model, tokenizer, batch_size=128):
    """
    Takes a list of unique forms, formats them with the language ISO,
    and runs batch inference through the fine-tuned ByT5 model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    phonemized_results = []

    # Process in batches to avoid OOM errors
    for i in range(0, len(forms), batch_size):
        batch_forms = forms[i : i + batch_size]
        # Format identically to training phase: "g2p | {iso}: {word}"
        inputs = [f"g2p | {iso}: {form}" for form in batch_forms]

        encodings = tokenizer(
            inputs, padding=True, truncation=True, max_length=96, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**encodings, max_length=64)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        phonemized_results.extend([pred.strip() for pred in decoded])

    return phonemized_results


# --- 4. Main Execution Loop ---
def process_unimorph_files(raw_dir, processed_dir, model, tokenizer):
    os.makedirs(processed_dir, exist_ok=True)

    print("Fetching list of trained languages...")
    trained_isos = get_trained_languages(WIKIPRON_SUMMARY_URL, WIKIPRON_COLS)

    # Get all csv files in the raw directory
    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

    for filename in tqdm(files, desc="Processing UniMorph files"):
        # Assuming filename is precisely the ISO code (e.g., 'afr.csv' -> 'afr')
        iso = filename.replace(".csv", "")

        # Check against training languages
        if iso not in trained_isos:
            continue

        filepath = os.path.join(raw_dir, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"\n[!] Failed to read {filename}: {e}")
            continue

        # Filter the DataFrame
        filtered_df = filter_unimorph_data(df)

        # If DataFrame is empty, it means it failed the filter conditions
        # (e.g., no freq data and > 200 lemmas). We skip it.
        if filtered_df.empty:
            continue

        # Extract UNIQUE forms to minimize inference overhead
        unique_forms = filtered_df["form"].unique().tolist()

        # Run batch inference
        phonemes_list = generate_phonemes(
            unique_forms, iso, model, tokenizer, batch_size=256
        )

        # Create a dictionary mapping the form to its generated phoneme
        form_to_phoneme = dict(zip(unique_forms, phonemes_list))

        # Map the dictionary back to the respective column in the filtered DataFrame
        filtered_df["phonemic_form"] = filtered_df["form"].map(form_to_phoneme)

        # Save processed file
        output_path = os.path.join(processed_dir, filename)
        filtered_df.to_csv(output_path, index=False)


# Kick off the processing pipeline
RAW_DIR = "unimorph/raw"
PROCESSED_DIR = "unimorph/processed"
CHECKPOINT_PATH = "checkpoint-3900/"

print("Starting UniMorph data processing loop...")
model, tokenizer = load_trained_g2p_model(CHECKPOINT_PATH)
process_unimorph_files(RAW_DIR, PROCESSED_DIR, model, tokenizer)
print(f"Done! Processed files saved to {PROCESSED_DIR}/")
