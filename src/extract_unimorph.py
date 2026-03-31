import os
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
from pathlib import Path


def scrape_unimorph_verbs():
    # 1. Create the target directory if it doesn't exist
    output_dir = "unimorph/raw"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Fetch and parse the main UniMorph index page
    print("Fetching UniMorph index...")
    url = "https://unimorph.github.io/"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", id="annotated")
    rows = table.find_all("tr")

    # 3. Iterate through the table rows
    for row in rows:
        tds = row.find_all("td")

        # Skip headers and the collapsible detail rows (which have fewer tds and colspans)
        if len(tds) < 10:
            continue

        # Extract ISO code (Index 1) and Verbs column (Index 5)
        iso_code = tds[1].text.strip()

        # something is wrong with Bulgarian's encoding
        if iso_code == "bul":
            continue
        verbs_cell = tds[5].text.strip()

        # Check if the verbs column contains the checkmark (Unicode U+2714)
        if "✔" in verbs_cell or "\u2714" in verbs_cell:
            print(f"Processing {iso_code}...")

            # Construct the raw download URL (not the blob URL)
            special_treatment = ["uzb", "kaz"]
            if iso_code not in special_treatment:
                raw_url = f"https://raw.githubusercontent.com/unimorph/{iso_code}/master/{iso_code}"
            elif iso_code == "kaz":
                raw_url = f"https://raw.githubusercontent.com/unimorph/{iso_code}/master/kaz.sm"
            elif iso_code == "uzb":
                raw_url = f"https://raw.githubusercontent.com/unimorph/{iso_code}/master/uzb_verbs"

            data_response = requests.get(raw_url)

            # Error handling: some older or newer repos might use 'main' instead of 'master'
            if data_response.status_code != 200 and iso_code == "fin":
                raw_url_1 = "https://raw.githubusercontent.com/unimorph/fin/refs/heads/master/fin.1"
                data_response_1 = requests.get(raw_url_1)
                raw_url_2 = "https://raw.githubusercontent.com/unimorph/fin/refs/heads/master/fin.2"
                data_response_2 = requests.get(raw_url_2)
                data_response = data_response_1.text + data_response_2.text

            elif data_response.status_code != 200:
                raw_url = f"https://raw.githubusercontent.com/unimorph/{iso_code}/main/{iso_code}"
                data_response = requests.get(raw_url)
                if data_response.status_code != 200:
                    print(f"  [!] Skipping {iso_code}: Could not find data file.")
                    continue

            lines = (
                data_response.strip().split("\n")
                if type(data_response) == str
                else data_response.text.strip().split("\n")
            )

            # 4. Filter for verbs and save to CSV
            output_file = os.path.join(output_dir, f"{iso_code}.csv")

            with open(output_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                # Write our own custom header
                writer.writerow(["lemma", "form", "paradigm_slot"])

                verb_count = 0
                for line in lines:
                    parts = line.split("\t")
                    # Ensure the line is well-formed (at least 3 columns)
                    if len(parts) >= 3:
                        lemma, form, paradigm = parts[0], parts[1], parts[2]

                        # UniMorph paradigm tags for verbs typically start with 'V;' or are exactly 'V'
                        if paradigm.startswith("V;") or paradigm == "V":
                            writer.writerow([lemma, form, paradigm])
                            verb_count += 1

            print(f"  -> Saved {verb_count} verb entries.")

    print(f"\nScraping complete! All files saved to the {output_dir} directory.")

    return output_dir


def process_unimorph_directory(dir_path, save_file=False):
    # Path to the directory containing the csv files
    path = Path(dir_path)

    # List to store results for each language
    results = []

    # Iterate over all .csv files in the directory
    for file_path in path.glob("*.csv"):
        # The ISO code is the filename stem (e.g., 'spa' from 'spa.csv')
        iso_code = file_path.stem

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # 1. Total number of unique lemmas
            num_lemmas = df["lemma"].nunique()

            # 2. Average number of cells (rows) per lemma
            # We group by lemma and count the rows for each, then take the mean
            avg_cells_per_lemma = df.groupby("lemma").size().mean()

            # Append the data to our list
            results.append(
                {
                    "iso": iso_code,
                    "num_lemmas": num_lemmas,
                    "avg_cells_per_lemma": round(avg_cells_per_lemma, 2),
                }
            )

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Create the final DataFrame
    summary_df = pd.DataFrame(results)
    summary_df.sort_values("iso", inplace=True)

    if save_file:
        summary_path = dir_path.rsplit("/", 1)[0] + "/summary.csv"
        summary_df.to_csv(summary_path, index=False)

    # Sort by ISO code for readability
    return summary_df


if __name__ == "__main__":
    output_dir = scrape_unimorph_verbs()
    process_unimorph_directory(output_dir, save_file=True)
