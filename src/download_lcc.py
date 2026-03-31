import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin

# --- Configuration ---
RAW_DATA_DIR = Path("unimorph/raw")
OUTPUT_DIR = Path("lcc_corpora")
BASE_URL = "https://wortschatz.uni-leipzig.de/en/download/"

# Priority order mapped to HTML ids
CATEGORY_PRIORITY = [
    "mixed-typical",
    "web-public",
    "web",
    "newscrawl-public",
    "newscrawl",
    "news-typical",
    "news",
    "wikipedia",
]


def parse_size(size_str):
    """Converts a string like '10K' or '1M' into an integer for accurate comparison."""
    size_str = size_str.strip().upper()
    multiplier = 1
    if size_str.endswith("K"):
        multiplier = 1000
        size_str = size_str[:-1]
    elif size_str.endswith("M"):
        multiplier = 1000000
        size_str = size_str[:-1]

    try:
        return float(size_str) * multiplier
    except ValueError:
        return 0


def download_file(url, dest_path):
    """Streams the file download to avoid loading huge files into memory."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def process_language(isocode):
    print(f"\n--- Processing: {isocode} ---")
    url = f"{BASE_URL}{isocode}"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[{isocode}] Network error: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    # 1. Check for missing data alerts
    danger_alerts = soup.find_all("div", class_="alert alert-danger")
    for alert in danger_alerts:
        text = alert.get_text(strip=True)
        if text in [
            "This language is not known to us.",
            "There are no downloads for the selected language.",
        ]:
            print(f"[{isocode}] Missing data: {text}")
            return

    # 2. Extract available categories
    h3_tags = soup.find_all("h3")
    available_categories = [h3.get("id") for h3 in h3_tags if h3.get("id")]

    if not available_categories:
        print(f"[{isocode}] No valid categories found on page.")
        return

    # 3. Check for "Community only"
    if set(available_categories) == {"community"}:
        print(f"[{isocode}] Skipped: Only 'Community' corpus available.")
        return

    # 4. Find the highest priority category
    selected_category = None
    for priority in CATEGORY_PRIORITY:
        if priority in available_categories:
            selected_category = priority
            break

    if not selected_category:
        print(f"[{isocode}] Skipped: No matching priority categories found.")
        return

    print(f"[{isocode}] Selected category: {selected_category}")

    # 5. Locate the table for the chosen category
    header_tag = soup.find("h3", id=selected_category)
    table = header_tag.find_next_sibling("table")

    if not table or not table.find("tbody"):
        print(f"[{isocode}] Error: Data table not found for {selected_category}.")
        return

    # 6. Parse table rows to find the best candidate
    rows_data = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        try:
            year = int(tds[0].get_text(strip=True))
        except ValueError:
            continue

        country = tds[1].get_text(strip=True)
        is_unnamed = country == ""

        # Extract all available downloads in this row
        links = []
        for a in tds[2].find_all("a", class_="download-link"):
            size_text = a.get_text(strip=True)
            href = a.get("href")
            links.append(
                {"size_raw": size_text, "size_val": parse_size(size_text), "href": href}
            )

        if not links:
            continue

        # Get the largest corpus size for this specific row
        largest_link = max(links, key=lambda x: x["size_val"])

        rows_data.append(
            {"year": year, "is_unnamed": is_unnamed, "best_link": largest_link}
        )

    if not rows_data:
        print(f"[{isocode}] No downloadable files found in the table.")
        return

    # 7. Apply heuristics: Prefer unnamed country (True > False), then latest year
    rows_data.sort(key=lambda x: (x["is_unnamed"], x["year"]), reverse=True)
    best_row = rows_data[0]

    download_href = best_row["best_link"]["href"]
    download_url = urljoin(
        url, download_href
    )  # Handles the // protocol-relative URLs securely
    filename = download_href.split("/")[-1]

    # 8. Create folder and download
    lang_dir = OUTPUT_DIR / isocode
    lang_dir.mkdir(parents=True, exist_ok=True)

    dest_path = lang_dir / filename

    if dest_path.exists():
        print(f"[{isocode}] File {filename} already exists. Skipping download.")
        return

    print(
        f"[{isocode}] Downloading {filename} (Size: {best_row['best_link']['size_raw']}) from {best_row['year']}..."
    )
    download_file(download_url, dest_path)
    print(f"[{isocode}] Success: Downloaded to {dest_path}")


def main():
    if not RAW_DATA_DIR.exists():
        print(f"Directory {RAW_DATA_DIR} not found. Please ensure it exists.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Iterate over all csv files in unimorph/raw
    for filepath in RAW_DATA_DIR.glob("*.csv"):
        # Extract 'deu' from 'deu.csv'
        isocode = filepath.stem
        process_language(isocode)


if __name__ == "__main__":
    main()
