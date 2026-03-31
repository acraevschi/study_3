import os
import csv
from tqdm import tqdm
import xml.etree.ElementTree as ET

### Need to think abot irregulars/strong verbs and how they are treated in Low German tradition.


def extract_mlg_verbs(input_dir, output_csv):
    # Set of all ReN verb PoS tags based on the manual
    verb_tags = {
        "VVINF",
        "VVFIN",
        "VVIMP",
        "VVPS",
        "VVPP",
        ### auxiliaries
        # "VAINF",
        # "VAFIN",
        # "VAIMP",
        # "VAPS",
        # "VAPP",
        ### modals
        # "VMINF",
        # "VMFIN",
        # "VMIMP",
        # "VMPS",
        # "VMPP",
    }

    # Open the output CSV file
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write the header row
        writer.writerow(
            [
                "document_name",
                "document_date",
                "token_id",
                "form_trans",
                "form_utf",
                "pos_tag",
                "morphology",
                "lemma",
                "lemma_simple",
                "lemma_wsd",
            ]
        )

        # Iterate through all XML files in the directory
        for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
            if filename.endswith(".xml"):
                file_path = os.path.join(input_dir, filename)

                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                except ET.ParseError:
                    print(f"Error parsing {filename}. Skipping.")
                    continue

                # 1. Extract Document Metadata
                # Try to get the clean name from cora-header, fallback to filename
                cora_header = root.find(".//cora-header")
                doc_name = (
                    cora_header.attrib.get("name", filename)
                    if cora_header is not None
                    else filename
                )

                # Parse the raw text header to find the date
                doc_date = "Unknown"
                header_node = root.find(".//header")
                if header_node is not None and header_node.text:
                    for line in header_node.text.split("\n"):
                        if line.startswith("date_ReN:"):
                            doc_date = line.replace("date_ReN:", "").strip()
                            break

                # 2. Extract Verb Tokens
                for token in root.findall(".//token"):
                    token_id = token.attrib.get("id", "")
                    anno = token.find("anno")

                    if anno is not None:
                        pos_node = anno.find("pos")

                        if pos_node is not None:
                            pos_tag = pos_node.attrib.get("tag", "")

                            # Check if the tag is in our verb set
                            if pos_tag in verb_tags:
                                # Get the forms
                                form_trans = anno.attrib.get("trans", "")
                                form_utf = anno.attrib.get("utf", "")

                                # Get morphology
                                morph_node = anno.find("morph")
                                morph = (
                                    morph_node.attrib.get("tag", "")
                                    if morph_node is not None
                                    else ""
                                )

                                # Get lemmas (base, simple, wsd)
                                lemma_node = anno.find("lemma")
                                lemma = (
                                    lemma_node.attrib.get("tag", "")
                                    if lemma_node is not None
                                    else ""
                                )

                                lemma_simple_node = anno.find("lemma_simple")
                                lemma_simple = (
                                    lemma_simple_node.attrib.get("tag", "")
                                    if lemma_simple_node is not None
                                    else ""
                                )

                                lemma_wsd_node = anno.find("lemma_wsd")
                                lemma_wsd = (
                                    lemma_wsd_node.attrib.get("tag", "")
                                    if lemma_wsd_node is not None
                                    else ""
                                )

                                # Write the row
                                writer.writerow(
                                    [
                                        doc_name,
                                        doc_date,
                                        token_id,
                                        form_trans,
                                        form_utf,
                                        pos_tag,
                                        morph,
                                        lemma,
                                        lemma_simple,
                                        lemma_wsd,
                                    ]
                                )

    print(f"Extraction complete! Verbs saved to {output_csv}")


# extract_mlg_verbs("CorA-ReN-XML_1.1/ReN_anno_2021-01-06", "extracted_verbs.csv")
