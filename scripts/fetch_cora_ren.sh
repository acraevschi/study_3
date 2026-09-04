#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fetch the CorA-XML release of the Reference Corpus Middle Low German /
# Low Rhenish (ReN), version 1.1, into germanic/cora_ren_xml_1.1/.
# ---------------------------------------------------------------------------
# The corpus is ~694 MB, so it is not kept in this repository. It is CC BY 4.0,
# hosted by the Hamburg Centre for Language Corpora under a DOI.
#
#   Barteld, Fabian, Katharina Dreessen, Sarah Ihden & Ingrid Schroder (2021).
#   Reference Corpus Middle Low German / Low Rhenish (1200-1650), version 1.1.
#   Universitat Hamburg. https://doi.org/10.25592/uhhfdm.9195
#
# Only the CorA-XML "anno" directory is actually read by
# src/low_german_extraction.py; the "trans" directory is transcription-only.
#
# The repository does not expose a stable direct-download URL for the archive,
# so this script cannot fully automate the fetch: the landing page requires you
# to accept the licence before the download link appears. It opens the record
# and tells you where to unpack.
#
# Usage:  bash scripts/fetch_cora_ren.sh
# ---------------------------------------------------------------------------
set -euo pipefail

DOI_URL="https://doi.org/10.25592/uhhfdm.9195"
TARGET="$(cd "$(dirname "$0")/.." && pwd)/germanic/cora_ren_xml_1.1"

if [ -d "$TARGET/ReN_anno_2021-01-06" ]; then
  n=$(find "$TARGET/ReN_anno_2021-01-06" -name '*.xml' | wc -l | tr -d ' ')
  echo "Corpus already present: $TARGET ($n annotated XML documents)."
  exit 0
fi

cat <<EOF
The ReN corpus is not in this repository (694 MB, CC BY 4.0, available by DOI).

  1. Open   $DOI_URL
  2. Download the CorA-XML release, version 1.1
  3. Unpack it so that these two directories exist:

       $TARGET/ReN_anno_2021-01-06/    (161 annotated XML documents)
       $TARGET/ReN_trans_2021-01-06/   ( 74 transcription-only documents)

  4. Then run:  python3 src/low_german_extraction.py

Only step 3's "anno" directory is read. The extraction output,
germanic/extracted_verbs.csv (183,450 verb tokens), IS committed to this
repository -- so if you only need the verb data, you do not need the corpus.
EOF

if command -v open >/dev/null 2>&1; then
  read -r -p "Open the record page now? [y/N] " reply
  [ "${reply:-n}" = "y" ] && open "$DOI_URL"
fi
