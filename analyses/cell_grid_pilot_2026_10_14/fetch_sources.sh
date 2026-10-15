#!/usr/bin/env bash
# Fetch the extra pilot sources into external/ (git-ignored) and print their SHA-256.
# Approved by the project owner on 2026-10-14. Run from the repository root.
#   Morph-it! (Zanchetta & Baroni 2005), CC BY-SA 2.0 / LGPL: Italian inflected forms
#   kaikki.org Wiktextract, Italian verbs (CC BY-SA): used only for essere/avere selection
set -euo pipefail
mkdir -p external/morph-it external/kaikki-ita-verbs
curl -sSL -o external/morph-it/morph-it.tgz \
  "https://docs.sslmit.unibo.it/lib/exe/fetch.php?media=resources:morph-it.tgz"
curl -sSL -o external/kaikki-ita-verbs/kaikki.org-dictionary-Italian-by-pos-verb.jsonl \
  "https://kaikki.org/dictionary/Italian/pos-verb/kaikki.org-dictionary-Italian-by-pos-verb.jsonl"
date -u +"fetched %Y-%m-%dT%H:%M:%SZ" > external/morph-it/FETCHED
cp external/morph-it/FETCHED external/kaikki-ita-verbs/FETCHED
shasum -a 256 external/morph-it/morph-it.tgz \
  external/kaikki-ita-verbs/kaikki.org-dictionary-Italian-by-pos-verb.jsonl
