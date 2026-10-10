#!/usr/bin/env bash
# Clone the external sources at the exact revisions used, into external/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
fetch() {  # name url revision
  if [ ! -d "external/$1/.git" ]; then git clone -q "$2" "external/$1"; fi
  git -C "external/$1" fetch -q origin && git -C "external/$1" checkout -q "$3"
  echo "$1 $(git -C "external/$1" rev-parse HEAD)"
}
fetch ALmorphinfl            https://github.com/smuradoglu/ALmorphinfl.git          3caf0d059846569ed0aff4b833492c104786c8e1
fetch JudiLing.jl            https://github.com/quantling/JudiLing.jl.git          ca77304cc6bf79ca38cb4b674797a0f09ebcd44d
fetch languages-of-the-world https://github.com/jnehring/languages-of-the-world.git d319631e916ae7f363d2e1c78f9dc9f3d61d3e51
# Only used to verify MGN provenance for the pilot languages (large):
fetch unimorph-ita           https://github.com/unimorph/ita                        fa2cc6ce173643e748bd8f4162709365bdc98805
fetch unimorph-fin           https://github.com/unimorph/fin                        fe0a2707244ed2ce2fe5d92a4a57c92271b32e1f
# Grambank inflection-extent outcome (typology stage; docs/TYPOLOGY.md). Shallow clones
# at the release tags; the stage checks HEAD against the pinned commits below.
fetch_tag() {  # name url tag commit
  if [ ! -d "external/$1/.git" ]; then git clone -q --depth 1 --branch "$3" "$2" "external/$1"; fi
  test "$(git -C "external/$1" rev-parse HEAD)" = "$4" || { echo "$1: HEAD is not $4 (tag $3)" >&2; exit 1; }
  echo "$1 $3 $(git -C "external/$1" rev-parse HEAD)"
}
fetch_tag grambank        https://github.com/grambank/grambank.git        v1.0.3 7ae000cf740f93cdb3e4ec67010668d6795337a9
fetch_tag glottolog-cldf  https://github.com/glottolog/glottolog-cldf.git  v5.3   072ca0d0410039fb8b779be8fc165bac575d2cda
