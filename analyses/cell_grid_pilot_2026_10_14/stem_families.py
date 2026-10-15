"""Rough size of the 'same stem, different prefix' problem (owner note, 2026-10-14).

Two lemmas join one family if one equals a known prefix + the other
(e.g. pisati / napisati / prepisati). Prefixes are closed per-language lists; the
remaining stem must have at least 4 characters. Union-find over the whole lemma list.
Suffixal partners (kupiti / kupovati) and secondary imperfectives of a prefixed verb
(zapisivati) are NOT joined by this rule, so the families are a lower bound.

Run from the repository root:
    .venv/bin/python analyses/cell_grid_pilot_2026_10_14/stem_families.py
"""
import csv
from collections import Counter
from pathlib import Path

PREFIXES = {
    "hbs": "na za po pre pri pro raz ras iz is s sa u uz us do od o ob nad pod iza pred "
           "prije pre nadi obe ode pode ra",
    "ita": "ri re s dis de in im ra ar ac ad con com sopra sotto contro tra stra pre",
}
FILES = {"hbs": "mgn_data/data/hbs-v.csv", "ita": "mgn_data/data/ita-v.csv"}
OUT = Path(__file__).with_name("stem_families_summary.csv")


def families(lemmas, prefixes):
    parent = {l: l for l in lemmas}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pref = sorted(set(prefixes.split()), key=len, reverse=True)
    for l in lemmas:
        for p in pref:
            base = l[len(p):]
            if l.startswith(p) and len(base) >= 4 and base in parent:
                parent[find(l)] = find(base)
    return Counter(find(l) for l in lemmas)


rows = []
for lang, path in FILES.items():
    lemmas = [r[0] for r in list(csv.reader(open(path)))[1:]]
    fam = families(lemmas, PREFIXES[lang])
    sizes = Counter(fam.values())
    in_multi = sum(s for s in fam.values() if s > 1)
    top = ", ".join(f"{k}:{v}" for k, v in fam.most_common(5))
    rows.append({"lang": lang, "lemmas": len(lemmas), "families": len(fam),
                 "lemmas_in_multi_member_families": in_multi,
                 "share_in_multi": round(in_multi / len(lemmas), 3),
                 "largest_families": top})
    print(rows[-1])

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
