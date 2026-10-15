"""Shared tooling for the cell-grid pilot (Phase 2).

Subcommands (run from the repository root with .venv/bin/python):

  prepare                 build derived inputs that need conversion (Italian: Morph-it! long
                          table and the essere/avere class lists from Wiktionary)
  inventory <iso>         write inputs/<iso>_inventory.csv (every source cell, coverage,
                          multiword share, examples) and inputs/<iso>_candidates.csv
                          (ranked candidate source cells for each active grid cell)
  check <iso>             validate mappings/<iso>.yaml, apply it to the data, write the
                          filled grid to outputs/cell_grid_pilot/<iso>_grid.csv and the
                          evidence report to mappings/<iso>_check.md. Exit code 1 on errors.

The tool never decides a mapping. It only lists candidates and tests a mapping against
the data: syncretism classes against form identity, auxiliary stripping against a small
token set, coverage, and the schema. No ancestry data is read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = HERE / "inputs"
MAPPINGS = HERE / "mappings"
OUT = ROOT / "outputs" / "cell_grid_pilot"
DERIVED = OUT / "inputs"

LANGS = {
    "ita": {"name": "Italian", "glottocode": "ital1282", "loader": "long",
            "path": DERIVED / "ita_morphit_long.csv",
            "resource": "morph-it 0.48 (Zanchetta & Baroni 2005), VER tags, clitic-suffixed forms dropped",
            "citation": "NFIN", "aux_order": "before"},
    "fin": {"name": "Finnish", "glottocode": "finn1318", "loader": "mgn_wide",
            "path": ROOT / "mgn_data/data/fin-v.csv", "resource": "mgn_data:fin-v", "citation": "NFIN"},
    "eng": {"name": "English", "glottocode": "stan1293", "loader": "mgn_long",
            "path": ROOT / "mgn_data/data-custom/english-v.csv",
            "resource": "mgn_custom:english-v (CELEX, phonemic, space-separated segments)",
            "citation": "inf"},
    "hye": {"name": "Eastern Armenian", "glottocode": "nucl1235", "loader": "mgn_wide",
            "path": ROOT / "mgn_data/data/hye-v.csv", "resource": "mgn_data:hye-v",
            "citation": "LEMMA", "aux_order": "after"},
    "hbs": {"name": "Serbo-Croatian", "glottocode": "sout1528", "loader": "mgn_wide",
            "path": ROOT / "mgn_data/data/hbs-v.csv", "resource": "mgn_data:hbs-v",
            "filters": ["drop_ekavian_doublets"], "citation": "LEMMA", "aux_order": "after"},
}

# feature dimensions used only to rank candidate source cells
DIMS = {
    "person": {"1", "2", "3", "4"}, "number": {"SG", "PL", "DU"},
    "tense": {"PRS", "PST", "FUT"}, "mood": {"IND", "IMP", "SBJV", "COND", "POT"},
    "aspect": {"PFV", "IPFV", "PRF", "PROG", "HAB"}, "polarity": {"POS", "NEG"},
    "voice": {"ACT", "PASS"}, "gender": {"MASC", "FEM", "NEUT"},
    "nonfinite": {"NFIN", "V.PTCP", "V.CVB", "V.CONV", "V.MSDR", "LEMMA"},
}
ENG_FEATS = {  # MGN English custom labels -> features (person/number left open where shared)
    "inf": "NFIN", "pres1s": "IND;PRS;1;SG", "pres3s": "IND;PRS;3;SG", "presothers": "IND;PRS",
    "past13": "IND;PST", "pastnot13": "IND;PST", "ppart": "V.PTCP;PST", "prespart": "V.PTCP;PRS",
}
MORPHIT = {"ind": "IND", "sub": "SBJV", "cond": "COND", "impr": "IMP", "inf": "NFIN",
           "part": "V.PTCP", "ger": "V.CVB", "pres": "PRS", "past": "PST;PFV", "impf": "PST;IPFV",
           "fut": "FUT", "s": "SG", "p": "PL", "m": "MASC", "f": "FEM",
           "1": "1", "2": "2", "3": "3"}
RULES = {"source", "syncretism", "main_verb", "restored"}
# LANGS[iso]["citation"]: source cell of the citation form (the infinitive; LEMMA = the
# lemma label where the data have no infinitive cell). It is written as an extra grid row
# NFIN for the LDL run (D36). LANGS[iso]["aux_order"]: where the auxiliary stands in the
# option-B form (D35), as in the context sentence and the data's own multiword forms.
CITATION_CELL = "NFIN"


# ----------------------------------------------------------------------------- loading
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def prepare() -> None:
    """Italian: Morph-it! -> long CSV with UniMorph-style labels; Wiktionary aux classes."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    src = ROOT / "external/morph-it/current_version/morph-it_048.txt"
    rows = []
    for line in open(src, encoding="latin-1"):
        p = line.rstrip("\n").split("\t")
        if len(p) != 3 or not p[2].startswith("VER:"):
            continue
        parts = p[2][4:].split("+")
        # keep plain inflected forms only: a trailing clitic (+si, +lo, ...) is not a cell
        if any(x not in MORPHIT for x in parts):
            continue
        if parts[0] == "part" and parts[1] == "past":
            label = "V.PTCP;PST;" + ";".join(MORPHIT[x] for x in parts[2:])
        elif parts[0] in ("inf", "ger"):
            label = MORPHIT[parts[0]]
        else:
            label = ";".join(MORPHIT[x] for x in parts)
        rows.append((p[1], label, p[0]))
    with open(DERIVED / "ita_morphit_long.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lemma", "cell", "form"])
        w.writerows(rows)  # file order kept: variant order = Morph-it order
    # essere/avere selection from Wiktionary (kaikki.org extract), union over entries
    aux = defaultdict(set)
    kk = ROOT / "external/kaikki-ita-verbs/kaikki.org-dictionary-Italian-by-pos-verb.jsonl"
    for line in open(kk, encoding="utf-8"):
        d = json.loads(line)
        if d.get("lang_code") != "it":
            continue
        for f in d.get("forms", []):
            if "auxiliary" in f.get("tags", []):
                a = strip_accents(f.get("form", "")).lower()
                if a in ("avere", "essere"):
                    aux[d["word"]].add(a)
    morphit_lemmas = {r[0] for r in rows}
    for cls, test in (("aux_avere", lambda s: s == {"avere"}),
                      ("aux_essere", lambda s: s == {"essere"}),
                      ("aux_both", lambda s: s == {"avere", "essere"})):
        with open(DERIVED / f"ita_{cls}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["lemma"])
            for lem in sorted(l for l, s in aux.items() if test(s) and l in morphit_lemmas):
                w.writerow([lem])
    n_cov = sum(1 for l in morphit_lemmas if l in aux)
    print(f"Morph-it VER rows kept: {len(rows)}; lemmas: {len(morphit_lemmas)}; "
          f"with Wiktionary auxiliary: {n_cov}")


def _read_mgn_wide(path):
    rows = list(csv.reader(open(path)))
    head = rows[0]
    data = defaultdict(lambda: defaultdict(list))
    for r in rows[1:]:
        lem = r[0]
        data[lem]["LEMMA"] = [lem]
        for c, x in zip(head[1:], r[1:]):
            if x not in ("NA", "") and x not in data[lem][c]:
                data[lem][c].append(x)  # repeated header columns are merged
    return data


def _read_long(path, lemma="lemma", cell="cell", form="form"):
    data = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        data[r[lemma]]["LEMMA"] = [r[lemma]]
        if r[form] and r[form] not in data[r[lemma]][r[cell]]:
            data[r[lemma]][r[cell]].append(r[form])
    return data


def drop_ekavian_doublets(data):
    """hbs: drop an ekavian lemma when its ijekavian doublet is also present."""
    lem = set(data)
    drop = set()
    for l in lem:
        for a, b in (("ije", "e"), ("je", "e")):
            if a in l and l.replace(a, b, 1) in lem:
                drop.add(l.replace(a, b, 1))
    return {k: v for k, v in data.items() if k not in drop}, len(drop)


def load(iso):
    cfg = LANGS[iso]
    if cfg["loader"] == "mgn_wide":
        data = _read_mgn_wide(cfg["path"])
    elif cfg["loader"] == "mgn_long":
        data = _read_long(cfg["path"], lemma="lexeme")
    else:
        data = _read_long(cfg["path"])
    notes = []
    for f in cfg.get("filters", []):
        data, n = globals()[f](data)
        notes.append(f"{f}: {n} lemmas dropped")
    return data, notes


def feats(iso, label):
    if iso == "eng":
        label = ENG_FEATS.get(label, label)
    return set(label.split(";"))


# ----------------------------------------------------------------------------- inventory
def grid_cells(active_only=True):
    rows = list(csv.DictReader(open(HERE / "grid.csv")))
    return [r for r in rows if not active_only or r["status"] == "active"]


def inventory(iso):
    data, notes = load(iso)
    INPUTS.mkdir(exist_ok=True)
    cnt, mw, ex = Counter(), Counter(), defaultdict(list)
    for lem, cells in data.items():
        for c, forms in cells.items():
            cnt[c] += 1
            mw[c] += " " in forms[0] and iso != "eng"
            if len(ex[c]) < 3:
                ex[c].append(f"{lem}: {' | '.join(forms)}")
    with open(INPUTS / f"{iso}_inventory.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_cell", "n_lemmas", "share_multiword", "examples"])
        for c in sorted(cnt, key=lambda c: (-cnt[c], c)):
            w.writerow([c, cnt[c], round(mw[c] / cnt[c], 3), " || ".join(ex[c])])
    with open(INPUTS / f"{iso}_candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grid_cell", "rank", "source_cell", "n_lemmas", "missing", "extra", "conflict"])
        for g in grid_cells():
            gf = set(g["unimorph_features"].split(";"))
            scored = []
            for c in cnt:
                sf = feats(iso, c)
                conflict = sorted(x for d in DIMS.values() for x in sf & d
                                  if (gf & d) and x not in gf)
                missing = sorted(gf - sf)
                extra = sorted(sf - gf)
                scored.append((len(conflict), len(missing), len(extra), -cnt[c], c,
                               missing, extra, conflict))
            scored.sort()
            for i, s in enumerate(scored[:8], 1):
                w.writerow([g["cell_id"], i, s[4], -s[3], ";".join(s[5]), ";".join(s[6]),
                            ";".join(s[7])])
    print(f"{iso}: {len(data)} lemmas, {len(cnt)} source cells. " + "; ".join(notes))


# ----------------------------------------------------------------------------- check/apply
def _apply_transform(form, t):
    if t is None:
        return form, None
    kind = t["type"]
    toks = form.split(" ")
    if kind == "drop_tokens":
        drop = set(t["tokens"])
        kept = [x for x in toks if x not in drop]
        dropped = [x for x in toks if x in drop]
        return " ".join(kept), " ".join(dropped) or None
    if kind == "keep_token":
        i = t["index"]
        return toks[i], " ".join(x for j, x in enumerate(toks) if j != i % len(toks)) or None
    if kind == "regex":
        new = re.sub(t["pattern"], t["repl"], form)
        return new, (form if new != form else None)
    if kind == "strip_marks":  # P6: remove combining marks, keeping the protected letters
        marks = {chr(int(cp, 16)) for cp in t["codepoints"]}
        protect = {unicodedata.normalize("NFC", p): chr(0xE000 + i)
                   for i, p in enumerate(t.get("protect") or [])}
        new = unicodedata.normalize("NFC", form)
        for p, pua in protect.items():
            new = new.replace(p, pua)
        new = "".join(ch for ch in unicodedata.normalize("NFD", new) if ch not in marks)
        new = unicodedata.normalize("NFC", new)
        for p, pua in protect.items():
            new = new.replace(pua, p)
        return new, (form if new != form else None)
    raise ValueError(f"unknown transform type {kind}")


def _tids(x):
    """a cell's transform: one id, a list of ids (applied in order, P6) or nothing"""
    return [] if not x else [x] if isinstance(x, str) else list(x)


def check(iso, mpath=None):
    errors, warns = [], []
    mpath = Path(mpath) if mpath else MAPPINGS / f"{iso}.yaml"
    m = yaml.safe_load(open(mpath))
    data, notes = load(iso)
    active = [g["cell_id"] for g in grid_cells()]
    inv = {c for cells in data.values() for c in cells}

    # ---- schema
    if m.get("grid_version") != "G14_v1":
        errors.append("grid_version must be G14_v1")
    cells = m.get("cells") or {}
    for c in active:
        if c not in cells or not cells[c]:
            errors.append(f"{c}: missing")
    for c in cells:
        if c not in active and cells[c]:
            errors.append(f"{c}: not an active grid cell")
    transforms = {t["id"]: t for t in (m.get("transforms") or [])}
    classes = {}
    for k in m.get("lexical_classes") or []:
        f = ROOT / k["membership_file"]
        if not f.exists():
            errors.append(f"class {k['id']}: membership file {f} not found")
            continue
        classes[k["id"]] = {r["lemma"] for r in csv.DictReader(open(f))}
    for c in active:
        e = cells.get(c) or {}
        r = e.get("rule")
        if r not in RULES:
            errors.append(f"{c}: rule {r!r} not in {sorted(RULES)}")
        if r in ("source", "main_verb"):
            if e.get("source_cell") not in inv:
                errors.append(f"{c}: source_cell {e.get('source_cell')!r} not in the data")
        if r in ("syncretism", "restored"):
            if e.get("same_as") not in active or e.get("same_as") == c:
                errors.append(f"{c}: same_as {e.get('same_as')!r} is not another active cell")
        for tid in _tids(e.get("transform")) + [t for ov in (e.get("by_class") or {}).values()
                                                 for t in _tids(ov.get("transform"))]:
            if tid not in transforms:
                errors.append(f"{c}: transform {tid!r} not declared")
        for cls, ov in (e.get("by_class") or {}).items():
            if cls not in classes:
                errors.append(f"{c}: by_class {cls!r} not a declared lexical class")
            if ov.get("source_cell") and ov["source_cell"] not in inv:
                errors.append(f"{c}/{cls}: source_cell {ov['source_cell']!r} not in the data")
        ref = e.get("reference") or {}
        refs = ref if isinstance(ref, list) else [ref]
        for rf in refs:
            if not all(rf.get(k) for k in ("author", "year", "locus")) or \
                    not isinstance(rf.get("locus_verified"), bool):
                errors.append(f"{c}: reference needs author, year, locus and locus_verified (bool)")
                break
        if e.get("confidence") not in ("high", "medium", "low"):
            errors.append(f"{c}: confidence must be high/medium/low")
        if not e.get("distinct_class"):
            errors.append(f"{c}: distinct_class missing")
        if not isinstance(e.get("aux_carried"), bool):
            errors.append(f"{c}: aux_carried must be true/false")
    exceptions = set()  # P2: a closed list of irregular lemmas, ignored by the identity checks
    for x in m.get("class_exceptions") or []:
        exceptions |= set(x.get("lemmas") or [])
    if len(exceptions) > 10:
        errors.append(f"class_exceptions: {len(exceptions)} lemmas (at most 10, rule P2)")
    for d in m.get("defective") or []:  # P5: defective only where a cited grammar says so
        rf = d.get("reference")
        if not rf or not all((r or {}).get(k) for r in (rf if isinstance(rf, list) else [rf])
                             for k in ("author", "year", "locus")):
            errors.append(f"defective {d.get('cells')}: needs a reference (rule P5); "
                          "use data_gaps otherwise")
    if errors:
        _report(iso, m, notes, errors, warns, None)
        return 1

    # ---- apply
    scope = m.get("dataset", {}).get("lemma_scope_rules", {}) or {}
    lemmas = sorted(data)
    out_scope = Counter()
    if scope.get("exclude_multiword_lemmas", True):
        out_scope["multiword_lemma"] = sum(" " in l for l in lemmas)
        lemmas = [l for l in lemmas if " " not in l]
    excl = set()
    for f in scope.get("exclude_lemma_files", []) or []:
        excl |= {r["lemma"] for r in csv.DictReader(open(ROOT / f))}
    out_scope["excluded_by_list"] = sum(l in excl for l in lemmas)
    lemmas = [l for l in lemmas if l not in excl]
    def lemma_lists(key):
        """defective / data_gaps entries -> {lemma: {cell: note}}"""
        out = defaultdict(dict)
        for d in m.get(key) or []:
            ls = set(d.get("lemmas") or [])
            if d.get("lemmas_file"):
                ls |= {r["lemma"] for r in csv.DictReader(open(ROOT / d["lemmas_file"]))}
            for l in ls:
                for c in d["cells"]:
                    out[l][c] = d.get("reason", "")
        return out

    defect = lemma_lists("defective")
    gaps = lemma_lists("data_gaps")  # forms present in the data but known to be wrong
    overrides = {}
    if m.get("variant_overrides"):
        for r in csv.DictReader(open(ROOT / m["variant_overrides"])):
            overrides[(r["lemma"], r["grid_cell"])] = r["form"]

    def order(outs, rule):
        if not rule:
            return outs
        pre, avo = rule.get("prefer"), rule.get("avoid")
        key = lambda f: (0 if pre and re.search(pre, f) else 1,
                         1 if avo and re.search(avo, f) else 0)
        return sorted(outs, key=key)  # stable: data order within ties

    def resolve(c, seen=()):
        e = cells[c]
        if e["rule"] in ("syncretism", "restored"):
            if e["same_as"] in seen:
                raise ValueError(f"syncretism cycle at {c}")
            return resolve(e["same_as"], seen + (c,))
        return c

    for c in active:
        try:
            resolve(c)
        except ValueError as ex:
            errors.append(str(ex))
    if errors:
        _report(iso, m, notes, errors, warns, None)
        return 1

    dropped_tokens = defaultdict(Counter)
    grid = []
    for lem in lemmas:
        cls = next((k for k in classes if lem in classes[k]), None)
        for c in active:
            base = resolve(c)
            e = cells[base]
            ov = (e.get("by_class") or {}).get(cls, {}) if cls else {}
            src = ov.get("source_cell", e["source_cell"])
            tids = _tids(ov.get("transform", e.get("transform")))
            tid = "+".join(tids)
            # P1: the auxiliary of this cell (D1). Without an `aux` field, an aux-carried
            # cell counts as having its own auxiliary
            own = (cells[c].get("by_class") or {}).get(cls, {}) if cls else {}
            aux = own.get("aux", cells[c].get("aux"))
            aux_b = aux or ""            # the real auxiliary string, for the option-B form
            aux = aux if aux is not None else (c if cells[c]["aux_carried"] else "")
            forms = data[lem].get(src, [])
            row = {"lemma": lem, "grid_cell": c, "rule": cells[c]["rule"],
                   "source_cell": src, "transform": tid, "lexical_class": cls or "", "aux": aux,
                   "aux_b": aux_b,
                   "distinct_class": cells[c]["distinct_class"],
                   "aux_carried": cells[c]["aux_carried"], "form": "", "variants": "",
                   "form_b": "", "variants_b": "",
                   "na_reason": "", "na_note": ""}
            if c in defect.get(lem, {}):
                row["na_reason"], row["na_note"] = "defective", defect[lem][c]
            elif c in gaps.get(lem, {}):
                row["na_reason"], row["na_note"] = "data_gap", gaps[lem][c]
            elif not forms:
                row["na_reason"] = "data_gap"
            else:
                outs = []
                for f in forms:
                    g = f
                    for t in tids:
                        g, d = _apply_transform(g, transforms[t])
                        if d:
                            dropped_tokens[t][d] += 1
                    outs.append(g)
                outs = order(outs, cells[c].get("variant_rule") or e.get("variant_rule"))
                ov_form = overrides.get((lem, c))
                if ov_form is not None:
                    if ov_form in outs:
                        outs = [ov_form] + [x for x in outs if x != ov_form]
                    else:
                        warns.append(f"variant override not among the forms: {lem} {c} {ov_form!r}")
                row["form"] = outs[0]
                row["variants"] = "|".join(outs[1:])
                if aux_b:  # option B (D35): the auxiliary joined to the lexical verb
                    if not LANGS[iso].get("aux_order"):
                        raise ValueError(f"{iso}: aux given but no aux_order in LANGS")
                    j = (lambda f: f"{aux_b} {f}") if LANGS[iso]["aux_order"] == "before" \
                        else (lambda f: f"{f} {aux_b}")
                    row["form_b"], row["variants_b"] = j(outs[0]), "|".join(j(x) for x in outs[1:])
                else:
                    row["form_b"], row["variants_b"] = row["form"], row["variants"]
                if " " in outs[0] and iso != "eng":
                    warns.append(f"multiword form after transform: {lem} {c} {outs[0]!r}")
            grid.append(row)

    # ---- class structure and evidence
    cls_of = {c: cells[c]["distinct_class"] for c in active}
    members = defaultdict(list)
    for c, k in cls_of.items():
        members[k].append(c)
    for row in grid:
        row["is_formally_distinct"] = len(members[row["distinct_class"]]) == 1
    by = defaultdict(dict)
    for row in grid:
        if row["form"]:
            by[row["lemma"]][row["grid_cell"]] = row["form"]
    # P1: per-lemma empirical class = same grid form and same auxiliary; labelled by the
    # first grid cell of the group
    key_of = defaultdict(dict)
    for row in grid:
        if row["form"]:
            key_of[row["lemma"]].setdefault((row["form"], row["aux"]), row["grid_cell"])
    for row in grid:
        row["emp_class"] = key_of[row["lemma"]][(row["form"], row["aux"])] if row["form"] else ""
    pair = {}
    for a, b in combinations(active, 2):
        both = [v for l, v in by.items() if a in v and b in v and l not in exceptions]
        if both:
            pair[(a, b)] = (sum(v[a] == v[b] for v in both) / len(both), len(both))
    flags = []
    for (a, b), (s, n) in pair.items():
        same = cls_of[a] == cls_of[b]
        aux_ok = cells[a]["aux_carried"] and cells[b]["aux_carried"]
        if same and s < 0.95:
            flags.append(f"same class, forms identical in only {s:.3f} of {n} lemmas: {a} ~ {b}")
        if not same and s >= 0.95 and not aux_ok:
            flags.append(f"different classes but forms identical in {s:.3f} of {n} lemmas "
                         f"(neither explained by aux_carried): {a} ~ {b}")
    warns.extend(flags)
    for tid, cnt in dropped_tokens.items():  # only token-dropping transforms remove auxiliaries
        if transforms[tid]["type"] in ("drop_tokens", "keep_token") and len(cnt) > 30:
            warns.append(f"transform {tid}: {len(cnt)} distinct dropped strings "
                         f"(an auxiliary set should be small)")

    OUT.mkdir(parents=True, exist_ok=True)
    version = hashlib.sha256(mpath.read_bytes()).hexdigest()[:12]
    fields = ["lemma", "grid_cell", "form", "variants", "form_b", "variants_b", "rule", "source_cell",
              "transform", "lexical_class", "distinct_class", "is_formally_distinct", "aux_carried", "aux",
              "emp_class", "na_reason", "na_note", "mapping_version"]
    # the citation form (D36): an extra row per lemma, not a grid cell (not in the stats);
    # the mapping's strip_marks transforms apply to it as to every cell
    cit_src = LANGS[iso].get("citation")
    marks = [t for t in transforms.values() if t["type"] == "strip_marks"]
    cit_rows = []
    for lem in lemmas:
        fs = [lem] if cit_src == "LEMMA" else data[lem].get(cit_src, [])
        outs = []
        for f in fs:
            for t in marks:
                f = _apply_transform(f, t)[0]
            outs.append(f)
        if lem in outs:  # the lemma label itself first (Morph-it lists apocopated parlar first)
            outs = [lem] + [x for x in outs if x != lem]
        cit_rows.append({"lemma": lem, "grid_cell": CITATION_CELL, "form": outs[0] if outs else "",
                         "variants": "|".join(outs[1:]), "form_b": outs[0] if outs else "",
                         "variants_b": "|".join(outs[1:]), "rule": "citation", "source_cell": cit_src,
                         "transform": "+".join(t["id"] for t in marks), "lexical_class": "",
                         "distinct_class": "", "is_formally_distinct": "", "aux_carried": False, "aux": "",
                         "emp_class": "", "na_reason": "" if outs else "data_gap", "na_note": ""})
    with open(OUT / f"{iso}_grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in grid + cit_rows:
            row["mapping_version"] = version
            w.writerow(row)
    # empirical distinctness, verb by verb (complete grids only): shows syncretism that
    # holds for some lexical classes only, which distinct_class (language level) cannot
    complete = [v for v in by.values() if len(v) == len(active)]
    n_forms = Counter(len(set(v.values())) for v in complete)
    complete_l = {l for l, v in by.items() if len(v) == len(active)}
    n_emp = Counter(len(key_of[l]) for l in complete_l)
    uniq = {c: round(sum(list(v.values()).count(v[c]) == 1 for v in complete) / len(complete), 3)
            for c in active} if complete else {}
    stats = {
        "lemmas_in_data": len(data), "out_of_scope": dict(out_scope), "lemmas_in_scope": len(lemmas),
        "complete_grids": len(lemmas) - len({r["lemma"] for r in grid if not r["form"]}),
        "empty_cells": dict(Counter(r["na_reason"] for r in grid if r["na_reason"])),
        "n_classes": len(members), "n_formally_distinct_cells": sum(len(v) == 1 for v in members.values()),
        "rule_counts": dict(Counter(cells[c]["rule"] for c in active)),
        "aux_carried_cells": sum(cells[c]["aux_carried"] for c in active),
        "classes": {k: v for k, v in members.items()},
        "dropped_tokens": {t: c.most_common(15) for t, c in dropped_tokens.items()},
        "empirical_distinct_forms_per_complete_grid": dict(sorted(n_forms.items())),
        "empirical_mean_distinct_forms": round(sum(k * n for k, n in n_forms.items()) /
                                               max(1, len(complete)), 2),
        "empirical_cell_unique_share": uniq,
        "emp_classes_per_complete_grid": dict(sorted(n_emp.items())),
        "emp_mean_classes": round(sum(k * n for k, n in n_emp.items()) / max(1, len(complete_l)), 2),
        "class_exceptions": sorted(exceptions),
        "pair_identity": {f"{a}~{b}": round(s, 3) for (a, b), (s, n) in pair.items() if s > 0.05},
    }
    _report(iso, m, notes, errors, warns, stats)
    return 0


def _report(iso, m, notes, errors, warns, stats):
    MAPPINGS.mkdir(exist_ok=True)
    lines = [f"# Check report: {iso}", "", f"Mapping: `mappings/{iso}.yaml`. "
             f"Data: {LANGS[iso]['resource']}. Loader notes: {'; '.join(notes) or 'none'}.", ""]
    lines += ["## Errors", ""] + ([f"* {e}" for e in errors] or ["none"]) + [""]
    shown = warns[:60]
    lines += ["## Warnings", ""] + ([f"* {w}" for w in shown] or ["none"])
    if len(warns) > len(shown):
        lines.append(f"* … {len(warns) - len(shown)} more")
    if stats:
        lines += ["", "## Summary", "", "```json", json.dumps(stats, ensure_ascii=False, indent=1),
                  "```"]
    (MAPPINGS / f"{iso}_check.md").write_text("\n".join(lines) + "\n")
    print(f"{iso}: {len(errors)} errors, {len(warns)} warnings -> mappings/{iso}_check.md")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "inventory":
        for iso in sys.argv[2:] or LANGS:
            inventory(iso)
    elif cmd == "check":
        sys.exit(check(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
    else:
        sys.exit(__doc__)
