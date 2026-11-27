"""Unicode multi-script + Sukhotin vowel-tier scansion and optional phonemic feature inventory.

CRITICAL INVARIANT (Zero Target Leakage):
- At test/prediction time on `(w_s -> w_t)`, `VowelClassifier` and `project_vowel_tier`
  operate strictly on the source word `w_s` (and parameters fitted on training forms).
  Target forms `w_t` are NEVER read during vowel-tier scansion.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Explicit Multi-Script Unicode Vowel Character Tables
# ---------------------------------------------------------------------------

# Unambiguous Latin & IPA single-codepoint vowels (excluding ambiguous glides y/w,
# which are resolved via Sukhotin co-occurrence context when corpus data are available,
# with y defaulting to V in consonant-adjacent environments).
_LATIN_IPA_VOWELS: frozenset[str] = frozenset(
    "aeiou"
    "àáâãäåāăąǎȁȃạảấầẩẫậắằẳẵặ"
    "èéêëēĕėęěȅȇẹẻẽếềểễệ"
    "ìíîïĩīĭįıǐȉȋịỉ"
    "òóôõöøōŏőǒȍȏơọỏốồổỗộớờởỡợ"
    "ùúûüũūŭůűųǔȕȗưụủứừửữự"
    "ýÿŷȳỳỵỷỹ"
    "æǣœɶɐɑɒɔɘəɚɛɜɝɞɨɪɯɵɶʉʊʌɤ"
)

# Bare ASCII/Latin glides whose syllabicity varies cross-linguistically (e.g., Finnish 'y' = [y] V
# vs. Turkish 'y' = [j] C; Welsh 'w' = [u] V vs. English/Polish 'w' = C).
_AMBIGUOUS_GLIDES: frozenset[str] = frozenset({"y", "w"})

# Cyrillic vowels including Slavic + Turkic/Uralic/Mongolic/Caucasian letters:
# ә ө ү і ӱ ӧ ы э я ю е ё а и ї о у ӗ ӑ ӳ ӯ ӣ ѣ ѫ ѧ ѩ ѭ
_CYRILLIC_VOWELS: frozenset[str] = frozenset(
    "аеёиієїоуыэюя"
    "әөүӱӧӗӑӳӯӣѣѫѧѩѭ"
    "ӒӓӚӛӖӗӢӣӦӧӮӯӰӱӲӳӴӵӸӹ"
)
_CYRILLIC_NON_VOWELS: frozenset[str] = frozenset("йьъјўҍ")

# Greek base vowels (monotonic + polytonic via NFD base stripping)
_GREEK_VOWELS: frozenset[str] = frozenset("αεηιουωἀἁἂἃἄἅἆἇἐἑἒἓἔἕἠἡἢἣἤἥἦἧἰἱἲἳἴἵἶἷὀὁὂὃὄὅὐὑὒὓὔὕὖὗὠὡὢὣὤὥὦὧ")

# Armenian vowels (ա ե է ը ի ո օ)
_ARMENIAN_VOWELS: frozenset[str] = frozenset("աեէըիոօ")

# Georgian Mkhedruli + Mtavruli vowels (ა ე ი ო უ)
_GEORGIAN_VOWELS: frozenset[str] = frozenset("აეიოუᲐᲔᲘᲝᲳ")

# Hebrew & Arabic vocalic codepoints (niqqud, harakat, Kurdish/Uyghur/Persian vocalic letters)
_HEBREW_ARABIC_VOWELS: frozenset[str] = frozenset(
    # Hebrew niqqud vowels (sheva..qubbuts, hataf vowels, holam)
    "\u05b0\u05b1\u05b2\u05b3\u05b4\u05b5\u05b6\u05b7\u05b8\u05b9\u05ba\u05bb"
    # Arabic harakat + alef variants + Kurdish/Uyghur vowel letters (ە ۆ ێ ۇ ۈ ى ا آ أ إ ٱ)
    "\u064b\u064c\u064d\u064e\u064f\u0650\u0653\u0670"
    "اآأإٱىەۆێۇۈې"
)

# Devanagari independent vowels & matras (plus generic Unicode name check for all Indic scripts)
_DEVANAGARI_VOWELS: frozenset[str] = frozenset(
    "अआइईउऊऋॠऌॡऍऎएऐऑओऔ"
    "\u093a\u093b\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0945\u0946\u0947\u0948"
    "\u0949\u094a\u094b\u094c\u094e\u094f\u0955\u0956\u0957\u0962\u0963"
)

# Hangul Jungseong (conjoining medial vowels \u1161-\u1175 & compatibility jamo \u314f-\u3163)
_HANGUL_VOWELS: frozenset[str] = frozenset(
    [chr(cp) for cp in range(0x1161, 0x1176)] + [chr(cp) for cp in range(0x314F, 0x3164)]
)

_ALL_EXPLICIT_VOWEL_CHARS: frozenset[str] = (
    _LATIN_IPA_VOWELS
    | _CYRILLIC_VOWELS
    | _GREEK_VOWELS
    | _ARMENIAN_VOWELS
    | _GEORGIAN_VOWELS
    | _HEBREW_ARABIC_VOWELS
    | _DEVANAGARI_VOWELS
    | _HANGUL_VOWELS
)

# Suprasegmental / length / tone / tie-bar / diacritic characters ignored when finding base segments
_SUPRASEGMENTAL_CHARS: frozenset[str] = frozenset("ːˑˈˌ˥˦˧˨˩‿͜͡'")

# Built-in core IPA binary feature bundles used when `phon` mode is active or as a fallback
# for standard IPA symbols in `{unit_id}_sounds.csv`.
_BUILTIN_IPA_FEATURES: dict[str, frozenset[str]] = {
    # Vowels
    "i": frozenset({"+syllabic", "-consonantal", "+sonorant", "+high", "-low", "-back", "-round", "+voice"}),
    "y": frozenset({"+syllabic", "-consonantal", "+sonorant", "+high", "-low", "-back", "+round", "+voice"}),
    "ɨ": frozenset({"+syllabic", "-consonantal", "+sonorant", "+high", "-low", "+back", "-round", "+voice"}),
    "ʉ": frozenset({"+syllabic", "-consonantal", "+sonorant", "+high", "-low", "+back", "+round", "+voice"}),
    "ɯ": frozenset({"+syllabic", "-consonantal", "+sonorant", "+high", "-low", "+back", "-round", "+voice"}),
    "u": frozenset({"+syllabic", "-consonantal", "+sonorant", "+high", "-low", "+back", "+round", "+voice"}),
    "ɪ": frozenset({"+syllabic", "-consonantal", "+sonorant", "+high", "-low", "-back", "-round", "-tense", "+voice"}),
    "ʏ": frozenset({"+syllabic", "-consonantal", "+sonorant", "+high", "-low", "-back", "+round", "-tense", "+voice"}),
    "ʊ": frozenset({"+syllabic", "-consonantal", "+sonorant", "+high", "-low", "+back", "+round", "-tense", "+voice"}),
    "e": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "-low", "-back", "-round", "+voice"}),
    "ø": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "-low", "-back", "+round", "+voice"}),
    "ə": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "-low", "+back", "-round", "+voice"}),
    "o": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "-low", "+back", "+round", "+voice"}),
    "ɛ": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "-low", "-back", "-round", "-tense", "+voice"}),
    "œ": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "-low", "-back", "+round", "-tense", "+voice"}),
    "ʌ": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "-low", "+back", "-round", "-tense", "+voice"}),
    "ɔ": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "-low", "+back", "+round", "-tense", "+voice"}),
    "æ": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "+low", "-back", "-round", "+voice"}),
    "a": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "+low", "-back", "-round", "+voice"}),
    "ɑ": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "+low", "+back", "-round", "+voice"}),
    "ɒ": frozenset({"+syllabic", "-consonantal", "+sonorant", "-high", "+low", "+back", "+round", "+voice"}),
    # Stops
    "p": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "-nasal", "+labial", "-coronal", "-dorsal", "-voice"}),
    "b": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "-nasal", "+labial", "-coronal", "-dorsal", "+voice"}),
    "t": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "-nasal", "-labial", "+coronal", "+anterior", "-dorsal", "-voice"}),
    "d": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "-nasal", "-labial", "+coronal", "+anterior", "-dorsal", "+voice"}),
    "k": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "-nasal", "-labial", "-coronal", "+dorsal", "-voice"}),
    "g": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "-nasal", "-labial", "-coronal", "+dorsal", "+voice"}),
    "ɡ": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "-nasal", "-labial", "-coronal", "+dorsal", "+voice"}),
    "q": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "-nasal", "-labial", "-coronal", "+dorsal", "-voice"}),
    "ʔ": frozenset({"-syllabic", "-consonantal", "-sonorant", "-continuant", "-nasal", "-voice"}),
    # Affricates
    "ts": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "+strident", "+coronal", "+anterior", "-voice"}),
    "dz": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "+strident", "+coronal", "+anterior", "+voice"}),
    "tʃ": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "+strident", "+coronal", "-anterior", "-voice"}),
    "dʒ": frozenset({"-syllabic", "+consonantal", "-sonorant", "-continuant", "+strident", "+coronal", "-anterior", "+voice"}),
    # Fricatives
    "f": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "+labial", "-coronal", "-dorsal", "-voice"}),
    "v": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "+labial", "-coronal", "-dorsal", "+voice"}),
    "θ": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "+coronal", "+anterior", "-strident", "-voice"}),
    "ð": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "+coronal", "+anterior", "-strident", "+voice"}),
    "s": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "+coronal", "+anterior", "+strident", "-voice"}),
    "z": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "+coronal", "+anterior", "+strident", "+voice"}),
    "ʃ": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "+coronal", "-anterior", "+strident", "-voice"}),
    "ʒ": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "+coronal", "-anterior", "+strident", "+voice"}),
    "ç": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "-coronal", "+dorsal", "-voice"}),
    "x": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "-labial", "-coronal", "+dorsal", "-voice"}),
    "ɣ": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "-labial", "-coronal", "+dorsal", "+voice"}),
    "χ": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "-labial", "-coronal", "+dorsal", "-voice"}),
    "ʁ": frozenset({"-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal", "-labial", "-coronal", "+dorsal", "+voice"}),
    "h": frozenset({"-syllabic", "-consonantal", "-sonorant", "+continuant", "-nasal", "-voice"}),
    # Nasals
    "m": frozenset({"-syllabic", "+consonantal", "+sonorant", "-continuant", "+nasal", "+labial", "+voice"}),
    "n": frozenset({"-syllabic", "+consonantal", "+sonorant", "-continuant", "+nasal", "+coronal", "+anterior", "+voice"}),
    "ɲ": frozenset({"-syllabic", "+consonantal", "+sonorant", "-continuant", "+nasal", "+coronal", "-anterior", "+dorsal", "+voice"}),
    "ŋ": frozenset({"-syllabic", "+consonantal", "+sonorant", "-continuant", "+nasal", "+dorsal", "+voice"}),
    # Liquids & Glides
    "l": frozenset({"-syllabic", "+consonantal", "+sonorant", "+continuant", "-nasal", "+lateral", "+coronal", "+voice"}),
    "ʎ": frozenset({"-syllabic", "+consonantal", "+sonorant", "+continuant", "-nasal", "+lateral", "+dorsal", "+voice"}),
    "r": frozenset({"-syllabic", "+consonantal", "+sonorant", "+continuant", "-nasal", "-lateral", "+coronal", "+voice"}),
    "ɾ": frozenset({"-syllabic", "+consonantal", "+sonorant", "+continuant", "-nasal", "-lateral", "+coronal", "+voice"}),
    "ɹ": frozenset({"-syllabic", "+consonantal", "+sonorant", "+continuant", "-nasal", "-lateral", "+coronal", "+voice"}),
    "j": frozenset({"-syllabic", "-consonantal", "+sonorant", "+continuant", "-nasal", "+high", "-back", "+voice"}),
    "w": frozenset({"-syllabic", "-consonantal", "+sonorant", "+continuant", "-nasal", "+high", "+back", "+round", "+voice"}),
}


def parse_segments(segmented_form: str | None, raw_form: str | None = None) -> tuple[str, ...]:
    """Parse space-separated `segmented_form` (or NFC grapheme clusters of `raw_form`) into tokens."""
    if segmented_form is not None:
        norm = unicodedata.normalize("NFC", str(segmented_form)).strip()
        if norm:
            return tuple(tok for tok in norm.split() if tok)
    if raw_form is None:
        return ()
    norm_raw = unicodedata.normalize("NFC", str(raw_form)).strip()
    if not norm_raw:
        return ()
    # Group base characters with any trailing Unicode combining marks (Mn/Mc/Me) or length marks
    clusters: list[str] = []
    for ch in norm_raw:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if clusters and (cat.startswith("M") or ch in {"ː", "ˑ"}):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return tuple(clusters)


def _unicode_char_is_vowel(ch: str) -> bool | None:
    """Return True (vowel), False (consonant), or None (unknown/non-letter or ambiguous glide)."""
    low = unicodedata.normalize("NFC", ch).lower()
    if low in _AMBIGUOUS_GLIDES:
        return None
    if low in _CYRILLIC_NON_VOWELS:
        return False
    if low in _ALL_EXPLICIT_VOWEL_CHARS:
        return True

    # Check NFD base character (strips combining accents, tones, macrons, umlauts, breathings)
    nfd = unicodedata.normalize("NFD", low)
    base_chars = [c for c in nfd if not unicodedata.category(c).startswith("M")]
    if base_chars:
        b0 = base_chars[0]
        if b0 in _AMBIGUOUS_GLIDES:
            return None
        if b0 in _CYRILLIC_NON_VOWELS:
            return False
        if b0 in _ALL_EXPLICIT_VOWEL_CHARS:
            return True

    # Generic Unicode character name inspection for Indic matras, Ethiopic, Thaana, Khmer, Thai, etc.
    uname = unicodedata.name(ch, "")
    if uname:
        if "VOWEL" in uname and "SEMI-VOWEL" not in uname and "SEMIVOWEL" not in uname:
            return True
        if any(
            kw in uname
            for kw in ("CONSONANT", "LETTER", "SYLLABLE", "STOP", "CLICK", "FRICATIVE", "NASAL")
        ):
            #Wait: only mark known alphabetic scripts as False; unknown scripts without VOWEL/CONSONANT
            # distinction in Unicode names (e.g. cipher symbols, PUA, digits, syllabaries) return None!
            if any(
                scr in uname
                for scr in (
                    "LATIN",
                    "IPA",
                    "CYRILLIC",
                    "GREEK",
                    "ARMENIAN",
                    "GEORGIAN",
                    "HEBREW",
                    "ARABIC",
                    "DEVANAGARI",
                    "BENGALI",
                    "GURMUKHI",
                    "GUJARATI",
                    "ORIYA",
                    "TAMIL",
                    "TELUGU",
                    "KANNADA",
                    "MALAYALAM",
                    "SINHALA",
                    "THAI",
                    "LAO",
                    "TIBETAN",
                    "MYANMAR",
                    "KHMER",
                    "HANGUL",
                )
            ):
                return False
    return None


_MALOUF_STEM_MARKERS: frozenset[str] = frozenset({"HNEG", "HPOS", "HNEUT"})


def _classify_segment_unicode(seg: str) -> str | None:
    """Classify a space-separated token `seg` into `'V'`, `'C'`, or `None` (unknown/ambiguous)."""
    norm = unicodedata.normalize("NFC", seg).strip()
    if not norm:
        return None
    upper = norm.upper()
    # Malouf (2017) harmony pre-abstraction tokens (`ingestion/harmony_malouf.py`):
    # `VH1, VH2, ...` are abstract harmonizing vowel archiphonemes ('V');
    # `HNEG, HPOS, HNEUT` are non-syllabic stem-polarity prefix markers ('C').
    if upper in _MALOUF_STEM_MARKERS:
        return "C"
    if len(upper) >= 3 and upper.startswith("VH") and upper[2:].isdigit():
        return "V"
    if norm.lower() in _AMBIGUOUS_GLIDES:
        return None

    # Strip suprasegmental/length/stress marks and modifier letters (Lm like ʲ ʷ ʰ ˠ ˤ)
    # while keeping vocalic matras/points (Mn/Mc that are explicitly vowels, e.g. Indic/Hebrew/Arabic).
    explicit_vowel = False
    explicit_cons = False
    has_unknown = False

    for ch in norm:
        if ch in _SUPRASEGMENTAL_CHARS:
            continue
        cat = unicodedata.category(ch)
        if cat == "Lm":
            # Secondary articulation modifiers (ʲ, ʷ, ʰ, ˤ, ʼ) do not make a consonant into a vowel
            continue
        if cat.startswith("M"):
            if ch in _ALL_EXPLICIT_VOWEL_CHARS or "VOWEL" in unicodedata.name(ch, ""):
                explicit_vowel = True
            continue
        verdict = _unicode_char_is_vowel(ch)
        if verdict is True:
            explicit_vowel = True
        elif verdict is False:
            explicit_cons = True
        else:
            has_unknown = True

    if explicit_vowel:
        return "V"
    if explicit_cons and not has_unknown:
        return "C"
    return None


def sukhotin_discover_vowels(forms: Iterable[Sequence[str]]) -> set[str]:
    """Sukhotin's (1962) unsupervised co-occurrence algorithm for vowel discovery.

    Constructs the symmetric immediate-neighbor co-occurrence matrix `A[u, v]` (`u != v`)
    across segmented `forms`, initializes all symbols as consonants (`C`), and iteratively
    promotes the symbol with maximal positive excess of `C`-neighbors over `V`-neighbors
    to `V` until no remaining consonant has `S[u] > 0`.
    """
    adj: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    vocab: set[str] = set()
    for segs in forms:
        for idx, tok in enumerate(segs):
            if tok == "#":
                continue
            vocab.add(tok)
            if idx > 0:
                prev = segs[idx - 1]
                if prev != "#" and prev != tok:
                    adj[tok][prev] += 1
                    adj[prev][tok] += 1

    if not vocab:
        return set()

    # Row sums S[u] = sum_{v in C} A[u, v] - sum_{v in V} A[u, v]; initially V = empty.
    scores: dict[str, int] = {u: sum(adj[u].values()) for u in vocab}
    vowels: set[str] = set()
    remaining: set[str] = set(vocab)

    while remaining:
        # Deterministic tie-breaking: highest score, then lexicographically smallest symbol
        best_sym = min(remaining, key=lambda u: (-scores[u], u))
        if scores[best_sym] <= 0:
            break
        vowels.add(best_sym)
        remaining.remove(best_sym)
        for neighbor, count in adj[best_sym].items():
            if neighbor in remaining:
                scores[neighbor] -= 2 * count

    return vowels


# ---------------------------------------------------------------------------
# Paralex / CLDF `sounds.csv` Phonological Feature Inventory (`phon` mode)
# ---------------------------------------------------------------------------

_PROSE_TO_FEATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vowel", ("+syllabic", "-consonantal", "+sonorant", "+voice")),
    ("diphthong", ("+syllabic", "-consonantal", "+sonorant", "+voice")),
    ("syllabic", ("+syllabic",)),
    ("consonant", ("-syllabic", "+consonantal")),
    ("plosive", ("-syllabic", "+consonantal", "-sonorant", "-continuant", "-nasal")),
    ("stop", ("-syllabic", "+consonantal", "-sonorant", "-continuant")),
    ("affricate", ("-syllabic", "+consonantal", "-sonorant", "-continuant", "+strident")),
    ("fricative", ("-syllabic", "+consonantal", "-sonorant", "+continuant", "-nasal")),
    ("sibilant", ("+strident", "+coronal")),
    ("nasal", ("+nasal", "+sonorant")),
    ("approximant", ("+sonorant", "+continuant", "-nasal")),
    ("lateral", ("+lateral", "+sonorant", "+continuant")),
    ("trill", ("+sonorant", "+continuant", "-lateral", "+coronal")),
    ("tap", ("+sonorant", "+continuant", "-lateral", "+coronal")),
    ("flap", ("+sonorant", "+continuant", "-lateral", "+coronal")),
    ("voiced", ("+voice",)),
    ("voiceless", ("-voice",)),
    ("bilabial", ("+labial", "-coronal", "-dorsal")),
    ("labiodental", ("+labial", "-coronal", "-dorsal")),
    ("labial", ("+labial",)),
    ("dental", ("+coronal", "+anterior")),
    ("alveolar", ("+coronal", "+anterior")),
    ("postalveolar", ("+coronal", "-anterior")),
    ("retroflex", ("+coronal", "-anterior")),
    ("palatal", ("-labial", "+dorsal", "-back", "+high")),
    ("velar", ("-labial", "-coronal", "+dorsal", "+back")),
    ("uvular", ("-labial", "-coronal", "+dorsal", "+back")),
    ("glottal", ("-labial", "-coronal", "-dorsal")),
    ("close-mid", ("-high", "-low", "+tense")),
    ("open-mid", ("-high", "-low", "-tense")),
    ("near-close", ("+high", "-low", "-tense")),
    ("near-open", ("-high", "+low")),
    ("close", ("+high", "-low")),
    ("high", ("+high", "-low")),
    ("mid", ("-high", "-low")),
    ("open", ("-high", "+low")),
    ("low", ("-high", "+low")),
    ("front", ("-back",)),
    ("central", ("+back", "-round")),
    ("back", ("+back",)),
    ("unrounded", ("-round",)),
    ("rounded", ("+round",)),
    ("long", ("+long",)),
)


def _extract_ipa_fallback_features(sym: str) -> frozenset[str]:
    norm = unicodedata.normalize("NFC", sym).strip()
    if norm in _BUILTIN_IPA_FEATURES:
        return _BUILTIN_IPA_FEATURES[norm]
    # Strip length/tie-bar/modifiers and look up base IPA symbol
    stripped = "".join(
        ch for ch in norm if ch not in _SUPRASEGMENTAL_CHARS and unicodedata.category(ch) != "Lm"
    )
    feats: set[str] = set()
    if "ː" in norm:
        feats.add("+long")
    if "ʰ" in norm:
        feats.add("+spread_glottis")
    if "ʲ" in norm:
        feats.add("+palatalized")
    if "ʷ" in norm:
        feats.add("+round")
    if stripped in _BUILTIN_IPA_FEATURES:
        feats.update(_BUILTIN_IPA_FEATURES[stripped])
    elif stripped and stripped[0] in _BUILTIN_IPA_FEATURES:
        feats.update(_BUILTIN_IPA_FEATURES[stripped[0]])
    return frozenset(feats)


@dataclass(frozen=True)
class SoundInventory:
    """Binary phonological feature lookup loaded from `{unit_id}_sounds.csv` (plus IPA fallback)."""

    vc_map: Mapping[str, str] = field(default_factory=dict)
    feature_map: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: str | Path) -> SoundInventory:
        csv_path = Path(path)
        vc_map: dict[str, str] = {}
        feature_map: dict[str, frozenset[str]] = {}

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return cls()
            headers = list(reader.fieldnames)
            # Identify symbol ID column
            id_col = next(
                (
                    c
                    for c in headers
                    if c.lower() in {"sound_id", "id", "phoneme", "segment", "grapheme", "symbol"}
                ),
                headers[0],
            )
            desc_cols = [
                c
                for c in headers
                if c.lower()
                in {"name", "label", "description", "clts_name", "bipa", "type", "value", "comment"}
            ]
            ignore_cols = {id_col, *desc_cols, "clts_id", "ipa", "frequency", "count", "examples"}

            for row in reader:
                raw_sym = row.get(id_col, "")
                if not raw_sym:
                    continue
                sym = unicodedata.normalize("NFC", raw_sym).strip()
                if not sym:
                    continue

                feats: set[str] = set(_extract_ipa_fallback_features(sym))

                # Parse prose descriptions (CLTS / Paralex style: "voiced alveolar plosive", etc.)
                prose = " ".join(str(row.get(dc, "") or "").lower() for dc in desc_cols)
                prose_tokens = set(prose.replace(",", " ").replace(";", " ").split())
                for kw, implied in _PROSE_TO_FEATURES:
                    # Match whole hyphenated/word token to avoid "unrounded" matching "rounded"
                    if kw in prose_tokens:
                        for f_str in implied:
                            # Explicit prose overrides fallback sign on same feature name
                            fname = f_str[1:]
                            feats.discard("+" + fname)
                            feats.discard("-" + fname)
                            feats.add(f_str)

                # Parse explicit binary/categorical feature columns
                for col in headers:
                    if col in ignore_cols or col.lower() in ignore_cols:
                        continue
                    val = str(row.get(col, "") or "").strip()
                    if not val or val.lower() in {"na", "nan", "none", "null", "0.0"}:
                        continue
                    cname = col.strip().lower()
                    if val in {"+", "1", "true", "True", "yes"}:
                        feats.discard("-" + cname)
                        feats.add("+" + cname)
                    elif val in {"-", "0", "false", "False", "no"}:
                        feats.discard("+" + cname)
                        feats.add("-" + cname)
                    else:
                        feats.add(f"+{cname}={val.lower()}")

                # Determine V vs. C strictly from [+syllabic] / vowel indicators
                if "+syllabic" in feats:
                    vc_map[sym] = "V"
                elif "-syllabic" in feats:
                    vc_map[sym] = "C"
                else:
                    u_cls = _classify_segment_unicode(sym)
                    vc_map[sym] = u_cls if u_cls is not None else "C"
                    feats.add("+syllabic" if vc_map[sym] == "V" else "-syllabic")

                feature_map[sym] = frozenset(feats)

        return cls(vc_map=vc_map, feature_map=feature_map)

    def features_of(self, seg: str) -> frozenset[str]:
        norm = unicodedata.normalize("NFC", seg).strip()
        if norm in self.feature_map:
            return self.feature_map[norm]
        return _extract_ipa_fallback_features(norm)

    def vc_of(self, seg: str) -> str | None:
        norm = unicodedata.normalize("NFC", seg).strip()
        return self.vc_map.get(norm)


# ---------------------------------------------------------------------------
# Unified VowelClassifier (Unicode + `sounds.csv` + Sukhotin Fallback)
# ---------------------------------------------------------------------------


class VowelClassifier:
    """Classifies space-separated segments into `'V'` vs. `'C'` and projects source Vowel Tiers.

    Combines:
      1. Optional `{unit_id}_sounds.csv` (`[+syllabic]` / binary phonological features) in `phon` mode,
      2. Multi-script Unicode vowel character tables, and
      3. Sukhotin's (1962) co-occurrence vowel-discovery fallback for unknown scripts and
         ambiguous glides (`y`, `w`) fitted strictly on training forms.
    """

    def __init__(
        self,
        sound_inventory: SoundInventory | None = None,
        use_phon_features: bool = False,
    ) -> None:
        self.sound_inventory = sound_inventory
        self.use_phon_features = bool(use_phon_features or sound_inventory is not None)
        self._learned_vc: dict[str, str] = {}
        self._cache: dict[str, str] = {}
        self._seg_vowels_cache: dict[str, tuple[str, ...]] = {}
        self._feat_cache: dict[str, frozenset[str]] = {}
        self._pattern_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], object] = {}
        self._gen_env_cache: dict[tuple[object, object], object] = {}

    def fit(self, training_forms: Iterable[Sequence[str]]) -> VowelClassifier:
        """Fit Sukhotin co-occurrence statistics strictly on training forms for unknown/glide tokens."""
        forms_list = [tuple(f) for f in training_forms]
        vocab: set[str] = {tok for segs in forms_list for tok in segs if tok != "#"}

        known_vc: dict[str, str] = {}
        unresolved: set[str] = set()
        for tok in vocab:
            if self.sound_inventory is not None:
                sv = self.sound_inventory.vc_of(tok)
                if sv is not None:
                    known_vc[tok] = sv
                    continue
            u_cls = _classify_segment_unicode(tok)
            if u_cls is not None:
                known_vc[tok] = u_cls
            else:
                unresolved.add(tok)

        if unresolved:
            # Build neighbor counts against already-known V/C symbols first (exact Sukhotin step
            # conditioned on known anchors); if no symbols are known (pure unknown/cipher script),
            # run full unsupervised Sukhotin over all symbols!
            if not known_vc:
                sukhotin_v = sukhotin_discover_vowels(forms_list)
                for tok in unresolved:
                    known_vc[tok] = "V" if tok in sukhotin_v else "C"
            else:
                c_neighbors: dict[str, int] = defaultdict(int)
                v_neighbors: dict[str, int] = defaultdict(int)
                for segs in forms_list:
                    for idx, tok in enumerate(segs):
                        if tok not in unresolved:
                            continue
                        for nb_idx in (idx - 1, idx + 1):
                            if 0 <= nb_idx < len(segs):
                                nb = segs[nb_idx]
                                nb_cls = known_vc.get(nb)
                                if nb_cls == "C":
                                    c_neighbors[tok] += 1
                                elif nb_cls == "V":
                                    v_neighbors[tok] += 1

                # Also run full Sukhotin as tie-breaker when neighbors are all unresolved
                sukhotin_v = sukhotin_discover_vowels(forms_list)
                for tok in sorted(unresolved):
                    cn = c_neighbors[tok]
                    vn = v_neighbors[tok]
                    if cn + vn > 0:
                        if cn > vn:
                            known_vc[tok] = "V"
                        elif vn > cn:
                            known_vc[tok] = "C"
                        else:
                            # Exact tie: 'y' defaults to V if in Sukhotin V else C
                            known_vc[tok] = "V" if (tok in sukhotin_v or tok.lower() == "y") else "C"
                    else:
                        known_vc[tok] = "V" if (tok in sukhotin_v or tok.lower() == "y") else "C"

        self._learned_vc = known_vc
        self._cache = dict(known_vc)
        self._seg_vowels_cache.clear()
        self._feat_cache.clear()
        self._pattern_cache.clear()
        self._gen_env_cache.clear()
        return self

    def classify(self, seg: str) -> str:
        """Return `'V'` or `'C'` for a single segment token `seg`."""
        if seg == "#":
            return "#"
        cached = self._cache.get(seg)
        if cached is not None:
            return cached

        norm = unicodedata.normalize("NFC", seg).strip()
        if norm in self._learned_vc:
            res = self._learned_vc[norm]
            self._cache[seg] = res
            return res

        if self.sound_inventory is not None:
            sv = self.sound_inventory.vc_of(norm)
            if sv is not None:
                self._cache[seg] = sv
                return sv

        u_cls = _classify_segment_unicode(norm)
        if u_cls is not None:
            self._cache[seg] = u_cls
            return u_cls

        # Unseen ambiguous glide fallback: 'y' defaults to 'V' in isolation unless learned otherwise;
        # unseen unknown symbols default to 'C'.
        res = "V" if norm.lower() == "y" else "C"
        self._cache[seg] = res
        return res

    def classify_sequence(self, segs: Sequence[str]) -> tuple[str, ...]:
        """Classify each segment of `segs` into `'V'` or `'C'`."""
        return tuple(self.classify(s) for s in segs)

    def segment_vowels(self, seg: str) -> tuple[str, ...]:
        """Return Vowel-Tier symbol(s) projected by a single local segment token `seg`.

        - Vocalic segments (`classify(seg) == 'V'`, including `VH1, VH2, ...`): `(seg,)`.
        - Precomposed NFC Hangul syllable blocks (`0xAC00 <= ord(ch) <= 0xD7A3`): extracts the
          medial Jungseong vowel jamo `chr(0x1161 + ((ord(ch) - 0xAC00) // 28) % 21)` while keeping
          the full syllable token intact on the local tier.
        - Consonants, `#`, and stem polarity tags (`HNEG`, `HPOS`, `HNEUT`): `()`.
        """
        if seg == "#":
            return ()
        cached = self._seg_vowels_cache.get(seg)
        if cached is not None:
            return cached
        if self.classify(seg) == "V":
            res: tuple[str, ...] = (seg,)
        else:
            norm = unicodedata.normalize("NFC", seg).strip()
            res = tuple(
                chr(0x1161 + ((ord(ch) - 0xAC00) // 28) % 21)
                for ch in norm
                if 0xAC00 <= ord(ch) <= 0xD7A3
            )
        self._seg_vowels_cache[seg] = res
        return res

    def project_vowel_tier(self, source_segs: Sequence[str]) -> tuple[str, ...]:
        """Project ONLY the source word `source_segs` (`w_s`) onto its Vowel Tier `(v_1, ..., v_k)`."""
        out: list[str] = []
        for s in source_segs:
            if s != "#":
                out.extend(self.segment_vowels(s))
        return tuple(out)

    def segment_features(self, seg: str) -> frozenset[str]:
        """Return binary feature bundle for `seg`: full distinctive features in `phon` mode,
        or major-class `{'+syllabic'}` / `{'-syllabic'}` in `orth` mode.
        """
        if seg == "#":
            return frozenset()
        cached_f = self._feat_cache.get(seg)
        if cached_f is not None:
            return cached_f
        vc = self.classify(seg)
        major = "+syllabic" if vc == "V" else "-syllabic"
        if not self.use_phon_features:
            res = frozenset({major})
            self._feat_cache[seg] = res
            return res

        feats = (
            self.sound_inventory.features_of(seg)
            if self.sound_inventory is not None
            else _extract_ipa_fallback_features(seg)
        )
        if not feats:
            res = frozenset({major})
        elif "+syllabic" not in feats and "-syllabic" not in feats:
            res = feats | {major}
        else:
            res = feats
        self._feat_cache[seg] = res
        return res
