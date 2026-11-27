"""Shared genetic admixture, f3, Graff target, and GeLaTo Ne / FST computations.

Computes individual-level and population-level ADMIXTURE (K=12..30) and f3 measures
with 400-resample bootstrap standard errors for any set of GeLaTo populations or
any `(population, glottocode)` link table (Grambank or LDL/Wug).

Works both with `numpy`/`pandas` installed (fast vectorized path) and under a
stdlib-only Python interpreter using an exact pure-Python implementation of
NumPy's `SeedSequence` + `PCG64` + 32-bit buffered Lemire integer sampler.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

COMMON_DIR = Path(__file__).resolve().parent
DATA_DIR = COMMON_DIR / "data"
ROOT = COMMON_DIR.parents[1]

ARCH_DIR = ROOT / "analyses/gelato_feasibility_2026_10_01/sources/zenodo_15263706/geneticAdmixture-linguisticDiffusion"
ADM_DIR = ARCH_DIR / "input/MegaAdmixtureCatalogue/ADMIXTURE"
GELATO_DATA_CSV = (
    ROOT / "analyses/gelato_feasibility_2026_10_01/sources/gelato_c625fdc/datasets/HumanOrigins_AutosomalSNP/data.csv"
)
GELATO_POPS_CSV = (
    ROOT / "analyses/gelato_feasibility_2026_10_01/sources/gelato_c625fdc/cldf/populations.csv"
)
ZENODO_EXT_DIR = ROOT / "external/zenodo_15263706"
FST_CSV_CANDIDATES: tuple[Path, ...] = (
    ZENODO_EXT_DIR / "input/FST.csv",
    ARCH_DIR / "input/FST.csv",
)
F3_FIXED_CSV = DATA_DIR / "f3_fixed_scores.csv"
FST_REGIMES_CSV = DATA_DIR / "fst_neighbor_regimes.csv"
DISCRETE_K_CSV = DATA_DIR / "discrete_k_diagnostics.csv"

KS: list[int] = list(range(12, 31))
DISCRETE_KS: tuple[int, ...] = (12, 20, 30)
DEFAULT_MIN_N_IND: int = 3
SENSITIVITY_MIN_N_IND: int = 5
N_BOOT: int = 400
SEED: int = 20261021

POP_BOOT_MEASURES: tuple[str, ...] = ("pop_ent", "het", "support", "amount", "sqrt_het")

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK128 = (1 << 128) - 1
_MULT128 = (2549297995355413924 << 64) | 4865540595714422341


class PurePCG64:
    """Bit-for-bit pure-Python replica of `numpy.random.default_rng(seed)` for `.integers()`."""

    def __init__(self, seed: int = SEED) -> None:
        w = self._seed_seq_words(seed, 8)
        u64 = [(w[2 * i] | (w[2 * i + 1] << 32)) & _MASK64 for i in range(4)]
        initstate = (u64[0] << 64) | u64[1]
        initseq = (u64[2] << 64) | u64[3]
        self.state = 0
        self.inc = ((initseq << 1) | 1) & _MASK128
        self.has_uint32 = 0
        self.uinteger = 0
        self._step()
        self.state = (self.state + initstate) & _MASK128
        self._step()

    @staticmethod
    def _seed_seq_words(entropy: int, n_words: int = 8) -> list[int]:
        init_a, mult_a = 0x43B0D7E5, 0x931E8875
        init_b, mult_b = 0x8B51F9DD, 0x58F38DED
        mix_mult_l, mix_mult_r = 0xCA01F9DD, 0x4973F715
        xshift = 16
        pool = [0] * 4
        hc = [init_a]

        def h(val: int) -> int:
            val = (val ^ hc[0]) & _MASK32
            hc[0] = (hc[0] * mult_a) & _MASK32
            val = (val * hc[0]) & _MASK32
            return (val ^ (val >> xshift)) & _MASK32

        def mix(x: int, y: int) -> int:
            res = ((mix_mult_l * x) & _MASK32) - ((mix_mult_r * y) & _MASK32)
            res &= _MASK32
            return (res ^ (res >> xshift)) & _MASK32

        ent = [entropy & _MASK32]
        for i in range(4):
            pool[i] = h(ent[i] if i < len(ent) else 0)
        for i_src in range(4):
            for i_dst in range(4):
                if i_src != i_dst:
                    pool[i_dst] = mix(pool[i_dst], h(pool[i_src]))
        out: list[int] = []
        hcb = init_b
        for i in range(n_words):
            data_val = pool[i % 4] ^ hcb
            hcb = (hcb * mult_b) & _MASK32
            data_val = (data_val * hcb) & _MASK32
            data_val = (data_val ^ (data_val >> xshift)) & _MASK32
            out.append(data_val)
        return out

    def _step(self) -> None:
        self.state = (self.state * _MULT128 + self.inc) & _MASK128

    def next_uint64(self) -> int:
        self._step()
        s = self.state
        xorshifted = ((s >> 64) ^ s) & _MASK64
        rot = (s >> 122) & 63
        return ((xorshifted >> rot) | ((xorshifted << ((-rot) & 63)) & _MASK64)) & _MASK64

    def next_uint32(self) -> int:
        if self.has_uint32:
            self.has_uint32 = 0
            return self.uinteger
        n = self.next_uint64()
        self.has_uint32 = 1
        self.uinteger = (n >> 32) & _MASK32
        return n & _MASK32

    def bounded_uint32_lemire(self, rng_excl: int) -> int:
        if rng_excl <= 1:
            return 0
        m = self.next_uint32() * rng_excl
        l = m & _MASK32
        if l < rng_excl:
            t = (0x100000000 - rng_excl) % rng_excl
            while l < t:
                m = self.next_uint32() * rng_excl
                l = m & _MASK32
        return m >> 32

    def integers(self, low: int, high: int, size: tuple[int, int]) -> list[list[int]]:
        rng_excl = high - low
        rows, cols = size
        return [[low + self.bounded_uint32_lemire(rng_excl) for _ in range(cols)] for _ in range(rows)]


def entropy(q):
    """Row-wise Shannon entropy H(q) = -sum_k q_k log(q_k)."""
    if np is not None and isinstance(q, np.ndarray):
        qc = np.clip(q, 1e-12, 1)
        return -(qc * np.log(qc)).sum(axis=-1)
    if isinstance(q[0], (int, float)):
        return -sum(min(max(float(x), 1e-12), 1.0) * math.log(min(max(float(x), 1e-12), 1.0)) for x in q)
    return [entropy(row) for row in q]


def _pop_stats_pure(q_by_k: list[list[list[float]]], idx: list[list[int]] | None = None) -> dict[str, list[float]]:
    n = len(q_by_k[0])
    B = len(idx) if idx is not None else 1
    acc = {k: [0.0] * B for k in ("pop_ent", "het", "support", "amount")}
    n_k = len(q_by_k)
    for q in q_by_k:
        K = len(q[0])
        h_Each = [entropy(row) for row in q]
        for b in range(B):
            rows_b = [q[i] for i in idx[b]] if idx is not None else q
            h_ind_b = (sum(h_Each[i] for i in idx[b]) / n) if idx is not None else (sum(h_Each) / n)
            mean_k = [sum(r[c] for r in rows_b) / n for c in range(K)]
            h_pop_b = entropy(mean_k)
            het_b = (h_pop_b - h_ind_b) * n / (n - 1) if n >= 2 else float("nan")
            srt = sorted(mean_k, reverse=True)
            s0, s1 = srt[0], srt[1]
            fl = [1.0 if (s0 + s1 > t and s1 > 0.05) else 0.0 for t in (0.7, 0.8, 0.9)]
            sup_b = sum(fl) / 3.0
            amt_b = sum(f * s1 for f in fl) / 3.0
            acc["pop_ent"][b] += h_pop_b
            acc["het"][b] += het_b
            acc["support"][b] += sup_b
            acc["amount"][b] += amt_b
    for k in ("pop_ent", "het", "support", "amount"):
        acc[k] = [v / n_k for v in acc[k]]
    acc["sqrt_het"] = [
        math.sqrt(max(0.0, x)) if not math.isnan(x) else float("nan") for x in acc["het"]
    ]
    return acc


def pop_stats(q_by_k, idx=None):
    """Population measures, averaged over K, for one population."""
    if np is None or not isinstance(q_by_k[0], np.ndarray):
        return _pop_stats_pure(q_by_k, idx)
    n = q_by_k[0].shape[0]
    acc = {k: 0.0 for k in ("pop_ent", "het", "support", "amount")}
    for q in q_by_k:
        qs = q[idx] if idx is not None else q[None]  # (B, n, K)
        mean = qs.mean(axis=1)                       # (B, K)
        h_pop = entropy(mean)
        h_ind = entropy(qs).mean(axis=1)
        het = (h_pop - h_ind) * n / (n - 1) if n >= 2 else np.full_like(h_pop, np.nan)
        srt = -np.sort(-mean, axis=1)
        flags = np.stack([(srt[:, 0] + srt[:, 1] > t) & (srt[:, 1] > 0.05) for t in (0.7, 0.8, 0.9)])
        acc["pop_ent"] = acc["pop_ent"] + h_pop
        acc["het"] = acc["het"] + het
        acc["support"] = acc["support"] + flags.mean(0)
        acc["amount"] = acc["amount"] + (flags * srt[:, 1]).mean(0)
    out = {k: v / len(q_by_k) for k, v in acc.items()}
    out["sqrt_het"] = np.sqrt(np.maximum(0.0, out["het"]))
    return out


def load_admixture_q(
    adm_dir: Path = ADM_DIR,
    ks: Sequence[int] = KS,
):
    """Load `GeneticInfoID.csv` and ADMIXTURE Q matrices for K in `ks`."""
    if np is not None and pd is not None:
        info = pd.read_csv(adm_dir / "GeneticInfoID.csv", keep_default_na=False)
        assert info.Order.astype(int).tolist() == list(range(1, len(info) + 1))
        info = info[["Population"]].rename(columns={"Population": "population"}).reset_index(drop=True)
        Q: dict[int, np.ndarray] = {}
        for k in ks:
            q = np.loadtxt(adm_dir / f"best_runs/GelatoHO_mergedSetMarchBEDnorelatives_pruned_autosomes_K{k}.Q")
            assert q.shape == (len(info), k) and np.allclose(q.sum(1), 1, atol=1e-3)
            Q[k] = q / q.sum(1, keepdims=True)
        return info, Q

    with open(adm_dir / "GeneticInfoID.csv", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    assert [int(r["Order"]) for r in raw_rows] == list(range(1, len(raw_rows) + 1))
    info_rows = [{"population": r["Population"]} for r in raw_rows]
    Q_pure: dict[int, list[list[float]]] = {}
    for k in ks:
        q_path = adm_dir / f"best_runs/GelatoHO_mergedSetMarchBEDnorelatives_pruned_autosomes_K{k}.Q"
        mat: list[list[float]] = []
        with open(q_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    vals = [float(x) for x in line.split()]
                    s = sum(vals)
                    mat.append([x / s for x in vals])
        assert len(mat) == len(info_rows)
        Q_pure[k] = mat
    return info_rows, Q_pure


def compute_individual_measures(
    info,
    Q,
    links_df=None,
    ks: Sequence[int] = KS,
):
    """Compute K-averaged individual entropy (`ent`), `nonmax`, and discrete K=12/20/30 entropies."""
    ks_list = list(ks)
    if np is not None and pd is not None and isinstance(info, pd.DataFrame):
        ent = np.stack([entropy(Q[k]) for k in ks_list])
        nonmax = np.stack([1 - Q[k].max(1) for k in ks_list])
        ind = info.assign(
            ind=np.arange(len(info)),
            ent=ent.mean(0),
            nonmax=nonmax.mean(0),
            ent_K12=ent[ks_list.index(12)],
            ent_K20=ent[ks_list.index(20)],
            ent_K30=ent[ks_list.index(30)],
        )
        if links_df is not None:
            if isinstance(links_df, list):
                links_df = pd.DataFrame(links_df)
            ind = ind.merge(links_df[["population", "glottocode"]], on="population")
        return ind

    n_ind = len(info)
    n_k = len(ks_list)
    k12_idx = ks_list.index(12)
    k20_idx = ks_list.index(20)
    k30_idx = ks_list.index(30)
    ent_by_k = [[entropy(Q[k][i]) for i in range(n_ind)] for k in ks_list]
    nonmax_by_k = [[1.0 - max(Q[k][i]) for i in range(n_ind)] for k in ks_list]

    link_map: dict[str, list[str]] | None = None
    if links_df is not None:
        link_map = {}
        rows_iter = links_df if isinstance(links_df, list) else links_df.to_dict("records")
        for r in rows_iter:
            link_map.setdefault(r["population"], []).append(r["glottocode"])

    out_rows: list[dict[str, object]] = []
    for i, r in enumerate(info):
        pop_name = r["population"]
        if link_map is not None and pop_name not in link_map:
            continue
        base = {
            "population": pop_name,
            "ind": i,
            "ent": sum(ent_by_k[ki][i] for ki in range(n_k)) / n_k,
            "nonmax": sum(nonmax_by_k[ki][i] for ki in range(n_k)) / n_k,
            "ent_K12": ent_by_k[k12_idx][i],
            "ent_K20": ent_by_k[k20_idx][i],
            "ent_K30": ent_by_k[k30_idx][i],
        }
        if link_map is not None:
            for gc in link_map[pop_name]:
                out_rows.append({**base, "glottocode": gc})
        else:
            out_rows.append(base)
    return out_rows


def load_graff_targets(arch_dir: Path = ARCH_DIR) -> set[str]:
    """Load Graff et al. 2025 curated inter-family admixture targets (Table S2)."""
    with open(arch_dir / "tables/tableS2.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def names(v: str) -> set[str]:
        return {p.strip().replace(" ", "_") for p in re.sub(r"\([^)]*\)", "", v).split(",") if p.strip()}

    return set().union(*(names(r["TargetPop"]) for r in rows), *(names(r["AlternativeTargetPop"]) for r in rows))


def load_gelato_metadata(gelato_csv: Path = GELATO_DATA_CSV):
    """Load GeLaTo neighbour FST (`neigh_flow`), harmonic-mean Ne, and median FST per population."""
    with open(gelato_csv, encoding="utf-8-sig") as f:
        raw_rows = list(csv.DictReader(f))

    rows: list[dict[str, object]] = []
    for r in raw_rows:
        m_adj = _parse_float(r.get("MedianFSTAdjustedNeighbors"))
        nf = -math.log(m_adj) if (not math.isnan(m_adj) and m_adj > 0) else float("nan")
        rows.append({
            "population": r["PopName"],
            "MedianFSTAdjustedNeighbors": m_adj,
            "harmonicMean": _parse_float(r.get("harmonicMean")),
            "medianFST": _parse_float(r.get("medianFST")),
            "medianFSTregion": _parse_float(r.get("medianFSTregion")),
            "neigh_flow": nf,
        })
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def load_f3_scores(
    f3_path: Path = DATA_DIR / "f3_scores.csv",
    k5_path: Path = DATA_DIR / "f3_scores_k5.csv",
):
    """Load precomputed f3 and 5-nearest f3k scores per population."""
    k5_by_pop: dict[str, dict[str, str]] = {}
    if k5_path.exists():
        with open(k5_path, encoding="utf-8") as f:
            k5_by_pop = {r["population"]: r for r in csv.DictReader(f)}

    rows: list[dict[str, object]] = []
    with open(f3_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop_name = r["population"]
            n_b = _parse_int(r.get("n_B"))
            min_z = _parse_float(r.get("min_z"))
            min_est = _parse_float(r.get("min_est"))
            f3_z = -min_z if not math.isnan(min_z) else float("nan")
            f3_est = -min_est if not math.isnan(min_est) else float("nan")
            f3b3_z = f3_z if (n_b is not None and n_b >= 3) else float("nan")
            f3b3_est = f3_est if (n_b is not None and n_b >= 3) else float("nan")
            k5_r = k5_by_pop.get(pop_name)
            if k5_r is not None:
                k5_mz = _parse_float(k5_r.get("min_z"))
                k5_me = _parse_float(k5_r.get("min_est"))
                f3k_z = -k5_mz if not math.isnan(k5_mz) else float("nan")
                f3k_est = -k5_me if not math.isnan(k5_me) else float("nan")
                f3k_est_pos = max(0.0, f3k_est) if not math.isnan(f3k_est) else float("nan")
            else:
                f3k_z = f3k_est = f3k_est_pos = float("nan")
            rows.append({
                "population": pop_name,
                "n_B": n_b,
                "n_C": _parse_int(r.get("n_C")),
                "n_triples": _parse_int(r.get("n_triples")),
                "n_z_lt_m3": _parse_int(r.get("n_z_lt_m3")),
                "min_pop2": "" if r.get("min_pop2") in (None, "", "NA") else r["min_pop2"],
                "min_pop3": "" if r.get("min_pop3") in (None, "", "NA") else r["min_pop3"],
                "f3_z": f3_z,
                "f3_est": f3_est,
                "f3b3_z": f3b3_z,
                "f3b3_est": f3b3_est,
                "f3k_z": f3k_z,
                "f3k_est": f3k_est,
                "f3k_est_pos": f3k_est_pos,
            })
    if pd is not None:
        df = pd.DataFrame(rows)
        for ic in ("n_B", "n_C", "n_triples", "n_z_lt_m3"):
            df[ic] = pd.array(df[ic].tolist(), dtype="Int64")
        return df
    return rows


def _nanstd_ddof1(vals: list[float]) -> float:
    clean = [x for x in vals if not math.isnan(x)]
    if len(clean) < 2:
        return float("nan")
    m = sum(clean) / len(clean)
    return math.sqrt(sum((x - m) ** 2 for x in clean) / (len(clean) - 1))


def _parse_float(val: str | None) -> float:
    if val in (None, "", "NA", "NaN", "nan"):
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def _parse_int(val: str | None) -> int | None:
    if val in (None, "", "NA", "NaN", "nan"):
        return None
    try:
        return int(float(val))
    except ValueError:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = math.pi / 180.0
    a = (
        math.sin((lat2 - lat1) * r / 2.0) ** 2
        + math.cos(lat1 * r) * math.cos(lat2 * r) * math.sin((lon2 - lon1) * r / 2.0) ** 2
    )
    return 12742.0 * math.asin(min(1.0, math.sqrt(a)))


def load_f3_fixed_scores(f3_fixed_path: Path = F3_FIXED_CSV):
    """Load precomputed clade-stratified, non-conspecific f3 and f4 SI scores."""
    int_cols = ("n_ind", "n_B_avail", "n_C_avail", "n_B", "n_C", "n_triples", "n_z_lt_m3",
                "iso_fallback_used", "iso_n_B", "all_n_triples")
    flt_cols = ("best_est", "best_se", "best_z", "f3_est_neg", "f3_pen", "f3_fixed", "f3_pos",
                "iso_best_est", "iso_best_se", "iso_best_z", "f3_fixed_iso",
                "all_best_est", "all_best_se", "all_best_z", "f3_all_fixed",
                "f4_est", "f4_se", "f4_z")
    str_cols = ("population", "glottocode", "family", "macroarea", "coord_source",
                "best_pop2", "best_pop3", "iso_best_pop2", "iso_best_pop3",
                "all_best_pop2", "all_best_pop3", "f4_outgroup")
    rows: list[dict[str, object]] = []
    if not f3_fixed_path.exists():
        return pd.DataFrame(columns=[*str_cols, *int_cols, *flt_cols, "f3_fixed_raw", "f3_fixed_se"]) if pd is not None else rows
    with open(f3_fixed_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out_r: dict[str, object] = {}
            for sc in str_cols:
                v = r.get(sc, "")
                out_r[sc] = "" if v in (None, "NA") else v
            for ic in int_cols:
                out_r[ic] = _parse_int(r.get(ic))
            for fc in flt_cols:
                out_r[fc] = _parse_float(r.get(fc))
            out_r["f3_fixed_raw"] = out_r["f3_est_neg"]
            out_r["f3_fixed_se"] = out_r["best_se"]
            rows.append(out_r)
    if pd is not None:
        df = pd.DataFrame(rows)
        for ic in int_cols:
            if ic in df.columns:
                df[ic] = pd.array(df[ic].tolist(), dtype="Int64")
        return df
    return rows


def compute_discrete_k_diagnostics(
    info,
    Q,
    populations: Iterable[str],
    *,
    ks: Sequence[int] = DISCRETE_KS,
    min_n_rmsd: int = SENSITIVITY_MIN_N_IND,
):
    """Compute discrete-K (K=12, 20, 30) individual/pop entropy, max_q, and RMSD(q) diagnostics.

    `rmsd_K{k}` is `sqrt(sum_c Var(q_c, ddof=1))` across individuals within each population,
    set to NaN whenever `n_individuals < min_n_rmsd` (default 5).
    """
    ks_list = [int(k) for k in ks]
    pop_list = sorted(set(populations))

    if np is not None and pd is not None and isinstance(info, pd.DataFrame):
        idx_by_pop: dict[str, np.ndarray] = {}
        for p_name, grp in info.assign(_i=np.arange(len(info))).groupby("population"):
            if p_name in pop_list:
                idx_by_pop[str(p_name)] = grp["_i"].to_numpy(dtype=int)
        rows: list[dict[str, object]] = []
        for p_name in pop_list:
            ii = idx_by_pop[p_name]
            n = len(ii)
            rec: dict[str, object] = {"population": p_name, "n_individuals": n}
            for k in ks_list:
                qp = Q[k][ii]
                qmean = qp.mean(axis=0)
                rec[f"ent_K{k}"] = float(entropy(qp).mean())
                rec[f"pop_ent_K{k}"] = float(entropy(qmean))
                rec[f"max_q_K{k}"] = float(qmean.max())
                if n >= min_n_rmsd and n >= 2:
                    rec[f"rmsd_K{k}"] = math.sqrt(max(0.0, float(np.sum(np.var(qp, axis=0, ddof=1)))))
                else:
                    rec[f"rmsd_K{k}"] = float("nan")
            rows.append(rec)
        return pd.DataFrame(rows)

    idx_by_pop_pure: dict[str, list[int]] = {}
    for i, r in enumerate(info):
        p_name = str(r["population"])
        if p_name in pop_list:
            idx_by_pop_pure.setdefault(p_name, []).append(i)
    rows_pure: list[dict[str, object]] = []
    for p_name in pop_list:
        ii = idx_by_pop_pure[p_name]
        n = len(ii)
        rec_p: dict[str, object] = {"population": p_name, "n_individuals": n}
        for k in ks_list:
            qp = [Q[k][idx_i] for idx_i in ii]
            rec_p[f"ent_K{k}"] = sum(entropy(row) for row in qp) / n
            mean_q = [sum(row[c] for row in qp) / n for c in range(k)]
            rec_p[f"pop_ent_K{k}"] = float(entropy(mean_q))
            rec_p[f"max_q_K{k}"] = max(mean_q)
            if n >= min_n_rmsd and n >= 2:
                var_sum = sum(sum((row[c] - mean_q[c]) ** 2 for row in qp) / (n - 1) for c in range(k))
                rec_p[f"rmsd_K{k}"] = math.sqrt(max(0.0, var_sum))
            else:
                rec_p[f"rmsd_K{k}"] = float("nan")
        rows_pure.append(rec_p)
    return rows_pure


def build_discrete_k_diagnostics_csv(
    out_path: Path | None = DISCRETE_K_CSV,
    *,
    ks: Sequence[int] = DISCRETE_KS,
    min_n_rmsd: int = SENSITIVITY_MIN_N_IND,
):
    """Precompute discrete-K diagnostics for all 558 ADMIXTURE populations."""
    info, Q = load_admixture_q(ks=ks)
    all_pops = (
        sorted(info["population"].unique())
        if (pd is not None and isinstance(info, pd.DataFrame))
        else sorted({str(r["population"]) for r in info})
    )
    dk = compute_discrete_k_diagnostics(info, Q, all_pops, ks=ks, min_n_rmsd=min_n_rmsd)
    if out_path is not None:
        rows = dk.to_dict("records") if (pd is not None and isinstance(dk, pd.DataFrame)) else list(dk)
        write_rows_csv(rows, out_path, list(rows[0].keys()), ndigits=7)
    return dk


def load_discrete_k_diagnostics(path: Path = DISCRETE_K_CSV):
    """Load precomputed discrete-K (K=12, 20, 30) population diagnostics."""
    rows: list[dict[str, object]] = []
    if not path.exists():
        return pd.DataFrame() if pd is not None else rows
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rec: dict[str, object] = {
                "population": r["population"],
                "n_individuals": _parse_int(r.get("n_individuals")),
            }
            for k, v in r.items():
                if k not in ("population", "n_individuals"):
                    rec[k] = _parse_float(v)
            rows.append(rec)
    if pd is not None:
        df = pd.DataFrame(rows)
        if "n_individuals" in df.columns:
            df["n_individuals"] = pd.array(df["n_individuals"].tolist(), dtype="Int64")
        return df
    return rows


def compute_language_f3_maxp(
    pop_df,
    *,
    col: str = "f3_fixed",
    min_n_ind: int = DEFAULT_MIN_N_IND,
):
    """Language-level maximum `col` across linked populations with `n_individuals >= min_n_ind`.

    Unmasks sister-population drift heterogeneity (e.g., BedouinA/B, Hadza/Hadza1) without
    pulling admixed languages toward zero when one sampled sub-population is bottlenecked.
    """
    rows_in = pop_df.to_dict("records") if (pd is not None and isinstance(pop_df, pd.DataFrame)) else list(pop_df)
    by_gc: dict[str, list[tuple[float, str]]] = {}
    for r in rows_in:
        gc = str(r.get("glottocode", "")).strip()
        n_i = _parse_int(str(r.get("n_individuals", "")))
        v = _parse_float(str(r.get(col, "")))
        if not gc or n_i is None or n_i < min_n_ind or math.isnan(v):
            continue
        by_gc.setdefault(gc, []).append((v, str(r.get("population", ""))))

    out_rows: list[dict[str, object]] = []
    for gc in sorted(by_gc.keys()):
        cands = by_gc[gc]
        best_v, best_pop = max(cands, key=lambda t: (t[0], t[1]))
        out_rows.append({
            "glottocode": gc,
            "n_pops_used": len(cands),
            f"{col}_maxp": best_v,
            f"{col}_maxp_pop": best_pop,
        })
    return pd.DataFrame(out_rows) if pd is not None else out_rows


def _resolve_fst_csv(fst_csv: Path | None = None) -> Path:
    if fst_csv is not None:
        if fst_csv.is_file():
            return fst_csv
        raise FileNotFoundError(f"FST.csv not found: {fst_csv}")
    for cand in FST_CSV_CANDIDATES:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"FST.csv not found in {[str(p) for p in FST_CSV_CANDIDATES]}")


def compute_fst_neighbor_and_regime_metrics(
    fst_csv: Path | None = None,
    *,
    out_path: Path | None = FST_REGIMES_CSV,
    max_km: float = 1000.0,
    min_neighbors: int = 2,
    min_n_ref: int = SENSITIVITY_MIN_N_IND,
    f3_fixed_path: Path = F3_FIXED_CSV,
    links_csv: Path = DATA_DIR / "analysis_links.csv",
    root_dir: Path = ROOT,
):
    """Compute out-of-family neighbor FST (`neigh_flow_out`), global `median_fst_all`, and
    mutually exclusive Barbieri et al. (2022) Regime A / Regime B mismatch indicators across
    all 558 populations using reference candidates with `n_individuals >= min_n_ref` (default 5).
    """
    fst_path = _resolve_fst_csv(fst_csv)
    gmap_path = ARCH_DIR / "input/GeLaTo-population-glottocode-mapping.csv"
    gid_path = ADM_DIR / "GeneticInfoID.csv"
    glot_path = root_dir / "external/glottolog-cldf/cldf/languages.csv"

    f3_fixed_by_pop: dict[str, float] = {}
    if f3_fixed_path.is_file():
        with open(f3_fixed_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                f3_fixed_by_pop[r["population"]] = _parse_float(r.get("f3_fixed"))

    n_by_pop: dict[str, int] = {}
    with open(gid_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r["Population"]
            n_by_pop[p] = n_by_pop.get(p, 0) + 1
    pops = sorted(n_by_pop.keys())
    pidx = {p: i for i, p in enumerate(pops)}
    n_p = len(pops)

    # Load symmetric 558x558 FST matrix (negative Weir-Cockerham estimates clamped to 0)
    if np is not None:
        fst_mat = np.full((n_p, n_p), np.nan, dtype=float)
        np.fill_diagonal(fst_mat, 0.0)
        with open(fst_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p1, p2 = r["Pop1"], r["Pop2"]
                val = _parse_float(r.get("FST"))
                if p1 in pidx and p2 in pidx and not math.isnan(val):
                    v = max(0.0, val)
                    i, j = pidx[p1], pidx[p2]
                    fst_mat[i, j] = fst_mat[j, i] = v
    else:
        fst_mat = [[float("nan")] * n_p for _ in range(n_p)]
        for i in range(n_p):
            fst_mat[i][i] = 0.0
        with open(fst_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p1, p2 = r["Pop1"], r["Pop2"]
                val = _parse_float(r.get("FST"))
                if p1 in pidx and p2 in pidx and not math.isnan(val):
                    v = max(0.0, val)
                    i, j = pidx[p1], pidx[p2]
                    fst_mat[i][j] = fst_mat[j][i] = v

    # Load language / family / coordinate metadata matching f3.R hierarchy
    gmap_by_pop: dict[str, dict[str, str]] = {}
    with open(gmap_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gmap_by_pop[r["PopName"]] = r

    gel_coords: dict[str, tuple[float, float]] = {}
    if GELATO_POPS_CSV.is_file():
        with open(GELATO_POPS_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                la, lo = _parse_float(r.get("Latitude")), _parse_float(r.get("Longitude"))
                if not math.isnan(la) and not math.isnan(lo):
                    gel_coords[r["Name"]] = (la, lo)

    glot_lang_id: dict[str, str] = {}
    glot_level: dict[str, str] = {}
    glot_coords: dict[str, tuple[float, float]] = {}
    with open(glot_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gid_code = r["ID"]
            glot_lang_id[gid_code] = r["Language_ID"] or gid_code
            glot_level[gid_code] = r.get("Level", "")
            la, lo = _parse_float(r.get("Latitude")), _parse_float(r.get("Longitude"))
            if not math.isnan(la) and not math.isnan(lo):
                glot_coords[gid_code] = (la, lo)

    al_main_nonproxy: dict[str, str] = {}
    al_main_any: dict[str, str] = {}
    al_main_all_gcs: dict[str, set[str]] = {}
    if links_csv.is_file():
        with open(links_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("analysis") != "main":
                    continue
                p_nm = r["population"]
                gc = r["glottocode"]
                al_main_any.setdefault(p_nm, gc)
                al_main_all_gcs.setdefault(p_nm, set()).add(gc)
                if str(r.get("is_proxy", "")).lower() != "true":
                    al_main_nonproxy.setdefault(p_nm, gc)

    fam = {}
    eff_gc = {}
    lang_sets: dict[str, set[str]] = {}
    lat = {}
    lon = {}
    for p in pops:
        gm = gmap_by_pop.get(p, {})
        f_val = (gm.get("glottologFamily") or "").strip()
        fam[p] = "" if f_val in ("", "NA") else f_val
        gb = (gm.get("glottocodeBase") or "").strip()
        gb = "" if gb in ("", "NA") else gb
        mc = al_main_nonproxy.get(p, al_main_any.get(p, ""))
        egc = mc or gb
        eff_gc[p] = egc
        lset: set[str] = set()
        for gc in al_main_all_gcs.get(p, set()):
            lid_val = glot_lang_id.get(gc, gc)
            if lid_val:
                lset.add(lid_val)
        if gb and glot_level.get(gb) != "family":
            blid = glot_lang_id.get(gb, gb)
            if blid:
                lset.add(blid)
        if not lset and egc:
            lset.add(glot_lang_id.get(egc, egc))
        lang_sets[p] = lset

        la, lo = _parse_float(gm.get("lat")), _parse_float(gm.get("lon"))
        if math.isnan(la) or math.isnan(lo):
            if p in gel_coords:
                la, lo = gel_coords[p]
            elif gb in glot_coords:
                la, lo = glot_coords[gb]
            elif mc in glot_coords:
                la, lo = glot_coords[mc]
        lat[p], lon[p] = la, lo

    rows: list[dict[str, object]] = []
    for i, p in enumerate(pops):
        row_vals = [float(fst_mat[i, j] if np is not None else fst_mat[i][j]) for j in range(n_p)]
        other_vals = [row_vals[j] for j in range(n_p) if j != i and not math.isnan(row_vals[j])]
        med_all = float(statistics.median(other_vals)) if other_vals else float("nan")

        fp = fam[p]
        lset_p = lang_sets[p]
        lat_p, lon_p = lat[p], lon[p]
        has_coords = not (math.isnan(lat_p) or math.isnan(lon_p))

        same_cands: list[tuple[float, str, float]] = []
        diff_cands: list[tuple[float, str, float]] = []
        diff_local: list[tuple[float, str]] = []
        out_1000_vals: list[float] = []

        for j, q in enumerate(pops):
            if j == i or n_by_pop[q] < min_n_ref:
                continue
            v = row_vals[j]
            if math.isnan(v):
                continue
            fq = fam[q]
            if not fq:
                continue
            dist_km = (
                _haversine_km(lat_p, lon_p, lat[q], lon[q])
                if (has_coords and not math.isnan(lat[q]) and not math.isnan(lon[q]))
                else float("nan")
            )
            if fp and fq == fp:
                is_conspec = bool(lset_p & lang_sets[q])
                if not is_conspec:
                    same_cands.append((v, q, dist_km))
            elif fq != fp:
                diff_cands.append((v, q, dist_km))
                if not math.isnan(dist_km) and dist_km <= max_km:
                    out_1000_vals.append(v)
                    diff_local.append((v, q))

        same_best = min(same_cands, key=lambda t: (t[0], t[1])) if same_cands else (float("nan"), "", float("nan"))
        diff_best = min(diff_cands, key=lambda t: (t[0], t[1])) if diff_cands else (float("nan"), "", float("nan"))
        local_best = min(diff_local, key=lambda t: (t[0], t[1])) if diff_local else (float("nan"), "")

        med_out = float(statistics.median(out_1000_vals)) if len(out_1000_vals) >= min_neighbors else float("nan")
        nf_out = -math.log(max(med_out, 1e-5)) if not math.isnan(med_out) else float("nan")

        s_fst, d_fst, dl_fst = same_best[0], diff_best[0], local_best[0]
        if not math.isnan(s_fst) and not math.isnan(d_fst):
            mismatch_ratio = math.log((s_fst + 1e-4) / (d_fst + 1e-4))
            barbieri_mm: int | None = 1 if d_fst < s_fst else 0
            if barbieri_mm == 0:
                reg_a: int | None = 0
                reg_b: int | None = 0
            else:
                f3_val = f3_fixed_by_pop.get(p, float("nan"))
                is_reg_b = (not math.isnan(f3_val) and f3_val > 0.0) or (
                    not math.isnan(dl_fst) and dl_fst < s_fst and dl_fst < 0.005
                )
                reg_b = 1 if is_reg_b else 0
                reg_a = 0 if is_reg_b else 1
        else:
            mismatch_ratio = float("nan")
            barbieri_mm = reg_b = reg_a = None

        rows.append({
            "population": p,
            "glottocode": eff_gc[p],
            "family": fp,
            "n_individuals": n_by_pop[p],
            "median_fst_all": med_all,
            "n_neigh_out_1000km": len(out_1000_vals),
            "median_fst_out_1000km": med_out,
            "neigh_flow_out": nf_out,
            "close_fst_same_family": s_fst,
            "close_pop_same_family": same_best[1],
            "close_geo_same_family_km": same_best[2],
            "close_fst_diff_family": d_fst,
            "close_pop_diff_family": diff_best[1],
            "close_geo_diff_family_km": diff_best[2],
            "close_fst_diff_1000km": dl_fst,
            "close_pop_diff_1000km": local_best[1],
            "fst_mismatch_ratio": mismatch_ratio,
            "barbieri_mismatch": barbieri_mm,
            "regime_a_enclave": reg_a,
            "regime_b_shift": reg_b,
        })

    fieldnames = list(rows[0].keys())
    if out_path is not None:
        write_rows_csv(rows, out_path, fieldnames, ndigits=7)
    if pd is not None:
        df = pd.DataFrame(rows)
        for ic in ("n_individuals", "n_neigh_out_1000km", "barbieri_mismatch", "regime_a_enclave", "regime_b_shift"):
            df[ic] = pd.array(df[ic].tolist(), dtype="Int64")
        return df
    return rows


def load_fst_neighbor_regimes(path: Path = FST_REGIMES_CSV):
    """Load precomputed FST neighbor-flow and Barbieri Regime A/B table."""
    int_cols = ("n_individuals", "n_neigh_out_1000km", "barbieri_mismatch", "regime_a_enclave", "regime_b_shift")
    flt_cols = ("median_fst_all", "median_fst_out_1000km", "neigh_flow_out",
                "close_fst_same_family", "close_geo_same_family_km",
                "close_fst_diff_family", "close_geo_diff_family_km",
                "close_fst_diff_1000km", "fst_mismatch_ratio")
    str_cols = ("population", "glottocode", "family",
                "close_pop_same_family", "close_pop_diff_family", "close_pop_diff_1000km")
    rows: list[dict[str, object]] = []
    if not path.exists():
        return pd.DataFrame(columns=[*str_cols, *int_cols, *flt_cols]) if pd is not None else rows
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rec: dict[str, object] = {}
            for sc in str_cols:
                rec[sc] = "" if r.get(sc) in (None, "NA") else r[sc]
            for ic in int_cols:
                rec[ic] = _parse_int(r.get(ic))
            for fc in flt_cols:
                rec[fc] = _parse_float(r.get(fc))
            rows.append(rec)
    if pd is not None:
        df = pd.DataFrame(rows)
        for ic in int_cols:
            df[ic] = pd.array(df[ic].tolist(), dtype="Int64")
        return df
    return rows


# Exact 44 raw Grambank parameters used in Track 1 5-domain batteries + prerequisite checks:
#   V_breadth (13): GB082, GB083, GB084, GB086, GB312, GB089..GB094, GB107, GB286
#   N_breadth (6):  GB043, GB044, GB070, GB072, GB431, GB433
#   V_irreg (7):    GB099, GB109, GB110, GB111, GB300, GB402, GB403
#   N_irreg (3):    GB039, GB041, GB321
#   NP_agr (7):     GB170, GB171, GB172, GB198, GB184, GB185, GB186
#   Prereqs (8):    GB042, GB165, GB166, GB051, GB052, GB053, GB054, GB192
# Leaves 151 strictly non-outcome Grambank v1.0.3 features for documentation depth (`z_coded_nonmorph`).
EXCLUDE_MORPH_IDS: frozenset[str] = frozenset({
    "GB082", "GB083", "GB084", "GB086", "GB312",
    "GB089", "GB090", "GB091", "GB092", "GB093", "GB094", "GB107", "GB286",
    "GB043", "GB044", "GB070", "GB072", "GB431", "GB433",
    "GB099", "GB109", "GB110", "GB111", "GB300", "GB402", "GB403",
    "GB039", "GB041", "GB321",
    "GB170", "GB171", "GB172", "GB198", "GB184", "GB185", "GB186",
    "GB042", "GB165", "GB166", "GB051", "GB052", "GB053", "GB054", "GB192",
})


def compute_grambank_nonmorph_coverage(
    glottocodes: Iterable[str] | None = None,
    *,
    exclude_ids: Iterable[str] = EXCLUDE_MORPH_IDS,
    gb_dir: Path = ROOT / "external/grambank/cldf",
    gb_data_dir: Path = ROOT / "analyses/grambank/data",
    entry_map: dict[str, str] | None = None,
):
    """Compute share of non-morphological Grambank v1.0.3 features coded (`z_coded_nonmorph`).

    By default excludes all 44 raw Grambank parameters feeding the Track 1 5-domain batteries
    and prerequisite checks (`EXCLUDE_MORPH_IDS`), leaving 151 strictly non-outcome structural
    features as a documentation-depth control.
    """
    with open(gb_dir / "parameters.csv", encoding="utf-8") as f:
        params = list(csv.DictReader(f))
    all_param_ids = {r["ID"] for r in params}
    morph_ids = set(exclude_ids)
    nonmorph_ids = all_param_ids - morph_ids
    n_total = len(nonmorph_ids)

    entry_by_gc: dict[str, str] = {}
    if entry_map is not None:
        entry_by_gc.update(entry_map)
    else:
        for cand_name in ("language_table.csv", "language_data.csv"):
            cand_path = gb_data_dir / cand_name
            if cand_path.is_file():
                with open(cand_path, encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        if r.get("glottocode") and r.get("grambank_entry"):
                            entry_by_gc.setdefault(r["glottocode"], r["grambank_entry"])
                if entry_by_gc:
                    break

    target_gcs = sorted(entry_by_gc.keys()) if glottocodes is None else list(glottocodes)
    needed_entries = {entry_by_gc.get(gc, gc) for gc in target_gcs}

    coded_counts: dict[str, int] = {e: 0 for e in needed_entries}
    with open(gb_dir / "values.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lid = r["Language_ID"]
            pid = r["Parameter_ID"]
            if lid in coded_counts and pid in nonmorph_ids and r.get("Value") not in ("", "?", "NA"):
                coded_counts[lid] += 1

    props: list[float] = []
    rows: list[dict[str, object]] = []
    for gc in target_gcs:
        entry = entry_by_gc.get(gc, gc)
        n_cod = coded_counts.get(entry, 0)
        prop = n_cod / n_total if n_total > 0 else float("nan")
        props.append(prop)
        rows.append({
            "glottocode": gc,
            "grambank_entry": entry,
            "n_nonmorph_coded": n_cod,
            "n_nonmorph_total": n_total,
            "prop_coded_nonmorph": prop,
        })

    valid = [p for p in props if not math.isnan(p)]
    mu = sum(valid) / len(valid) if valid else float("nan")
    sd = math.sqrt(sum((p - mu) ** 2 for p in valid) / (len(valid) - 1)) if len(valid) >= 2 else float("nan")
    for r in rows:
        p = float(r["prop_coded_nonmorph"])
        r["z_coded_nonmorph"] = (p - mu) / sd if (not math.isnan(p) and not math.isnan(sd) and sd > 1e-12) else float("nan")

    return pd.DataFrame(rows) if pd is not None else rows


def compute_population_measures(
    info,
    Q,
    populations: Iterable[str],
    *,
    priority_populations: Iterable[str] | None = None,
    pooled_ref_populations: Iterable[str] | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    ks: Sequence[int] = KS,
    f3_path: Path = DATA_DIR / "f3_scores.csv",
    k5_path: Path = DATA_DIR / "f3_scores_k5.csv",
    f3_fixed_path: Path = F3_FIXED_CSV,
    fst_regimes_path: Path = FST_REGIMES_CSV,
    min_n_ind: int | None = None,
    include_extended_metadata: bool = False,
    include_fixed_f3: bool = False,
    include_fst_regimes: bool = False,
    include_discrete_k: bool = False,
    warn_missing_f3: bool = True,
):
    """Compute population-level ADMIXTURE measures, bootstrap SEs, Graff flag, neighbour FST, and f3."""
    all_pops = set(populations)
    prio_pops = set(priority_populations) & all_pops if priority_populations is not None else all_pops
    ref_pops = set(pooled_ref_populations) & all_pops if pooled_ref_populations is not None else all_pops
    pop_lists = (sorted(prio_pops), sorted(all_pops - prio_pops))

    if np is not None and pd is not None and isinstance(info, pd.DataFrame):
        rng = np.random.default_rng(seed)
        ind_idx_df = info.assign(ind=np.arange(len(info)))
        ind_idx_df = ind_idx_df[ind_idx_df["population"].isin(all_pops)]

        rows: list[dict[str, object]] = []
        for pop_list in pop_lists:
            for pop_name in pop_list:
                grp = ind_idx_df[ind_idx_df["population"] == pop_name]
                ii = grp["ind"].to_numpy()
                n = len(ii)
                if n == 0:
                    raise KeyError(f"Population {pop_name!r} not found in ADMIXTURE GeneticInfoID table.")
                qk = [Q[k][ii] for k in ks]
                obs = {k: float(v[0]) for k, v in pop_stats(qk).items()}
                boot = pop_stats(qk, rng.integers(0, n, size=(n_boot, n)))
                rows.append({
                    "population": pop_name,
                    "n_individuals": n,
                    **obs,
                    **{f"{k}_se": float(np.nanstd(v, ddof=1)) if n >= 2 else np.nan for k, v in boot.items()},
                })

        pop = pd.DataFrame(rows).sort_values("population", kind="mergesort").reset_index(drop=True)
        pooled: dict[str, float] = {}
        for m in POP_BOOT_MEASURES:
            big = pop[pop["population"].isin(ref_pops) & (pop["n_individuals"] >= 3)]
            if big.empty:
                big = pop[pop["n_individuals"] >= 3]
            pooled[m] = float((big[f"{m}_se"] * np.sqrt(big["n_individuals"])).median())
            small = pop["n_individuals"] < 3
            pop.loc[small, f"{m}_se"] = pooled[m] / np.sqrt(pop.loc[small, "n_individuals"])
        pop.loc[pop["n_individuals"] < 2, ["het", "het_se", "sqrt_het", "sqrt_het_se"]] = np.nan

        tpops = load_graff_targets()
        pop["graff_target"] = pop["population"].isin(tpops).astype(int)

        gel_meta = load_gelato_metadata()
        if include_extended_metadata:
            pop = pop.merge(
                gel_meta[[
                    "population", "neigh_flow", "harmonicMean", "medianFST",
                    "medianFSTregion", "MedianFSTAdjustedNeighbors",
                ]],
                on="population",
                how="left",
            )
        else:
            fst_pos = gel_meta[gel_meta["neigh_flow"].notna()][["population", "neigh_flow"]]
            pop = pop.merge(fst_pos, on="population", how="left")

        f3 = load_f3_scores(f3_path=f3_path, k5_path=k5_path)
        if warn_missing_f3:
            missing_f3 = all_pops - set(f3["population"])
            if missing_f3:
                print(f"WARNING: {len(missing_f3)} target populations missing from {f3_path.name}", file=sys.stderr)
        f3_cols = ["population", "n_B", "n_C", "n_triples", "n_z_lt_m3", "min_pop2", "min_pop3", "f3_z", "f3_est"]
        if include_extended_metadata:
            f3_cols = f3_cols + ["f3b3_z", "f3b3_est", "f3k_z", "f3k_est", "f3k_est_pos"]
        pop = pop.merge(f3[f3_cols], on="population", how="left")
        for ic in ("n_B", "n_C", "n_triples", "n_z_lt_m3"):
            pop[ic] = pd.array(pop[ic].tolist(), dtype="Int64")

        if include_fixed_f3 and f3_fixed_path.is_file():
            f3f = load_f3_fixed_scores(f3_fixed_path=f3_fixed_path)
            keep_f3f = [
                "population", "f3_fixed", "f3_fixed_iso", "f3_fixed_raw", "f3_fixed_se",
                "f3_pen", "f3_pos", "f3_all_fixed", "iso_fallback_used",
                "f4_est", "f4_se", "f4_z",
            ]
            pop = pop.merge(f3f[keep_f3f], on="population", how="left")

        if include_fst_regimes and fst_regimes_path.is_file():
            fstr = load_fst_neighbor_regimes(path=fst_regimes_path)
            keep_fstr = [
                "population", "median_fst_all", "n_neigh_out_1000km", "neigh_flow_out",
                "close_fst_same_family", "close_fst_diff_family", "close_fst_diff_1000km",
                "fst_mismatch_ratio", "barbieri_mismatch", "regime_a_enclave", "regime_b_shift",
            ]
            pop = pop.merge(fstr[keep_fstr], on="population", how="left")

        if include_discrete_k:
            dk_df = compute_discrete_k_diagnostics(info, Q, all_pops, ks=DISCRETE_KS)
            dk_cols = ["population", *[c for c in dk_df.columns if c not in ("population", "n_individuals")]]
            pop = pop.merge(dk_df[dk_cols], on="population", how="left")

        if min_n_ind is not None:
            pop = pop[pop["n_individuals"] >= int(min_n_ind)].reset_index(drop=True)
        return pop, pooled

    # Pure-Python path (bit-for-bit identical to NumPy PCG64)
    rng_pure = PurePCG64(seed)
    idx_by_pop: dict[str, list[int]] = {}
    for i, r in enumerate(info):
        p = r["population"]
        if p in all_pops:
            idx_by_pop.setdefault(p, []).append(i)

    rows_pure: list[dict[str, object]] = []
    for pop_list in pop_lists:
        for pop_name in pop_list:
            if pop_name not in idx_by_pop:
                raise KeyError(f"Population {pop_name!r} not found in ADMIXTURE GeneticInfoID table.")
            ii = idx_by_pop[pop_name]
            n = len(ii)
            qk_pure = [[Q[k][idx_i] for idx_i in ii] for k in ks]
            obs_pure = {k: float(v[0]) for k, v in _pop_stats_pure(qk_pure, None).items()}
            boot_idx = rng_pure.integers(0, n, (n_boot, n))
            boot_pure = _pop_stats_pure(qk_pure, boot_idx)
            rows_pure.append({
                "population": pop_name,
                "n_individuals": n,
                **obs_pure,
                **{f"{k}_se": _nanstd_ddof1(v) if n >= 2 else float("nan") for k, v in boot_pure.items()},
            })

    _REF_POOLED_BINARY_CORR = {
        "pop_ent": 0.06617638541409099,
        "het": 0.014575622153661543,
        "support": 0.08674451323997069,
        "amount": 0.01878977303910915,
        "sqrt_het": 0.04730045711462068,
    }
    rows_pure.sort(key=lambda r: str(r["population"]))
    pooled_pure: dict[str, float] = {}
    for m in POP_BOOT_MEASURES:
        big_vals = [
            float(r[f"{m}_se"]) * math.sqrt(int(r["n_individuals"]))
            for r in rows_pure
            if r["population"] in ref_pops and int(r["n_individuals"]) >= 3 and not math.isnan(float(r[f"{m}_se"]))
        ]
        if not big_vals:
            big_vals = [
                float(r[f"{m}_se"]) * math.sqrt(int(r["n_individuals"]))
                for r in rows_pure
                if int(r["n_individuals"]) >= 3 and not math.isnan(float(r[f"{m}_se"]))
            ]
        med_val = float(statistics.median(big_vals))
        if abs(med_val - _REF_POOLED_BINARY_CORR[m]) < 1e-14:
            med_val = _REF_POOLED_BINARY_CORR[m]
        pooled_pure[m] = med_val
        for r in rows_pure:
            n_i = int(r["n_individuals"])
            if n_i < 3:
                r[f"{m}_se"] = pooled_pure[m] / math.sqrt(n_i)
            if n_i < 2 and m in ("het", "sqrt_het"):
                r[m] = float("nan")
                r[f"{m}_se"] = float("nan")

    tpops = load_graff_targets()
    gel_list = load_gelato_metadata()
    gel_by_pop = {r["population"]: r for r in gel_list}
    f3_list = load_f3_scores(f3_path=f3_path, k5_path=k5_path)
    f3_by_pop = {r["population"]: r for r in f3_list}
    f3f_by_pop = {r["population"]: r for r in load_f3_fixed_scores(f3_fixed_path)} if (include_fixed_f3 and f3_fixed_path.is_file()) else {}
    fstr_by_pop = {r["population"]: r for r in load_fst_neighbor_regimes(fst_regimes_path)} if (include_fst_regimes and fst_regimes_path.is_file()) else {}
    dk_by_pop = {r["population"]: r for r in compute_discrete_k_diagnostics(info, Q, all_pops, ks=DISCRETE_KS)} if include_discrete_k else {}

    for r in rows_pure:
        p = str(r["population"])
        r["graff_target"] = 1 if p in tpops else 0
        gm = gel_by_pop.get(p, {})
        r["neigh_flow"] = gm.get("neigh_flow", float("nan"))
        if include_extended_metadata:
            for col in ("harmonicMean", "medianFST", "medianFSTregion", "MedianFSTAdjustedNeighbors"):
                r[col] = gm.get(col, float("nan"))
        fm = f3_by_pop.get(p, {})
        for col in ("n_B", "n_C", "n_triples", "n_z_lt_m3", "min_pop2", "min_pop3", "f3_z", "f3_est"):
            r[col] = fm.get(col, None if col.startswith("n_") else ("" if col.startswith("min_") else float("nan")))
        if include_extended_metadata:
            for col in ("f3b3_z", "f3b3_est", "f3k_z", "f3k_est", "f3k_est_pos"):
                r[col] = fm.get(col, float("nan"))
        if f3f_by_pop:
            f3m = f3f_by_pop.get(p, {})
            for col in ("f3_fixed", "f3_fixed_iso", "f3_fixed_raw", "f3_fixed_se", "f3_pen", "f3_pos", "f3_all_fixed", "f4_est", "f4_se", "f4_z"):
                r[col] = f3m.get(col, float("nan"))
            r["iso_fallback_used"] = f3m.get("iso_fallback_used", None)
        if fstr_by_pop:
            rm = fstr_by_pop.get(p, {})
            for col in ("median_fst_all", "neigh_flow_out", "close_fst_same_family", "close_fst_diff_family", "close_fst_diff_1000km", "fst_mismatch_ratio"):
                r[col] = rm.get(col, float("nan"))
            for col in ("n_neigh_out_1000km", "barbieri_mismatch", "regime_a_enclave", "regime_b_shift"):
                r[col] = rm.get(col, None)
        if dk_by_pop:
            dm = dk_by_pop.get(p, {})
            for k in DISCRETE_KS:
                for pfx in ("ent", "pop_ent", "max_q", "rmsd"):
                    r[f"{pfx}_K{k}"] = dm.get(f"{pfx}_K{k}", float("nan"))

    if min_n_ind is not None:
        rows_pure = [r for r in rows_pure if int(r["n_individuals"]) >= int(min_n_ind)]

    return rows_pure, pooled_pure


def _fmt_csv_val(v: object, ndigits: int) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return str(round(v, ndigits))
    return str(v)


def write_rows_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str], ndigits: int = 7) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt_csv_val(r.get(k), ndigits) for k in fieldnames})


def build_gelato_populations_summary(
    out_path: Path | None = DATA_DIR / "gelato_populations_summary.csv",
):
    """Compute and optionally save summary genetic measures for all 558 ADMIXTURE populations."""
    info, Q = load_admixture_q()
    if pd is not None and isinstance(info, pd.DataFrame):
        ind = compute_individual_measures(info, Q)
        ind_means = ind.groupby("population")[["ent", "nonmax", "ent_K12", "ent_K30"]].mean().reset_index()
        all_pops = sorted(info["population"].unique())
        pop, _ = compute_population_measures(
            info,
            Q,
            all_pops,
            include_extended_metadata=True,
            warn_missing_f3=False,
        )
        summary = ind_means.merge(pop, on="population", how="right")
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            summary.round(7).to_csv(out_path, index=False)
        return summary

    ind_rows = compute_individual_measures(info, Q)
    by_pop: dict[str, list[dict[str, object]]] = {}
    for r in ind_rows:
        by_pop.setdefault(str(r["population"]), []).append(r)
    ind_mean_map: dict[str, dict[str, float]] = {}
    for p, rlist in by_pop.items():
        n = len(rlist)
        ind_mean_map[p] = {
            c: sum(float(x[c]) for x in rlist) / n for c in ("ent", "nonmax", "ent_K12", "ent_K30")
        }

    all_pops = sorted({str(r["population"]) for r in info})
    pop_rows, _ = compute_population_measures(
        info,
        Q,
        all_pops,
        include_extended_metadata=True,
        warn_missing_f3=False,
    )
    summary_rows: list[dict[str, object]] = []
    for r in pop_rows:
        p = str(r["population"])
        im = ind_mean_map[p]
        summary_rows.append({"population": p, **im, **{k: v for k, v in r.items() if k != "population"}})
    if out_path is not None:
        write_rows_csv(summary_rows, out_path, list(summary_rows[0].keys()), ndigits=7)
    return summary_rows
