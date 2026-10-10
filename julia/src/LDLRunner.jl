"""
LDLRunner: end-state Linear Discriminative Learning (JudiLing 1.0.1) for paradigm cell
filling (PCFP) with known lexemes, and the LDL selector's candidate scoring.

See docs/LDL_PROTOCOL.md for the protocol and the leakage audit.

Information flow (enforced by function signatures):
  fit_background(train rows, cfg)          -> Background (shown forms of training verbs only)
  predict_known(bg, queries)               -> predictions for (lemma_id, target_cell) of verbs
                                              that are in the training sample; no gold argument
  add_row(bg, lemma_id, cell, segments)    -> exact rank-one extension of a fitted background
                                              by one (form, simulated meaning) row
  score_candidate(bg, lemma_id, cit_cell, cit_segments, shown_cells)
                                           -> selector: decode a pool candidate's pre-drawn
                                              shown cells after adding its citation row
  score_mapping(bg, queries, gold)         -> gold diagnostics, run only after predictions

The pilot_v1 source-binding (`wug_refit`) protocol was removed on 2026-10-08; it is
preserved at commit 24390cf.
"""
module LDLRunner

using JudiLing
using CSV, DataFrames, JSON, SHA
using LinearAlgebra, SparseArrays, Statistics

export LDLConfig, Background, fit_background, predict_known, add_row, score_candidate,
       target_semantics, gaussian_vector, seed_of, form_semantics, read_queries, read_training,
       read_candidates, run_job, score_job, score_round, full_refit_with_row, row_chat

const SEP = " "                    # segments are space-separated (CONTRACT §2)
const QUERY_COLUMNS = ["lemma_id", "target_cell"]
const CANDIDATE_COLUMNS = ["lemma_id", "citation_cell", "citation_segments", "shown_cells"]
const PREDICTION_COLUMNS = ["lemma_id", "target_cell", "prediction", "prediction_segments",
    "status", "n_candidates", "top_candidates", "support", "unseen_target_features",
    "n_train_forms_lemma", "max_t"]
const RUNNER_VERSION = "ldl-runner-3-pcfp"
const CELL_LIST_SEP = "|"

# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------

Base.@kwdef struct LDLConfig
    grams::Int = 2
    boundary::String = "#"
    sem_dim::Int = 1000
    sem_sd_lexeme::Float64 = 4.0
    sem_sd_inflection::Float64 = 0.4
    sem_sd_noise::Float64 = 1.0
    sem_sd_cell::Float64 = 0.0           # cell-specific vector V(cell); 0 = additive features only
    semantic_seed::Int = 0
    ridge_shift::Float64 = 0.02          # JudiLing make_transform_fac default (:additive)
    threshold::Float64 = 0.05
    max_can::Int = 10
    max_t_margin::Int = 4
    adjacency::Symbol = :full            # :full (all overlapping n-gram pairs) | :attested
    is_tolerant::Bool = false            # learn_paths tolerant mode: up to max_tolerance n-grams per
    tolerance::Float64 = -1000.0         #   path may have support in (tolerance, threshold]
    max_tolerance::Int = 1
    predict_chunk::Int = 400             # items per learn_paths call (results do not depend on it)
    train_diagnostics::Bool = true       # seen-item comprehension/production accuracy
end

const ALLOWED = Dict(:adjacency => (:full, :attested))

"""Build an LDLConfig from a (JSON/YAML-derived) Dict, e.g. the resolved `ldl:` section
plus `semantic_seed`. Unsupported or removed options fail loudly instead of being ignored."""
function LDLConfig(d::AbstractDict)
    g(k, default) = haskey(d, k) && d[k] !== nothing ? d[k] : default
    ridge = Float64(g("ridge_shift", 0.02))
    ridge > 0 || error("ridge_shift must be > 0")
    Bool(g("sem_isdeep", false)) && error("sem_isdeep=true is not implemented (LDL_PROTOCOL §3.2)")
    String(g("decoder", "learn_paths")) == "learn_paths" ||
        error("only decoder=learn_paths is supported (build_paths: looping paths and >100 s per item with bigram cues)")
    haskey(d, "source_binding") && error("source_binding was removed with the pilot_v1 task (PCFP uses known lexemes)")
    haskey(d, "semantic_seed") || error("config lacks semantic_seed")
    c = LDLConfig(
        grams = Int(g("cue_ngram", 2)),
        boundary = String(g("boundary", "#")),
        sem_dim = Int(g("sem_dim", 1000)),
        sem_sd_lexeme = Float64(g("sem_sd_lexeme", 4.0)),
        sem_sd_inflection = Float64(g("sem_sd_inflection", 0.4)),
        sem_sd_noise = Float64(g("sem_sd_noise", 1.0)),
        sem_sd_cell = Float64(g("sem_sd_cell", 0.0)),
        semantic_seed = Int(d["semantic_seed"]),
        ridge_shift = ridge,
        threshold = Float64(g("threshold", 0.05)),
        max_can = Int(g("max_can", 10)),
        max_t_margin = Int(g("max_t_margin", 4)),
        adjacency = Symbol(g("adjacency", "full")),
        is_tolerant = Bool(g("tolerance", false)),
        tolerance = Float64(g("tolerance_floor", -1000.0)),
        max_tolerance = Int(g("max_tolerance", 1)),
        predict_chunk = Int(g("predict_chunk", 400)),
        train_diagnostics = Bool(g("train_diagnostics", true)),
    )
    for (k, ok) in ALLOWED
        getfield(c, k) in ok || error("unsupported $k = $(getfield(c, k)); allowed $(ok)")
    end
    c.grams >= 2 || error("cue_ngram must be >= 2")
    c.predict_chunk >= 1 || error("predict_chunk must be >= 1")
    c.max_tolerance >= 0 || error("max_tolerance must be >= 0")
    c.tolerance < c.threshold || error("tolerance_floor must be below threshold")
    c.sem_sd_cell >= 0 || error("sem_sd_cell must be >= 0")
    c
end

# ----------------------------------------------------------------------------------------
# Deterministic, identifier-keyed Gaussian vectors (version-independent; mirrored in Python)
# ----------------------------------------------------------------------------------------

"""First 8 bytes (big-endian) of sha256(join(parts, "|")) as UInt64."""
function seed_of(parts...)
    d = sha256(join(string.(parts), "|"))
    s = UInt64(0)
    for b in d[1:8]
        s = (s << 8) | UInt64(b)
    end
    s
end

mutable struct SplitMix64
    s::UInt64
end
@inline function next!(r::SplitMix64)
    r.s += 0x9e3779b97f4a7c15
    z = r.s
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end
@inline uniform01(r::SplitMix64) = Float64((next!(r) >> 11) + 1) * 2.0^-53   # in (0, 1]

"""N(0, sd^2)^n via Box-Muller on a SplitMix64 stream seeded by `seed`."""
function gaussian_vector(seed::UInt64, n::Integer, sd::Real)
    r = SplitMix64(seed)
    v = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        u1 = uniform01(r); u2 = uniform01(r)
        rad = sqrt(-2.0 * log(u1)); th = 2.0 * pi * u2
        v[i] = rad * cos(th); i += 1
        if i <= n
            v[i] = rad * sin(th); i += 1
        end
    end
    v .* sd
end

cell_features(cell::AbstractString) = String.(split(cell, ";"))

lexeme_vec(c::LDLConfig, lemma) =
    gaussian_vector(seed_of(c.semantic_seed, "lexeme", lemma), c.sem_dim, c.sem_sd_lexeme)
feature_vec(c::LDLConfig, f) =
    gaussian_vector(seed_of(c.semantic_seed, "feature", f), c.sem_dim, c.sem_sd_inflection)
noise_vec(c::LDLConfig, lemma, cell) =
    gaussian_vector(seed_of(c.semantic_seed, "noise", lemma, cell), c.sem_dim, c.sem_sd_noise)

cell_vec(c::LDLConfig, cell) =
    gaussian_vector(seed_of(c.semantic_seed, "cell", cell), c.sem_dim, c.sem_sd_cell)

"""Inflectional meaning of a cell: its feature vectors, plus the cell's own vector when
`sem_sd_cell > 0` (meaning specific to the feature combination, not shared with other cells)."""
function feature_sum(c::LDLConfig, cell)
    v = sum(feature_vec(c, f) for f in cell_features(cell))
    c.sem_sd_cell > 0 ? v .+ cell_vec(c, cell) : v
end

"""Simulated meaning of an observed (lemma, cell) form: lexeme + inflection + noise."""
function form_semantics(c::LDLConfig, lemma, cell)
    s = lexeme_vec(c, lemma) .+ feature_sum(c, cell)
    c.sem_sd_noise > 0 && (s .+= noise_vec(c, lemma, cell))
    s
end

"""Target meaning of an unseen cell of a known lexeme: lexeme + inflection (no noise)."""
target_semantics(c::LDLConfig, lemma, cell) = lexeme_vec(c, lemma) .+ feature_sum(c, cell)

# ----------------------------------------------------------------------------------------
# Cues and adjacency
# ----------------------------------------------------------------------------------------

tokens_of(segs::AbstractString) = String.(split(strip(segs), SEP))
cue_ngrams(c::LDLConfig, segs::AbstractString) =
    String.(JudiLing.make_ngrams(tokens_of(segs), c.grams, true, SEP, c.boundary))

ngram_prefix(ng) = join(split(ng, SEP)[1:end-1], SEP)
ngram_suffix(ng) = join(split(ng, SEP)[2:end], SEP)

"""Full overlap adjacency (same relation as JudiLing.make_full_adjacency_matrix)."""
function full_adjacency(i2f::Dict{Int,String}, k::Int)
    by_prefix = Dict{String,Vector{Int}}()
    for i in 1:k
        push!(get!(by_prefix, ngram_prefix(i2f[i]), Int[]), i)
    end
    I = Int[]; J = Int[]
    for i in 1:k
        for j in get(by_prefix, ngram_suffix(i2f[i]), Int[])
            push!(I, i); push!(J, j)
        end
    end
    sparse(I, J, ones(Int, length(I)), k, k)
end

"""Attested transitions only (what JudiLing.make_cue_matrix returns as `A`)."""
function attested_adjacency(paths::Vector{Vector{Int}}, k::Int)
    I = Int[]; J = Int[]
    for p in paths, t in 2:length(p)
        push!(I, p[t-1]); push!(J, p[t])
    end
    A = sparse(I, J, ones(Int, length(I)), k, k)
    A.nzval .= 1
    A
end

# ----------------------------------------------------------------------------------------
# Background (training-sample) model
# ----------------------------------------------------------------------------------------

struct Background
    cfg::LDLConfig
    train::DataFrame               # lemma_id, cell_norm, segments (shown forms only)
    paths::Vector{Vector{Int}}     # cue-index path per training row
    f2i::Dict{String,Int}
    i2f::Dict{Int,String}
    C::SparseMatrixCSC{Float64,Int}
    S::Matrix{Float64}
    F::Matrix{Float64}             # comprehension C -> S   (k x d)
    G::Matrix{Float64}             # production   S -> C   (d x k)
    facS::Cholesky{Float64,Matrix{Float64}}   # S'S + shift I
    facC                           # CHOLMOD factor of C'C + shift I
    A::SparseMatrixCSC{Int,Int}
    max_len::Int                   # longest training form, in segments
    features_seen::Set{String}
    forms_per_lemma::Dict{String,Int}
    fit_seconds::Float64
end

"""Fit comprehension and production on the training sample only (end-state, type-based)."""
function fit_background(train::DataFrame, c::LDLConfig)
    t0 = time()
    for col in ("lemma_id", "cell_norm", "segments")
        col in names(train) || error("training table lacks column $col")
    end
    tr = DataFrame(lemma_id = String.(train.lemma_id), cell_norm = String.(train.cell_norm),
                   segments = String.(train.segments))
    nrow(tr) > 0 || error("empty training table")
    any(occursin(c.boundary, s) for s in tr.segments) &&
        error("boundary symbol $(c.boundary) occurs in training segments")
    ngs = [cue_ngrams(c, s) for s in tr.segments]
    f2i = Dict{String,Int}(); i2f = Dict{Int,String}()
    for v in ngs, x in v
        if !haskey(f2i, x)
            f2i[x] = length(f2i) + 1; i2f[f2i[x]] = x
        end
    end
    k = length(f2i); n = nrow(tr)
    paths = [[f2i[x] for x in v] for v in ngs]
    I = Int[]; J = Int[]
    for (i, p) in enumerate(paths), j in unique(p)
        push!(I, i); push!(J, j)
    end
    C = sparse(I, J, ones(Float64, length(I)), n, k)
    S = Matrix{Float64}(undef, n, c.sem_dim)
    for i in 1:n
        S[i, :] = form_semantics(c, tr.lemma_id[i], tr.cell_norm[i])
    end
    facC = JudiLing.make_transform_fac(C; method = :additive, shift = c.ridge_shift)
    F = Matrix(JudiLing.make_transform_matrix(facC, C, S; output_format = :dense))
    facS = JudiLing.make_transform_fac(S; method = :additive, shift = c.ridge_shift)
    G = Matrix(JudiLing.make_transform_matrix(facS, S, Matrix(C); output_format = :dense))
    A = c.adjacency == :full ? full_adjacency(i2f, k) : attested_adjacency(paths, k)
    max_len = maximum(length(tokens_of(s)) for s in tr.segments)
    feats = Set{String}(f for cell in unique(tr.cell_norm) for f in cell_features(cell))
    fpl = Dict{String,Int}()
    for l in tr.lemma_id
        fpl[l] = get(fpl, l, 0) + 1
    end
    Background(c, tr, paths, f2i, i2f, C, S, F, G, facS, facC, A, max_len, feats, fpl, time() - t0)
end

"""Translate a cue path into segments (boundary removed)."""
function path_segments(path::Vector{Int}, i2f::Dict{Int,String}, bnd::String)
    toks = [split(i2f[i], SEP)[1] for i in path]
    append!(toks, split(i2f[path[end]], SEP)[2:end])
    String.(filter(!=(bnd), toks))
end
segments_to_form(segs::Vector{String}) = replace(join(segs, ""), "_" => " ")

"""Run learn_paths for target semantics `S_tgt` (T x d) with predicted cues `Chat` on the
given decoder training state. Only `size(data_val, 1)` of data_val is read (no forms).
learn_paths maps Ĉ to the n-gram at each position t (positional mappings trained on the
training forms), chains n-grams with support above `threshold` (in tolerant mode, also up
to `max_tolerance` n-grams per path with support in (`tolerance`, `threshold`]), and ranks
complete paths by synthesis-by-analysis (cor(c F, s)), keeping `max_can`."""
function decode(c::LDLConfig, data_train::DataFrame, C_train, S_tgt::Matrix{Float64},
                F::Matrix{Float64}, Chat::Matrix{Float64}, A, i2f, f2i, max_t::Int)
    T = size(S_tgt, 1)
    JudiLing.learn_paths(
        data_train, DataFrame(query_index = 1:T),
        C_train, S_tgt, F, Chat, A, i2f, f2i;
        check_gold_path = false, gold_ind = nothing, Shat_val = nothing,
        max_t = max_t, max_can = c.max_can, threshold = c.threshold,
        is_tolerant = c.is_tolerant, tolerance = c.tolerance, max_tolerance = c.max_tolerance,
        grams = c.grams, tokenized = true, sep_token = SEP,
        keep_sep = true, target_col = :segments, start_end_token = c.boundary,
        issparse = :auto, verbose = false)
end

# ----------------------------------------------------------------------------------------
# Known-lexeme prediction (the evaluated LDL)
# ----------------------------------------------------------------------------------------

error_row(lid, cell, max_t) = (lemma_id = String(lid), target_cell = String(cell), prediction = "",
    prediction_segments = "", status = "error", n_candidates = 0, top_candidates = "[]",
    support = NaN, unseen_target_features = "", n_train_forms_lemma = -1, max_t = max_t)

"""Predict the queried cells of verbs that are in the training sample.

Target meaning = lexeme vector + the cell's feature vectors; Ĉ = Ŝ G. Decoding uses the
training forms only: cue inventory, adjacency, decoder training rows and max_t
(= longest training form + margin) never see a gold form. Items are decoded in chunks;
learn_paths treats rows independently, so results do not depend on chunking or order."""
function predict_known(bg::Background, q::DataFrame)
    c = bg.cfg
    lids = String.(q.lemma_id); cells = String.(q.target_cell)
    unknown = unique(filter(l -> !haskey(bg.forms_per_lemma, l), lids))
    isempty(unknown) || error("query lemmas without training forms (not known lexemes): $(first(unknown, 3))")
    max_t = bg.max_len + c.max_t_margin
    rows = Vector{NamedTuple}(undef, length(lids))
    for start in 1:c.predict_chunk:length(lids)
        idx = start:min(start + c.predict_chunk - 1, length(lids))
        S_tgt = Matrix{Float64}(undef, length(idx), c.sem_dim)
        for (t, i) in enumerate(idx)
            S_tgt[t, :] = target_semantics(c, lids[i], cells[i])
        end
        Chat = S_tgt * bg.G
        res = try
            decode(c, bg.train[:, [:segments]], bg.C, S_tgt, bg.F, Chat, bg.A, bg.i2f, bg.f2i, max_t)
        catch err
            @warn "learn_paths failed for a chunk" exception = (err, catch_backtrace())
            nothing
        end
        for (t, i) in enumerate(idx)
            if res === nothing
                rows[i] = error_row(lids[i], cells[i], max_t)
                continue
            end
            cands = res[t]
            preds = [path_segments(r.ngrams_ind, bg.i2f, c.boundary) for r in cands]
            top = [Dict("prediction" => segments_to_form(p), "support" => round(r.support, digits = 6))
                   for (p, r) in zip(preds, cands)]
            unseen = filter(f -> !(f in bg.features_seen), cell_features(cells[i]))
            rows[i] = (lemma_id = lids[i], target_cell = cells[i],
                prediction = isempty(preds) ? "" : segments_to_form(preds[1]),
                prediction_segments = isempty(preds) ? "" : join(preds[1], SEP),
                status = isempty(cands) ? "no_candidate" : "ok", n_candidates = length(cands),
                top_candidates = JSON.json(top), support = isempty(cands) ? NaN : cands[1].support,
                unseen_target_features = join(unseen, ";"),
                n_train_forms_lemma = bg.forms_per_lemma[lids[i]], max_t = max_t)
        end
    end
    df = isempty(rows) ? DataFrame([k => [] for k in PREDICTION_COLUMNS]) : DataFrame(rows)
    df[:, PREDICTION_COLUMNS]
end

# ----------------------------------------------------------------------------------------
# Exact rank-one extension by one row (selector citation row)
# ----------------------------------------------------------------------------------------

"""A fitted background extended by one (form, meaning) row; the background is not mutated.

Production: G_h = argmin ‖[S; s]G − [C 0; c]‖² + λ‖G‖² via Sherman–Morrison on
P = (SᵀS + λI)⁻¹; only Ĉ for requested targets is materialised (`row_chat`).
Comprehension: F_h = argmin ‖[C 0; c]F − [S; s]‖² + λ‖F‖², exact rank-one update of
[F; 0] (novel cues have no background weights)."""
struct RowState
    f2i::Dict{String,Int}
    i2f::Dict{Int,String}
    novel::Vector{String}
    c_row::Vector{Float64}         # extended cue vector of the added form
    s_row::Vector{Float64}         # its simulated meaning
    Ps::Vector{Float64}            # (SᵀS + λI)⁻¹ s
    alpha::Float64                 # s P sᵀ
    resid::Vector{Float64}         # c − s G_ext
    F_h::Matrix{Float64}
    C_h::SparseMatrixCSC{Float64,Int}
    data_h::DataFrame
    paths_h::Vector{Vector{Int}}
    A_h::SparseMatrixCSC{Int,Int}
    max_t::Int
end

function add_row(bg::Background, lemma_id::AbstractString, cell::AbstractString, segs::AbstractString)
    c = bg.cfg
    occursin(c.boundary, segs) && error("boundary symbol in added form")
    ng = cue_ngrams(c, segs)
    k = length(bg.f2i); d = c.sem_dim
    novel = String[]
    for x in ng
        (!haskey(bg.f2i, x) && !(x in novel)) && push!(novel, x)
    end
    kx = k + length(novel)
    f2i = copy(bg.f2i); i2f = copy(bg.i2f)
    for (j, x) in enumerate(novel)
        f2i[x] = k + j; i2f[k + j] = x
    end
    crow = zeros(kx)
    for x in ng
        crow[f2i[x]] = 1.0
    end
    ck = crow[1:k]
    s = form_semantics(c, lemma_id, cell)
    # production (rank-one on G_ext = [G 0])
    Ps = bg.facS \ s
    alpha = dot(s, Ps)
    resid = crow .- vcat(vec(s' * bg.G), zeros(length(novel)))
    # comprehension (rank-one on F_ext = [F; 0])
    F_h = vcat(bg.F, zeros(length(novel), d))
    r = s .- vec(crow' * F_h)
    Pc = vcat(bg.facC \ ck, crow[k+1:end] ./ c.ridge_shift)
    beta = dot(crow, Pc)
    F_h = F_h .+ (Pc ./ (1 + beta)) * r'
    data_h = vcat(bg.train[:, [:segments]], DataFrame(segments = [String(segs)]))
    C_h = vcat(hcat(bg.C, spzeros(nrow(bg.train), length(novel))), sparse(crow'))
    paths_h = vcat(bg.paths, [[f2i[x] for x in ng]])
    A_h = if c.adjacency == :full
        isempty(novel) ? bg.A : full_adjacency(i2f, kx)
    else
        attested_adjacency(paths_h, kx)
    end
    max_t = max(bg.max_len, length(tokens_of(segs))) + c.max_t_margin
    RowState(f2i, i2f, novel, crow, s, Ps, alpha, resid, F_h, C_h, data_h, paths_h, A_h, max_t)
end

"""Ĉ = S_tgt G_h for target meanings, from the rank-one state (exact ridge solution)."""
function row_chat(bg::Background, st::RowState, S_tgt::Matrix{Float64})
    Chat0 = hcat(S_tgt * bg.G, zeros(size(S_tgt, 1), length(st.novel)))
    Chat0 .+ ((S_tgt * st.Ps) ./ (1 + st.alpha)) * st.resid'
end

"""Reference implementation for tests: refit F and G with JudiLing on the augmented
matrices (same inventory order as `add_row`). Returns (Chat, F_full)."""
function full_refit_with_row(bg::Background, lemma_id, cell, segs, S_tgt::Matrix{Float64})
    c = bg.cfg
    st = add_row(bg, lemma_id, cell, segs)
    C2 = st.C_h
    S2 = vcat(bg.S, st.s_row')
    facC2 = JudiLing.make_transform_fac(C2; method = :additive, shift = c.ridge_shift)
    F2 = Matrix(JudiLing.make_transform_matrix(facC2, C2, S2; output_format = :dense))
    facS2 = JudiLing.make_transform_fac(S2; method = :additive, shift = c.ridge_shift)
    G2 = Matrix(JudiLing.make_transform_matrix(facS2, S2, Matrix(C2); output_format = :dense))
    (S_tgt * G2, F2, row_chat(bg, st, S_tgt), st.F_h)
end

# ----------------------------------------------------------------------------------------
# Selector: score one pool candidate from its citation row
# ----------------------------------------------------------------------------------------

"""Decode a candidate's pre-drawn shown cells after adding its citation row.

Inputs are only the candidate's lemma id (for its simulated lexeme vector), its citation
cell and the segments of its citation label, and the *names* of its shown cells. No other
form of the candidate exists here. Returns per-cell rows and one comprehension-check row
(how far c·F of the citation cues is from the simulated citation meaning, and the share of
citation cues unseen in training), computed on the round's background."""
function score_candidate(bg::Background, lemma_id::AbstractString, cit_cell::AbstractString,
                         cit_segs::AbstractString, shown::Vector{String})
    c = bg.cfg
    st = add_row(bg, lemma_id, cit_cell, cit_segs)
    T = length(shown)
    S_tgt = Matrix{Float64}(undef, T, c.sem_dim)
    for (t, cell) in enumerate(shown)
        S_tgt[t, :] = target_semantics(c, lemma_id, cell)
    end
    Chat = row_chat(bg, st, S_tgt)
    status = fill("ok", T)
    res = try
        decode(c, st.data_h, st.C_h, S_tgt, st.F_h, Chat, st.A_h, st.i2f, st.f2i, st.max_t)
    catch err
        @warn "learn_paths failed" lemma_id exception = (err, catch_backtrace())
        status .= "error"
        [JudiLing.Result_Path_Info_Struct[] for _ in 1:T]
    end
    cit_toks = tokens_of(cit_segs)
    rows = NamedTuple[]
    for t in 1:T
        cands = res[t]
        status[t] == "ok" && isempty(cands) && (status[t] = "no_candidate")
        preds = [path_segments(r.ngrams_ind, st.i2f, c.boundary) for r in cands]
        sup = [r.support for r in cands]
        push!(rows, (lemma_id = String(lemma_id), target_cell = shown[t], status = status[t],
            n_candidates = length(cands), supports = JSON.json(round.(sup, digits = 8)),
            top_support = isempty(sup) ? NaN : sup[1],
            top_prediction_segments = isempty(preds) ? "" : join(preds[1], SEP),
            top_equals_citation = !isempty(preds) && preds[1] == cit_toks))
    end
    # comprehension-side check (not used for selection)
    k = length(bg.f2i)
    s_hat = vec(st.c_row[1:k]' * bg.F)
    s_true = st.s_row
    n_cues = length(unique(cue_ngrams(c, cit_segs)))
    comp = (lemma_id = String(lemma_id), comp_cor = cor(s_hat, s_true),
            comp_rel_dist = norm(s_hat .- s_true) / norm(s_true),
            n_citation_cues = n_cues, n_citation_cues_unseen = length(st.novel),
            share_citation_cues_unseen = length(st.novel) / n_cues,
            citation_len = length(cit_toks))
    rows, comp
end

"""Score every candidate of one acquisition round on one fitted background. Each
candidate starts from the same immutable background, so scores do not depend on the
order in which candidates are scored."""
function score_round(bg::Background, cands::DataFrame)
    cell_rows = NamedTuple[]; comp_rows = NamedTuple[]
    for r in eachrow(cands)
        shown = String.(split(r.shown_cells, CELL_LIST_SEP))
        cr, comp = try
            score_candidate(bg, r.lemma_id, r.citation_cell, r.citation_segments, shown)
        catch err
            @warn "candidate failed" r.lemma_id exception = (err, catch_backtrace())
            ([(lemma_id = String(r.lemma_id), target_cell = cell, status = "error", n_candidates = 0,
               supports = "[]", top_support = NaN, top_prediction_segments = "",
               top_equals_citation = false) for cell in shown],
             (lemma_id = String(r.lemma_id), comp_cor = NaN, comp_rel_dist = NaN, n_citation_cues = -1,
              n_citation_cues_unseen = -1, share_citation_cues_unseen = NaN, citation_len = -1))
        end
        append!(cell_rows, cr); push!(comp_rows, comp)
    end
    DataFrame(cell_rows), DataFrame(comp_rows)
end

# ----------------------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------------------

"""Read a query table, keeping only (lemma_id, target_cell); other columns are ignored."""
function read_queries(path::AbstractString)
    q = CSV.read(path, DataFrame; types = String, stringtype = String)
    for col in QUERY_COLUMNS
        col in names(q) || error("query table lacks column $col")
    end
    q[:, QUERY_COLUMNS]
end

"""Read a candidate table, keeping only the four permitted columns."""
function read_candidates(path::AbstractString)
    q = CSV.read(path, DataFrame; types = String, stringtype = String)
    for col in CANDIDATE_COLUMNS
        col in names(q) || error("candidate table lacks column $col")
    end
    q[:, CANDIDATE_COLUMNS]
end

"""Read training rows: variant 0 of non-missing cells (CONTRACT §4)."""
function read_training(path::AbstractString)
    t = CSV.read(path, DataFrame; types = String, stringtype = String)
    for col in ("lemma_id", "cell_norm", "segments")
        col in names(t) || error("training table lacks column $col")
    end
    n0 = nrow(t)
    "variant_idx" in names(t) && (t = t[parse.(Int, t.variant_idx) .== 0, :])
    "is_missing" in names(t) && (t = t[lowercase.(coalesce.(t.is_missing, "false")) .!= "true", :])
    t = t[.!ismissing.(t.segments), :]
    t = t[strip.(t.segments) .!= "", :]
    dup = nonunique(t[:, [:lemma_id, :cell_norm]])
    any(dup) && error("training table has duplicate (lemma_id, cell_norm) rows after variant filter")
    t[:, ["lemma_id", "cell_norm", "segments"]], n0
end

# ----------------------------------------------------------------------------------------
# Diagnostics and batch jobs
# ----------------------------------------------------------------------------------------

"""Seen-item diagnostics on the training sample (training forms are not held-out gold)."""
function train_diagnostics(bg::Background)
    c = bg.cfg
    Shat = Matrix(bg.C) * bg.F
    comp = JudiLing.eval_SC(Shat, bg.S, bg.train, :segments)
    res = decode(c, bg.train[:, [:segments]], bg.C, bg.S, bg.F, bg.S * bg.G, bg.A, bg.i2f, bg.f2i,
                 bg.max_len + c.max_t_margin)
    prod = mean(!isempty(r) && join(path_segments(r[1].ngrams_ind, bg.i2f, c.boundary), SEP) ==
                strip(bg.train.segments[i]) for (i, r) in enumerate(res))
    Dict("train_comprehension_accuracy" => comp, "train_production_accuracy" => prod,
         "train_chat_c_cor_mean" => mean(cor((bg.S*bg.G)[i, :], Vector(bg.C[i, :])) for i in 1:nrow(bg.train)))
end

function atomic_write(f::Function, path::AbstractString)
    tmp = path * ".tmp"
    f(tmp)
    mv(tmp, path; force = true)
end

"""Fit one background and predict its known-lexeme queries. `job` keys: train_csv,
queries_csv, out_dir, config (resolved ldl config incl. semantic_seed), job_config
(written last as the completion marker)."""
function run_job(job::AbstractDict)
    t0 = time()
    out = job["out_dir"]; mkpath(out)
    cfg = LDLConfig(job["config"])
    train, n_raw = read_training(job["train_csv"])
    q = read_queries(job["queries_csv"])
    bg = fit_background(train, cfg)
    t1 = time()
    pred = predict_known(bg, q)
    tpred = time() - t1
    atomic_write(tmp -> CSV.write(tmp, pred), joinpath(out, "predictions.csv"))
    td = cfg.train_diagnostics ? train_diagnostics(bg) : Dict{String,Any}()
    probe = Dict(string(bg.train.lemma_id[i], "|", bg.train.cell_norm[i]) => bg.S[i, 1:5]
                 for i in 1:min(3, nrow(bg.train)))
    diag = Dict{String,Any}(
        "runner_version" => RUNNER_VERSION,
        "judiling_version" => string(pkgversion(JudiLing)), "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(), "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "n_train_rows_raw" => n_raw, "n_train_rows" => nrow(bg.train),
        "n_train_lemmas" => length(bg.forms_per_lemma),
        "n_cues_background" => length(bg.f2i), "sem_dim" => cfg.sem_dim, "cue_ngram" => cfg.grams,
        "sem_sd_inflection" => cfg.sem_sd_inflection, "semantic_seed" => cfg.semantic_seed,
        "ridge_shift" => cfg.ridge_shift, "adjacency" => string(cfg.adjacency),
        "decoder" => "learn_paths", "threshold" => cfg.threshold, "tolerant" => cfg.is_tolerant,
        "tolerance_floor" => cfg.tolerance, "max_tolerance" => cfg.max_tolerance,
        "sem_sd_noise" => cfg.sem_sd_noise, "sem_sd_cell" => cfg.sem_sd_cell, "sem_sd_lexeme" => cfg.sem_sd_lexeme, "max_train_len" => bg.max_len,
        "max_t" => bg.max_len + cfg.max_t_margin,
        "n_query_lemmas" => length(unique(q.lemma_id)), "n_query_items" => nrow(pred),
        "status_counts" => Dict(s => count(==(s), pred.status) for s in unique(pred.status)),
        "n_items_with_unseen_target_features" => count(!isempty, pred.unseen_target_features),
        "unseen_target_features" => sort(unique(filter(!isempty, vcat(split.(pred.unseen_target_features, ";")...)))),
        "features_seen" => sort(collect(bg.features_seen)),
        "background_fit_seconds" => bg.fit_seconds, "predict_seconds" => tpred,
        "job_seconds" => time() - t0, "semantic_probe" => probe,
    )
    merge!(diag, td)
    atomic_write(tmp -> open(io -> JSON.print(io, diag, 2), tmp, "w"), joinpath(out, "diagnostics.json"))
    atomic_write(tmp -> open(io -> JSON.print(io, job["job_config"], 2), tmp, "w"), joinpath(out, "job_config.json"))
    out
end

"""Gold-side mapping diagnostics for known-lexeme items. `gold` maps (lemma_id,
target_cell) => gold variant segment strings. Recomputes Ĉ deterministically; never
feeds back into prediction."""
function score_mapping(bg::Background, q::DataFrame, gold::Dict)
    c = bg.cfg
    max_t = bg.max_len + c.max_t_margin
    rows = NamedTuple[]
    for r in eachrow(q)
        s = target_semantics(c, r.lemma_id, r.target_cell)
        chat = vec(s' * bg.G)
        best = nothing
        gv = gold[(r.lemma_id, r.target_cell)]
        for (vi, gsegs) in enumerate(gv)
            gpath = cue_ngrams(c, gsegs)
            gng = unique(gpath)
            cg = zeros(length(bg.f2i))
            n_out = 0
            for x in gng
                haskey(bg.f2i, x) ? (cg[bg.f2i[x]] = 1.0) : (n_out += 1)
            end
            cr = cor(chat, cg)
            n_below = count(x -> haskey(bg.f2i, x) && chat[bg.f2i[x]] <= c.threshold, gng)
            row = (lemma_id = String(r.lemma_id), target_cell = String(r.target_cell),
                   chat_gold_cor = cr, gold_variant_idx = vi - 1, n_gold_variants = length(gv),
                   n_gold_cues = length(gng), n_gold_cues_outside_inventory = n_out,
                   n_gold_cues_below_threshold = n_below,
                   gold_reachable = n_out == 0 && n_below == 0,
                   gold_path_longer_than_max_t = length(gpath) > max_t)
            if best === nothing || (isnan(best.chat_gold_cor) && !isnan(cr)) || cr > best.chat_gold_cor
                best = row
            end
        end
        push!(rows, best)
    end
    rows
end

"""Gold-side mapping quality for a finished job; `job` adds gold_csv (lemma_id,
target_cell, gold_variants = segment strings joined by " || "). Writes mapping_quality.csv."""
function score_job(job::AbstractDict)
    out = job["out_dir"]
    isfile(joinpath(out, "predictions.csv")) || error("predictions.csv missing in $out; score only after prediction")
    cfg = LDLConfig(job["config"])
    train, _ = read_training(job["train_csv"])
    q = read_queries(job["queries_csv"])
    bg = fit_background(train, cfg)
    gold = CSV.read(job["gold_csv"], DataFrame; types = String, stringtype = String)
    gmap = Dict((r.lemma_id, r.target_cell) => String.(strip.(split(r.gold_variants, " || "))) for r in eachrow(gold))
    pred = CSV.read(joinpath(out, "predictions.csv"), DataFrame; types = Dict(:top_candidates => String,
                    :prediction_segments => String, :prediction => String), stringtype = String)
    pmap = Dict((r.lemma_id, r.target_cell) => r for r in eachrow(pred))
    rows = NamedTuple[]
    for r in score_mapping(bg, q, gmap)
        p = pmap[(r.lemma_id, r.target_cell)]
        gforms = [segments_to_form(String.(split(g, SEP))) for g in gmap[(r.lemma_id, r.target_cell)]]
        cands = ismissing(p.top_candidates) ? [] : JSON.parse(p.top_candidates)
        rank = findfirst(x -> x["prediction"] in gforms, cands)
        push!(rows, merge(r, (gold_in_top_candidates = rank !== nothing,
                              gold_rank_in_candidates = rank === nothing ? 0 : rank)))
    end
    atomic_write(tmp -> CSV.write(tmp, DataFrame(rows)), joinpath(out, "mapping_quality.csv"))
    open(io -> JSON.print(io, job["score_config"], 2), joinpath(out, "mapping_quality.json"), "w")
    out
end

end # module
