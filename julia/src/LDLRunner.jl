"""
LDLRunner: end-state Linear Discriminative Learning (JudiLing) for source-known
paradigm completion of held-out lemmas, with a wug-style source-binding protocol.

See docs/LDL_PROTOCOL.md for the protocol and the leakage audit.

Information flow (enforced by function signatures):
  fit_background(train rows, cfg)                -> Background   (training forms only)
  predict_lemma(bg, lemma_id, src_cell, src_segments, target_cells)
                                                 -> predictions  (no gold argument exists)
  score_mapping(bg, query, gold)                 -> gold diagnostics, run only after predictions
"""
module LDLRunner

using JudiLing
using CSV, DataFrames, JSON, SHA
using LinearAlgebra, SparseArrays, Statistics

export LDLConfig, Background, fit_background, predict_lemma, run_queries, score_mapping,
       gaussian_vector, seed_of, form_semantics, read_queries, read_training, run_job, score_job

const SEP = " "                    # segments are space-separated (CONTRACT §2)
const QUERY_COLUMNS = ["lemma_id", "source_cell", "source_form", "source_segments", "target_cell"]
const PREDICTION_COLUMNS = ["lemma_id", "target_cell", "prediction", "prediction_segments",
    "status", "n_candidates", "top_candidates", "support", "n_source_cues",
    "n_source_cues_unseen", "unseen_target_features", "binding_fit", "max_t"]
const RUNNER_VERSION = "ldl-runner-2"

# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------

Base.@kwdef struct LDLConfig
    grams::Int = 3
    boundary::String = "#"
    sem_dim::Int = 1000
    sem_sd_lexeme::Float64 = 4.0
    sem_sd_inflection::Float64 = 0.4
    sem_sd_noise::Float64 = 1.0
    semantic_seed::Int = 0
    ridge_shift::Float64 = 0.02          # JudiLing make_transform_fac default (:additive)
    threshold::Float64 = 0.05
    max_can::Int = 10
    max_t_margin::Int = 4
    source_binding::Symbol = :wug_refit  # :wug_refit | :lexeme_refit | :none
    adjacency::Symbol = :full            # :full (all overlapping n-gram pairs) | :attested
    decoder::Symbol = :learn_paths       # :learn_paths | :build_paths (not recommended)
    n_neighbors::Int = 10                # build_paths only
    train_diagnostics::Bool = true       # seen-item comprehension/production accuracy
end

const ALLOWED = Dict(
    :source_binding => (:wug_refit, :lexeme_refit, :none),
    :adjacency => (:full, :attested),
    :decoder => (:learn_paths, :build_paths),
)

"""Build an LDLConfig from a (JSON/YAML-derived) Dict, e.g. the resolved `ldl:` section
plus `semantic_seed`. Unsupported options fail loudly instead of being ignored."""
function LDLConfig(d::AbstractDict)
    g(k, default) = haskey(d, k) && d[k] !== nothing ? d[k] : default
    if haskey(d, "ridge_shift")
        ridge = Float64(d["ridge_shift"])
    else                                 # legacy key: 0 meant "JudiLing default"
        ridge = Float64(g("ridge_lambda", 0.0)); ridge = ridge > 0 ? ridge : 0.02
    end
    ridge > 0 || error("ridge_shift must be > 0")
    Bool(g("sem_isdeep", false)) && error("sem_isdeep=true is not implemented (LDL_PROTOCOL §3.2)")
    Bool(g("tolerance", false)) && error("tolerance mode is not implemented")
    haskey(d, "semantic_seed") || error("config lacks semantic_seed")
    c = LDLConfig(
        grams = Int(g("cue_ngram", 3)),
        boundary = String(g("boundary", "#")),
        sem_dim = Int(g("sem_dim", 1000)),
        sem_sd_lexeme = Float64(g("sem_sd_lexeme", 4.0)),
        sem_sd_inflection = Float64(g("sem_sd_inflection", 0.4)),
        sem_sd_noise = Float64(g("sem_sd_noise", 1.0)),
        semantic_seed = Int(d["semantic_seed"]),
        ridge_shift = ridge,
        threshold = Float64(g("threshold", 0.05)),
        max_can = Int(g("max_can", 10)),
        max_t_margin = Int(g("max_t_margin", 4)),
        source_binding = Symbol(g("source_binding", "wug_refit")),
        adjacency = Symbol(g("adjacency", "full")),
        decoder = Symbol(g("decoder", "learn_paths")),
        n_neighbors = Int(g("n_neighbors", 10)),
        train_diagnostics = Bool(g("train_diagnostics", true)),
    )
    for (k, ok) in ALLOWED
        getfield(c, k) in ok || error("unsupported $k = $(getfield(c, k)); allowed $(ok)")
    end
    c.grams >= 2 || error("cue_ngram must be >= 2")
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

feature_sum(c::LDLConfig, cell) = sum(feature_vec(c, f) for f in cell_features(cell))

"""Simulated meaning of (lemma, cell): lexeme + sum of inflectional features + noise."""
function form_semantics(c::LDLConfig, lemma, cell)
    s = lexeme_vec(c, lemma) .+ feature_sum(c, cell)
    c.sem_sd_noise > 0 && (s .+= noise_vec(c, lemma, cell))
    s
end

# ----------------------------------------------------------------------------------------
# Cues and adjacency
# ----------------------------------------------------------------------------------------

tokens_of(segs::AbstractString) = String.(split(strip(segs), SEP))
cue_ngrams(c::LDLConfig, segs::AbstractString) =
    String.(JudiLing.make_ngrams(tokens_of(segs), c.grams, true, SEP, c.boundary))

ngram_prefix(ng) = join(split(ng, SEP)[1:end-1], SEP)
ngram_suffix(ng) = join(split(ng, SEP)[2:end], SEP)

"""Full overlap adjacency (paper §3; same relation as JudiLing.make_full_adjacency_matrix)."""
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
    train::DataFrame               # lemma_id, cell_norm, segments (training forms only)
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
    Background(c, tr, paths, f2i, i2f, C, S, F, G, facS, facC, A, max_len, feats, time() - t0)
end

# ----------------------------------------------------------------------------------------
# Per-held-out-lemma state (the background is never mutated)
# ----------------------------------------------------------------------------------------

struct LemmaState
    f2i::Dict{String,Int}
    i2f::Dict{Int,String}
    novel::Vector{String}          # source cues absent from the background inventory
    c_src::Vector{Float64}         # extended cue vector of the source form
    s_src::Vector{Float64}         # source semantics used for the binding row
    S_tgt::Matrix{Float64}         # target semantics (T x d)
    Chat::Matrix{Float64}          # predicted target cue vectors (T x k_ext)
    F_h::Matrix{Float64}           # comprehension used for synthesis-by-analysis
    C_h::SparseMatrixCSC{Float64,Int}
    data_h::DataFrame              # decoder training forms (background [+ source])
    paths_h::Vector{Vector{Int}}   # cue paths of the decoder training forms
    A_h::SparseMatrixCSC{Int,Int}
    max_t::Int
    binding_fit::Float64           # cor(predicted c for the source semantics, c_src)
    unseen_features::Vector{Vector{String}}
end

"""Extend the background with one held-out lemma's source anchor and map to target cues.

Only (lemma_id, source cell, source segments, target cells) are inputs. The production
refit with the binding row (s_src -> c_src) is the exact ridge solution computed by a
rank-one (recursive least squares) update of the background G; see LDL_PROTOCOL.md §4."""
function lemma_state(bg::Background, lemma_id::AbstractString, src_cell::AbstractString,
                     src_segs::AbstractString, tgt_cells::Vector{String})
    c = bg.cfg
    occursin(c.boundary, src_segs) && error("boundary symbol in source segments")
    ng = cue_ngrams(c, src_segs)
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
    csrc = zeros(kx)
    for x in ng
        csrc[f2i[x]] = 1.0
    end
    ck = csrc[1:k]

    # source semantics
    s_src = if c.source_binding == :lexeme_refit
        form_semantics(c, lemma_id, src_cell)
    else                      # :wug_refit and :none -> comprehension estimate (paper §4.3.2)
        vec(ck' * bg.F)
    end
    fsrc = feature_sum(c, src_cell)
    T = length(tgt_cells)
    S_tgt = Matrix{Float64}(undef, T, d)
    for (t, cell) in enumerate(tgt_cells)
        S_tgt[t, :] = s_src .- fsrc .+ feature_sum(c, cell)
    end

    # production: G_ext = [G 0]; exact refit with binding row via RLS update
    Chat0 = hcat(S_tgt * bg.G, zeros(T, length(novel)))
    F_h = vcat(bg.F, zeros(length(novel), d))
    if c.source_binding == :none
        Chat = Chat0
        binding_fit = NaN
        data_h = bg.train[:, [:segments]]
        C_h = hcat(bg.C, spzeros(nrow(bg.train), length(novel)))
        paths_h = bg.paths
    else
        chat_src0 = vcat(vec(s_src' * bg.G), zeros(length(novel)))
        resid = csrc .- chat_src0
        Ps = bg.facS \ s_src
        alpha = dot(s_src, Ps)
        Chat = Chat0 .+ ((S_tgt * Ps) ./ (1 + alpha)) * resid'
        chat_src = chat_src0 .+ (alpha / (1 + alpha)) .* resid
        binding_fit = cor(chat_src, csrc)
        # comprehension: exact refit with binding row (c_src -> s_src); a no-op for
        # :wug_refit because s_src = c_src F already (LDL_PROTOCOL.md §4.3)
        r = s_src .- vec(csrc' * F_h)
        if norm(r) > 1e-9 * max(1.0, norm(s_src))
            Pc = vcat(bg.facC \ ck, csrc[k+1:end] ./ c.ridge_shift)
            beta = dot(csrc, Pc)
            F_h = F_h .+ (Pc ./ (1 + beta)) * r'
        end
        data_h = vcat(bg.train[:, [:segments]], DataFrame(segments = [String(src_segs)]))
        C_h = vcat(hcat(bg.C, spzeros(nrow(bg.train), length(novel))), sparse(csrc'))
        paths_h = vcat(bg.paths, [[f2i[x] for x in ng]])
    end

    A_h = if c.adjacency == :full
        isempty(novel) ? bg.A : full_adjacency(i2f, kx)
    else
        attested_adjacency(paths_h, kx)
    end
    max_t = max(bg.max_len, length(tokens_of(src_segs))) + c.max_t_margin
    unseen = [filter(f -> !(f in bg.features_seen), cell_features(cell)) for cell in tgt_cells]
    LemmaState(f2i, i2f, novel, csrc, s_src, S_tgt, Chat, F_h, C_h, data_h, paths_h, A_h, max_t,
               binding_fit, unseen)
end

"""Translate a cue path into segments (boundary removed)."""
function path_segments(path::Vector{Int}, i2f::Dict{Int,String}, bnd::String)
    toks = [split(i2f[i], SEP)[1] for i in path]
    append!(toks, split(i2f[path[end]], SEP)[2:end])
    String.(filter(!=(bnd), toks))
end
segments_to_form(segs::Vector{String}) = replace(join(segs, ""), "_" => " ")

"""Predict all target cells of one held-out lemma from its source anchor. No gold input."""
function predict_lemma(bg::Background, lemma_id::AbstractString, src_cell::AbstractString,
                       src_segs::AbstractString, tgt_cells::Vector{String})
    c = bg.cfg
    st = lemma_state(bg, lemma_id, src_cell, src_segs, tgt_cells)
    T = length(tgt_cells)
    status = fill("ok", T)
    res = try
        c.decoder == :build_paths ? JudiLing.build_paths(
            DataFrame(query_index = 1:T),                  # data_val: only its row count is read
            st.C_h, st.S_tgt, st.F_h, st.Chat, st.A_h, st.i2f, st.paths_h;
            max_t = st.max_t, max_can = c.max_can, n_neighbors = c.n_neighbors,
            grams = c.grams, tokenized = true, sep_token = SEP, target_col = :segments,
            start_end_token = c.boundary, verbose = false) :
        JudiLing.learn_paths(
            st.data_h, DataFrame(query_index = 1:T),      # data_val: only its row count is read
            st.C_h, st.S_tgt, st.F_h, st.Chat, st.A_h, st.i2f, st.f2i;
            check_gold_path = false, gold_ind = nothing, Shat_val = nothing,
            max_t = st.max_t, max_can = c.max_can, threshold = c.threshold,
            is_tolerant = false, grams = c.grams, tokenized = true, sep_token = SEP,
            keep_sep = true, target_col = :segments, start_end_token = c.boundary,
            issparse = :auto, verbose = false)
    catch err
        @warn "learn_paths failed" lemma_id exception = (err, catch_backtrace())
        status .= "error"
        [JudiLing.Result_Path_Info_Struct[] for _ in 1:T]
    end
    n_src = length(unique(cue_ngrams(c, src_segs)))
    rows = NamedTuple[]
    for t in 1:T
        cands = res[t]
        preds = [path_segments(r.ngrams_ind, st.i2f, c.boundary) for r in cands]
        if status[t] == "ok" && isempty(cands)
            status[t] = "no_candidate"
        end
        top = [Dict("prediction" => segments_to_form(p), "support" => round(r.support, digits = 6))
               for (p, r) in zip(preds, cands)]
        push!(rows, (
            lemma_id = String(lemma_id), target_cell = tgt_cells[t],
            prediction = isempty(preds) ? "" : segments_to_form(preds[1]),
            prediction_segments = isempty(preds) ? "" : join(preds[1], SEP),
            status = status[t], n_candidates = length(cands),
            top_candidates = JSON.json(top),
            support = isempty(cands) ? NaN : cands[1].support,
            n_source_cues = n_src, n_source_cues_unseen = length(st.novel),
            unseen_target_features = join(st.unseen_features[t], ";"),
            binding_fit = st.binding_fit, max_t = st.max_t,
        ))
    end
    rows
end

"""Read a CONTRACT §6 query table, keeping only the contract columns (others are ignored)."""
function read_queries(path::AbstractString)
    q = CSV.read(path, DataFrame; types = String, stringtype = String)
    for col in QUERY_COLUMNS
        col in names(q) || error("query table lacks column $col")
    end
    q[:, QUERY_COLUMNS]
end

error_row(lid, cell) = (lemma_id = String(lid), target_cell = cell, prediction = "",
    prediction_segments = "", status = "error", n_candidates = 0, top_candidates = "[]",
    support = NaN, n_source_cues = -1, n_source_cues_unseen = -1, unseen_target_features = "",
    binding_fit = NaN, max_t = -1)

"""Run all queries lemma by lemma (each lemma independently on the same background)."""
function run_queries(bg::Background, q::DataFrame)
    out = NamedTuple[]
    times = Float64[]
    order = unique(q.lemma_id)
    for lid in order
        sub = q[q.lemma_id .== lid, :]
        length(unique(sub.source_segments)) == 1 || error("several source anchors for $lid")
        length(unique(sub.source_cell)) == 1 || error("several source cells for $lid")
        t0 = time()
        tc = String.(sub.target_cell)
        rows = try
            predict_lemma(bg, lid, sub.source_cell[1], sub.source_segments[1], tc)
        catch err
            @warn "lemma failed" lid exception = (err, catch_backtrace())
            [error_row(lid, cell) for cell in tc]
        end
        append!(out, rows)
        push!(times, time() - t0)
    end
    df = isempty(out) ? DataFrame([k => [] for k in PREDICTION_COLUMNS]) : DataFrame(out)
    df[:, PREDICTION_COLUMNS], times
end

# ----------------------------------------------------------------------------------------
# Gold-isolated diagnostics (call only after predictions are written)
# ----------------------------------------------------------------------------------------

"""Gold-side mapping diagnostics for one lemma. `gold` maps target_cell => list of gold
variant segment strings. The variant whose cue vector correlates best with Ĉ is reported.
Recomputes the same lemma state as prediction (deterministic); never feeds back into it."""
function score_mapping(bg::Background, lemma_id, src_cell, src_segs, tgt_cells::Vector{String},
                       gold::Dict{String,Vector{String}})
    c = bg.cfg
    st = lemma_state(bg, lemma_id, src_cell, src_segs, tgt_cells)
    rows = NamedTuple[]
    for (t, cell) in enumerate(tgt_cells)
        best = nothing
        for (vi, gsegs) in enumerate(gold[cell])
            gpath = cue_ngrams(c, gsegs)
            gng = unique(gpath)
            cg = zeros(length(st.f2i))
            n_out = 0
            for x in gng
                haskey(st.f2i, x) ? (cg[st.f2i[x]] = 1.0) : (n_out += 1)
            end
            r = cor(st.Chat[t, :], cg)
            n_below = count(x -> haskey(st.f2i, x) && st.Chat[t, st.f2i[x]] <= c.threshold, gng)
            row = (lemma_id = String(lemma_id), target_cell = cell,
                   chat_gold_cor = r, gold_variant_idx = vi - 1, n_gold_variants = length(gold[cell]),
                   n_gold_cues = length(gng), n_gold_cues_outside_inventory = n_out,
                   n_gold_cues_below_threshold = n_below,
                   gold_reachable = n_out == 0 && n_below == 0,
                   gold_path_longer_than_max_t = length(gpath) > st.max_t)
            if best === nothing || (isnan(best.chat_gold_cor) && !isnan(r)) || r > best.chat_gold_cor
                best = row
            end
        end
        push!(rows, best)
    end
    rows
end

# ----------------------------------------------------------------------------------------
# Batch job layer (one background per job; many jobs per Julia process)
# ----------------------------------------------------------------------------------------

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

"""Seen-item diagnostics on the training sample (training forms are not held-out gold)."""
function train_diagnostics(bg::Background)
    c = bg.cfg
    Shat = Matrix(bg.C) * bg.F
    comp = JudiLing.eval_SC(Shat, bg.S, bg.train, :segments)
    res = JudiLing.learn_paths(bg.train, bg.train, bg.C, bg.S, bg.F, bg.S * bg.G, bg.A,
        bg.i2f, bg.f2i; max_t = bg.max_len + c.max_t_margin, max_can = c.max_can,
        threshold = c.threshold, grams = c.grams, tokenized = true, sep_token = SEP,
        keep_sep = true, target_col = :segments, start_end_token = c.boundary, verbose = false)
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

"""Fit one background and predict its queries. `job` keys: train_csv, queries_csv,
out_dir, config (resolved ldl config incl. semantic_seed), job_config (written last as the
completion marker)."""
function run_job(job::AbstractDict)
    t0 = time()
    out = job["out_dir"]; mkpath(out)
    cfg = LDLConfig(job["config"])
    train, n_raw = read_training(job["train_csv"])
    q = read_queries(job["queries_csv"])
    bg = fit_background(train, cfg)
    pred, times = run_queries(bg, q)
    atomic_write(tmp -> CSV.write(tmp, pred), joinpath(out, "predictions.csv"))
    td = cfg.train_diagnostics ? train_diagnostics(bg) : Dict{String,Any}()
    probe = Dict(string(bg.train.lemma_id[i], "|", bg.train.cell_norm[i]) => bg.S[i, 1:5]
                 for i in 1:min(3, nrow(bg.train)))
    diag = Dict{String,Any}(
        "runner_version" => RUNNER_VERSION,
        "judiling_version" => string(pkgversion(JudiLing)), "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(), "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "n_train_rows_raw" => n_raw, "n_train_rows" => nrow(bg.train),
        "n_train_lemmas" => length(unique(bg.train.lemma_id)),
        "n_cues_background" => length(bg.f2i), "sem_dim" => cfg.sem_dim, "cue_ngram" => cfg.grams,
        "semantic_seed" => cfg.semantic_seed, "ridge_shift" => cfg.ridge_shift,
        "source_binding" => string(cfg.source_binding), "adjacency" => string(cfg.adjacency),
        "decoder" => string(cfg.decoder), "threshold" => cfg.threshold,
        "max_train_len" => bg.max_len,
        "n_query_lemmas" => length(times), "n_query_items" => nrow(pred),
        "status_counts" => Dict(s => count(==(s), pred.status) for s in unique(pred.status)),
        "n_items_with_unseen_source_cues" => count(>(0), pred.n_source_cues_unseen),
        "n_unseen_source_cues_total" => sum(unique(pred[:, [:lemma_id, :n_source_cues_unseen]]).n_source_cues_unseen; init = 0),
        "n_items_with_unseen_target_features" => count(!isempty, pred.unseen_target_features),
        "background_fit_seconds" => bg.fit_seconds,
        "per_lemma_seconds" => isempty(times) ? Dict() : Dict("median" => median(times),
            "mean" => mean(times), "max" => maximum(times), "first" => times[1], "total" => sum(times)),
        "job_seconds" => time() - t0,
        "semantic_probe" => probe,
    )
    merge!(diag, td)
    atomic_write(tmp -> open(io -> JSON.print(io, diag, 2), tmp, "w"), joinpath(out, "diagnostics.json"))
    atomic_write(tmp -> open(io -> JSON.print(io, job["job_config"], 2), tmp, "w"), joinpath(out, "job_config.json"))
    out
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
    for lid in unique(q.lemma_id)
        sub = q[q.lemma_id .== lid, :]
        tc = String.(sub.target_cell)
        for r in score_mapping(bg, lid, sub.source_cell[1], sub.source_segments[1], tc,
                               Dict(cell => gmap[(lid, cell)] for cell in tc))
            p = pmap[(lid, r.target_cell)]
            gforms = [segments_to_form(String.(split(g, SEP))) for g in gmap[(lid, r.target_cell)]]
            cands = ismissing(p.top_candidates) ? [] : JSON.parse(p.top_candidates)
            rank = findfirst(x -> x["prediction"] in gforms, cands)
            push!(rows, merge(r, (gold_in_top_candidates = rank !== nothing,
                                  gold_rank_in_candidates = rank === nothing ? 0 : rank)))
        end
    end
    atomic_write(tmp -> CSV.write(tmp, DataFrame(rows)), joinpath(out, "mapping_quality.csv"))
    open(io -> JSON.print(io, job["score_config"], 2), joinpath(out, "mapping_quality.json"), "w")
    out
end

end # module
