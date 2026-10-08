# Phase-1 real-model smoke test of the LDL source-binding protocol.
# Usage: julia --project=julia -t 4 julia/scripts/smoke.jl <fixture_dir> <out_dir> [variants]
include(joinpath(@__DIR__, "..", "src", "LDLRunner.jl"))
using .LDLRunner
using CSV, DataFrames, JSON, Statistics, LinearAlgebra, SparseArrays, Random
import JudiLing

fixture = ARGS[1]
outdir = ARGS[2]
# variants: ';'-separated, each a ','-separated list of key=value overrides of `base`
variants = length(ARGS) >= 3 ? split(ARGS[3], ";") : ["source_binding=wug_refit"]
mkpath(outdir)

base = Dict(
    "cue_ngram" => 3, "boundary" => "#", "sem_dim" => 1000, "sem_sd_lexeme" => 4.0,
    "sem_sd_inflection" => 0.4, "sem_sd_noise" => 1.0, "ridge_lambda" => 0.0,
    "threshold" => 0.05, "max_can" => 10, "max_t_margin" => 4, "semantic_seed" => 1234567,
)

function levenshtein(a::Vector{String}, b::Vector{String})
    m, n = length(a), length(b)
    d = collect(0:n)
    for i in 1:m
        prev = d[1]; d[1] = i
        for j in 1:n
            cur = d[j+1]
            d[j+1] = min(d[j+1] + 1, d[j] + 1, prev + (a[i] == b[j] ? 0 : 1))
            prev = cur
        end
    end
    d[n+1]
end
segs(s) = isempty(s) ? String[] : String.(split(s, " "))
toseg(form) = [c == ' ' ? "_" : string(c) for c in form]

train1 = CSV.read(joinpath(fixture, "train_bg1.csv"), DataFrame; types = String)
train2 = CSV.read(joinpath(fixture, "train_bg2.csv"), DataFrame; types = String)
q = read_queries(joinpath(fixture, "test_queries.csv"))

# gold is read here only for scoring, after predictions exist
function score(pred::DataFrame)
    gold = CSV.read(joinpath(fixture, "gold.csv"), DataFrame; types = String)
    m = leftjoin(pred, gold, on = [:lemma_id, :target_cell])
    acc = Float64[]; ed = Float64[]; ned = Float64[]
    for r in eachrow(m)
        gv = split(r.gold_variants, "|")
        ok = r.status == "ok" && r.prediction in gv
        dist = minimum(levenshtein(segs(r.prediction_segments), toseg(g)) for g in gv)
        glen = minimum(length(toseg(g)) for g in gv)
        push!(acc, ok); push!(ed, dist); push!(ned, dist / glen)
    end
    m.correct = acc; m.edit_distance = ed; m.norm_edit_distance = ned
    m
end

summary = Dict{String,Any}()
for v in variants
    cfgd = copy(base)
    for kv in split(v, ",")
        k, val = split(kv, "=")
        cfgd[k] = something(tryparse(Int, val), tryparse(Float64, val), String(val))
    end
    cfg = LDLConfig(cfgd)
    bg = fit_background(train1, cfg)          # first call includes compilation
    bg = fit_background(train1, cfg)
    pred, times = run_queries(bg, q)
    _, times2 = run_queries(bg, q[q.lemma_id .== q.lemma_id[1], :])  # warm timing check
    tag = replace(v, "," => "_", "=" => "-")
    s_src_copy = 0
    CSV.write(joinpath(outdir, "predictions_$(tag).csv"), pred)
    m = score(pred)
    CSV.write(joinpath(outdir, "scored_$(tag).csv"), m)
    bycell = combine(groupby(m, :target_cell), :correct => mean => :acc, :edit_distance => mean => :ed)
    s = Dict(
        "n_items" => nrow(m), "accuracy" => mean(m.correct), "mean_edit_distance" => mean(m.edit_distance),
        "mean_norm_edit_distance" => mean(m.norm_edit_distance),
        "status_counts" => Dict(string(k) => count(==(k), m.status) for k in unique(m.status)),
        "items_with_unseen_source_cues" => count(>(0), m.n_source_cues_unseen),
        "mean_unseen_source_cues" => mean(m.n_source_cues_unseen),
        "items_with_unseen_target_features" => count(!isempty, m.unseen_target_features),
        "mean_binding_fit" => mean(filter(!isnan, unique(m.binding_fit))),
        "background_fit_seconds" => bg.fit_seconds, "n_background_rows" => nrow(bg.train),
        "n_cues_background" => length(bg.f2i),
        "per_lemma_seconds_median" => median(times[2:end]), "per_lemma_seconds_max" => maximum(times[2:end]),
        "per_lemma_seconds_first" => times[1],
        "prediction_equals_source" => mean(m.prediction .== [q.source_form[findfirst(==(l), q.lemma_id)] for l in m.lemma_id]),
        "by_cell" => Dict(r.target_cell => (r.acc, r.ed) for r in eachrow(bycell)),
    )
    summary[v] = s
    println("== ", v); println(JSON.json(s, 2))
    ex = first(m[:, [:lemma_id, :target_cell, :prediction, :gold_variants, :support, :status]], 16)
    show(stdout, ex; allcols = true, truncate = 40); println()
end

open(joinpath(outdir, "smoke_summary.json"), "w") do io
    JSON.print(io, summary, 2)
end
get(ENV, "SKIP_CHECKS", "0") == "1" && exit(0)
# ---------------------------------------------------------------- verification (primary variant)
cfg = LDLConfig(merge(base, Dict("source_binding" => "wug_refit", "adjacency" => "full")))
bg = fit_background(train1, cfg)
checks = Dict{String,Any}()

# (i) gold withheld / present / permuted in the query table -> identical predictions
gold = CSV.read(joinpath(fixture, "gold.csv"), DataFrame; types = String)
qg = leftjoin(read_queries(joinpath(fixture, "test_queries.csv")), gold, on = [:lemma_id, :target_cell])
qp = copy(qg); qp.gold_variants = shuffle(MersenneTwister(1), qp.gold_variants)
CSV.write(joinpath(outdir, "q_with_gold.csv"), qg); CSV.write(joinpath(outdir, "q_permuted_gold.csv"), qp)
p0, _ = run_queries(bg, q)
p1, _ = run_queries(bg, read_queries(joinpath(outdir, "q_with_gold.csv")))
p2, _ = run_queries(bg, read_queries(joinpath(outdir, "q_permuted_gold.csv")))
same(a, b) = isequal(a[:, Not(:binding_fit)], b[:, Not(:binding_fit)]) && isequal(a.binding_fit, b.binding_fit)
checks["i_gold_present_identical"] = same(p0, p1)
checks["i_gold_permuted_identical"] = same(p0, p2)

# (ii) per-lemma reset: alone vs batch vs reversed order vs after fitting other lemmas
lemmas = unique(q.lemma_id)
alone = vcat([run_queries(bg, q[q.lemma_id .== l, :])[1] for l in lemmas]...)
rev = vcat([run_queries(bg, q[q.lemma_id .== l, :])[1] for l in reverse(lemmas)]...)
rev = sort(rev, [order(:lemma_id, by = l -> findfirst(==(l), lemmas))])
qrev = vcat([q[q.lemma_id .== l, :] for l in reverse(lemmas)]...)
prev_, _ = run_queries(bg, qrev)
prev_ = sort(prev_, [order(:lemma_id, by = l -> findfirst(==(l), lemmas))])
checks["ii_alone_vs_batch_identical"] = same(alone, p0)
checks["ii_reversed_singletons_identical"] = same(rev, p0)
checks["ii_reversed_batch_identical"] = same(prev_, p0)
bgB = fit_background(train1, cfg)
checks["ii_background_refit_bitwise"] = bgB.F == bg.F && bgB.G == bg.G && bgB.S == bg.S

# (iii) semantic stability across backgrounds
bg2 = fit_background(train2, cfg)
shared = intersect(bg.train.lemma_id, bg2.train.lemma_id)
key(b) = Dict((b.train.lemma_id[i], b.train.cell_norm[i]) => i for i in 1:nrow(b.train))
k1 = key(bg); k2 = key(bg2)
diffs = [maximum(abs.(bg.S[k1[kk], :] .- bg2.S[k2[kk], :])) for kk in keys(k1) if haskey(k2, kk)]
checks["iii_shared_lemmas"] = length(shared)
checks["iii_shared_rows_max_abs_diff"] = maximum(diffs)
l = lemmas[1]
checks["iii_heldout_lexeme_vector_reproducible"] =
    LDLRunner.lexeme_vec(cfg, l) == LDLRunner.lexeme_vec(LDLConfig(merge(base, Dict("source_binding" => "lexeme_refit"))), l)

# rank-one update equals exact JudiLing refit; comprehension refit is a no-op for wug_refit
sub = q[q.lemma_id .== l, :]
st = LDLRunner.lemma_state(bg, l, sub.source_cell[1], sub.source_segments[1], String.(sub.target_cell))
Saug = vcat(bg.S, st.s_src'); Caug = Matrix(st.C_h)
Gex = JudiLing.make_transform_matrix(Saug, Caug; shift = cfg.ridge_shift, output_format = :dense)
checks["rls_vs_exact_G_max_abs_diff_on_Chat"] = maximum(abs.(st.S_tgt * Gex .- st.Chat))
Fex = JudiLing.make_transform_matrix(st.C_h, Saug; shift = cfg.ridge_shift, output_format = :dense)
checks["wug_comprehension_refit_max_abs_diff"] = maximum(abs.(Fex .- st.F_h))
cfgL = LDLConfig(merge(base, Dict("source_binding" => "lexeme_refit")))
bgL = fit_background(train1, cfgL)
stL = LDLRunner.lemma_state(bgL, l, sub.source_cell[1], sub.source_segments[1], String.(sub.target_cell))
GexL = JudiLing.make_transform_matrix(vcat(bgL.S, stL.s_src'), Matrix(stL.C_h); shift = cfgL.ridge_shift, output_format = :dense)
FexL = JudiLing.make_transform_matrix(stL.C_h, vcat(bgL.S, stL.s_src'); shift = cfgL.ridge_shift, output_format = :dense)
checks["lexeme_rls_vs_exact_G_max_abs_diff"] = maximum(abs.(stL.S_tgt * GexL .- stL.Chat))
checks["lexeme_rls_vs_exact_F_max_abs_diff"] = maximum(abs.(FexL .- stL.F_h))

# own full adjacency == JudiLing.make_full_adjacency_matrix
Aj = JudiLing.make_full_adjacency_matrix(bg.i2f; tokenized = true, sep_token = " ")
checks["full_adjacency_equals_judiling"] = (Aj .!= 0) == (bg.A .!= 0)

# seen-item (training) production diagnostic: decode background items from S G
Chat_tr = bg.S * bg.G
res_tr = JudiLing.learn_paths(bg.train, bg.train, bg.C, bg.S, bg.F, Chat_tr, bg.A, bg.i2f, bg.f2i;
    max_t = bg.max_len + cfg.max_t_margin, max_can = cfg.max_can, threshold = cfg.threshold,
    grams = 3, tokenized = true, sep_token = " ", keep_sep = true, target_col = :segments,
    start_end_token = "#", verbose = false)
checks["train_production_accuracy"] = mean(!isempty(r) && r[1].ngrams_ind == bg.paths[i] for (i, r) in enumerate(res_tr))
checks["train_comprehension_accuracy"] = JudiLing.eval_SC(Matrix(bg.C) * bg.F, bg.S, bg.train, :segments)

# gold-isolated mapping diagnostic (after predictions)
gseg = Dict((r.lemma_id, r.target_cell) => join(toseg(split(r.gold_variants, "|")[1]), " ") for r in eachrow(gold))
diag = NamedTuple[]
for l in lemmas
    local sb = q[q.lemma_id .== l, :]
    tc = String.(sb.target_cell)
    append!(diag, score_mapping(bg, l, sb.source_cell[1], sb.source_segments[1], tc,
                                Dict(c => gseg[(l, c)] for c in tc)))
end
dg = DataFrame(diag)
CSV.write(joinpath(outdir, "mapping_diagnostics_wug_refit_full.csv"), dg)
checks["mean_chat_gold_cor"] = mean(filter(!isnan, dg.chat_gold_cor))
checks["n_chat_gold_cor_nan"] = count(isnan, dg.chat_gold_cor)
checks["items_gold_outside_inventory"] = count(>(0), dg.n_gold_cues_outside_inventory)
checks["items_gold_cue_below_threshold"] = count(>(0), dg.n_gold_cues_below_threshold)
checks["items_gold_reachable"] = count((dg.n_gold_cues_outside_inventory .== 0) .& (dg.n_gold_cues_below_threshold .== 0))
checks["items_gold_longer_than_max_t"] = count(dg.gold_path_longer_than_max_t)
checks["n_items"] = nrow(dg)

println("== checks"); println(JSON.json(checks, 2))
summary["checks"] = checks
open(joinpath(outdir, "smoke_summary.json"), "w") do io
    JSON.print(io, summary, 2)
end
