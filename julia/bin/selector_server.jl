# Persistent LDL selector process: one per acquisition job, so Julia starts and compiles once.
#   julia --project=julia -t <threads> selector_server.jl
# Protocol: one JSON object per line on stdin, one JSON reply per line on stdout.
#   {"cmd": "ping"}                                   -> {"ok": true, "runner_version": ...}
#   {"cmd": "score", "train_csv": p, "candidates_csv": p, "out_cells_csv": p,
#    "out_comp_csv": p, "configs": [cfg, ...], "blas_threads": n}
#       fits one background per config (one per semantic seed) on the training rows and
#       scores every candidate from that same background; writes the two CSVs.
#   {"cmd": "quit"}
# Diagnostics and warnings go to stderr; stdout carries replies only.
include(joinpath(@__DIR__, "..", "src", "LDLRunner.jl"))
using .LDLRunner
using CSV, DataFrames, JSON, LinearAlgebra

function handle_score(msg)
    LinearAlgebra.BLAS.set_num_threads(Int(get(msg, "blas_threads", 1)))
    train, n_raw = read_training(msg["train_csv"])
    cands = read_candidates(msg["candidates_csv"])
    cells = DataFrame[]; comps = DataFrame[]
    fit_s = Float64[]; score_s = Float64[]; n_cues = Int[]
    for (i, cd) in enumerate(msg["configs"])
        cfg = LDLConfig(cd)
        bg = fit_background(train, cfg)
        push!(fit_s, bg.fit_seconds); push!(n_cues, length(bg.f2i))
        t0 = time()
        cr, cp = LDLRunner.score_round(bg, cands)
        push!(score_s, time() - t0)
        for df in (cr, cp)
            df[!, :semantic_seed_idx] .= i - 1
            df[!, :semantic_seed] .= cfg.semantic_seed
        end
        push!(cells, cr); push!(comps, cp)
    end
    LDLRunner.atomic_write(tmp -> CSV.write(tmp, vcat(cells...)), msg["out_cells_csv"])
    LDLRunner.atomic_write(tmp -> CSV.write(tmp, vcat(comps...)), msg["out_comp_csv"])
    Dict("ok" => true, "n_train_rows" => nrow(train), "n_train_rows_raw" => n_raw,
         "n_candidates" => nrow(cands), "fit_seconds" => fit_s, "score_seconds" => score_s,
         "n_cues" => n_cues, "runner_version" => LDLRunner.RUNNER_VERSION)
end

for line in eachline(stdin)
    isempty(strip(line)) && continue
    reply = try
        msg = JSON.parse(line)
        cmd = get(msg, "cmd", "")
        if cmd == "quit"
            println(JSON.json(Dict("ok" => true, "bye" => true))); flush(stdout)
            break
        elseif cmd == "ping"
            Dict("ok" => true, "runner_version" => LDLRunner.RUNNER_VERSION, "threads" => Threads.nthreads())
        elseif cmd == "score"
            handle_score(msg)
        else
            Dict("ok" => false, "error" => "unknown cmd $(cmd)")
        end
    catch err
        Dict("ok" => false, "error" => sprint(showerror, err, catch_backtrace()))
    end
    println(JSON.json(reply)); flush(stdout)
end
