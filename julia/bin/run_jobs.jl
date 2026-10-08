# Batch entry point: one Julia process handles a shard of LDL jobs (amortises compilation).
#   julia --project=julia -t <threads> run_jobs.jl <manifest.json>
# manifest = {"mode": "predict" | "score", "blas_threads": int, "jobs": [job, ...]}
# Each job failure is recorded in <out_dir>/error.json and does not stop the shard.
include(joinpath(@__DIR__, "..", "src", "LDLRunner.jl"))
using .LDLRunner
using JSON, LinearAlgebra, Dates

manifest = JSON.parsefile(ARGS[1])
mode = manifest["mode"]
LinearAlgebra.BLAS.set_num_threads(Int(get(manifest, "blas_threads", 1)))
n_fail = 0
for (i, job) in enumerate(manifest["jobs"])
    t0 = time()
    out = job["out_dir"]
    mkpath(out)
    errfile = joinpath(out, mode == "predict" ? "error.json" : "score_error.json")
    try
        isfile(errfile) && rm(errfile)
        mode == "predict" ? run_job(job) : mode == "score" ? score_job(job) : error("unknown mode $mode")
        println("[$(now())] $mode ok $(i)/$(length(manifest["jobs"])) $(round(time() - t0, digits = 1))s $out")
    catch err
        global n_fail += 1
        msg = sprint(showerror, err, catch_backtrace())
        open(io -> JSON.print(io, Dict("error" => msg, "mode" => mode), 2), errfile, "w")
        println(stderr, "[$(now())] $mode FAILED $out\n", msg)
    end
    flush(stdout)
end
exit(n_fail == 0 ? 0 : 1)
