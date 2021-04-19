import Printf:@sprintf
const REGRESS_MARK = ":x:"
const IMPROVE_MARK = ":white_check_mark:"

function create_benchmark(detector)
    SUITE["$detector fit"] = BenchmarkGroup()
    SUITE["$detector pre"] = BenchmarkGroup()
    for dimension ∈ (1,10,100)
        for points ∈ (100,1000)
            data = rand(MersenneTwister(1), dimension, points)
            model = fit(detector, data)
            SUITE["$detector fit"]["dim=$dimension, points=$points"] = @benchmarkable model = fit($detector, $data)
            SUITE["$detector pre"]["dim=$dimension, points=$points"] = @benchmarkable predict($detector, $model, $data)
        end
    end
end

function run_benchmarks(name)
    paramspath = joinpath(dirname(@__FILE__), "benchmark.jld")
    if !isfile(paramspath)
        println("Tuning benchmarks...")
        tune!(SUITE)
        JLD.save(paramspath, "SUITE", params(SUITE))
    end
    loadparams!(SUITE, JLD.load(paramspath, "SUITE"), :evals, :samples)
    println("Running benchmarks...")
    results = run(SUITE, verbose = true, seconds = 2)
    JLD.save(joinpath(dirname(@__FILE__), name * ".jld"), "results", results)
end

function generate_report_single(name)
    result = load(joinpath(dirname(@__FILE__), name * ".jld"), "results")
    open(joinpath(dirname(@__FILE__), "results_single.md"), "w") do f
        printreport(f, minimum(result); iscomparisonjob = false)
    end
end

function generate_report_comparison(v1, v2)
    v1_res = load(joinpath(dirname(@__FILE__), v1 * ".jld"), "results")
    v2_res = load(joinpath(dirname(@__FILE__), v2 * ".jld"), "results")
    open(joinpath(dirname(@__FILE__), "results_compare.md"), "w") do f
        printreport(f, judge(minimum(v1_res), minimum(v2_res)); iscomparisonjob = true)
    end
end

function printreport(io::IO, results; iscomparisonjob::Bool=false)

    if iscomparisonjob
        print(io, """
                  A ratio greater than `1.0` denotes a possible regression (marked with $(REGRESS_MARK)), while a ratio less
                  than `1.0` denotes a possible improvement (marked with $(IMPROVE_MARK)). Only significant results - results
                  that indicate possible regressions or improvements - are shown below (thus, an empty table means that all
                  benchmark results remained invariant between builds).
                  | ID | time ratio | memory ratio |
                  |----|------------|--------------|
                  """)
    else
        print(io, """
                  | ID | time | GC time | memory | allocations |
                  |----|------|---------|--------|-------------|
                  """)
    end

    entries = BenchmarkTools.leaves(results)

    try
        entries = entries[sortperm(map(x -> string(first(x)), entries))]
    catch
    end

    for (ids, t) in entries
        if !(iscomparisonjob) || BenchmarkTools.isregression(t) || BenchmarkTools.isimprovement(t)
            println(io, resultrow(ids, t))
        end
    end

    return nothing
end

idrepr(id) = (str = repr(id); str[first(something(findfirst('[', str))):end])
intpercent(p) = string(ceil(Int, p * 100), "%")
resultrow(ids, t::BenchmarkTools.Trial) = resultrow(ids, minimum(t))

function resultrow(ids, t::BenchmarkTools.TrialEstimate)
    t_tol = intpercent(BenchmarkTools.params(t).time_tolerance)
    m_tol = intpercent(BenchmarkTools.params(t).memory_tolerance)
    timestr = string(BenchmarkTools.prettytime(BenchmarkTools.time(t)), " (", t_tol, ")")
    memstr = string(BenchmarkTools.prettymemory(BenchmarkTools.memory(t)), " (", m_tol, ")")
    gcstr = BenchmarkTools.prettytime(BenchmarkTools.gctime(t))
    allocstr = string(BenchmarkTools.allocs(t))
    return "| `$(idrepr(ids))` | $(timestr) | $(gcstr) | $(memstr) | $(allocstr) |"
end

function resultrow(ids, t::BenchmarkTools.TrialJudgement)
    t_tol = intpercent(BenchmarkTools.params(t).time_tolerance)
    m_tol = intpercent(BenchmarkTools.params(t).memory_tolerance)
    t_ratio = @sprintf("%.2f", BenchmarkTools.time(BenchmarkTools.ratio(t)))
    m_ratio =  @sprintf("%.2f", BenchmarkTools.memory(BenchmarkTools.ratio(t)))
    t_mark = resultmark(BenchmarkTools.time(t))
    m_mark = resultmark(BenchmarkTools.memory(t))
    timestr = "$(t_ratio) ($(t_tol)) $(t_mark)"
    memstr = "$(m_ratio) ($(m_tol)) $(m_mark)"
    return "| `$(idrepr(ids))` | $(timestr) | $(memstr) |"
end

resultmark(sym::Symbol) = sym == :regression ? REGRESS_MARK : (sym == :improvement ? IMPROVE_MARK : "")
