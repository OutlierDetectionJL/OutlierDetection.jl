using JLD
using OutlierDetection
using BenchmarkTools
using Random: MersenneTwister
cd(dirname(@__FILE__))

cd("benchmark")

# Include utilities to print benchmark reports
include("./utils.jl")

# Create a benchmark suite
const SUITE = BenchmarkGroup()

# Create corresponding benchmarks
create_benchmark(DNN(1))
create_benchmark(DNN(1, parallel=true))
create_benchmark(KNN())
create_benchmark(KNN(parallel=true))
create_benchmark(LOF())
create_benchmark(LOF(parallel=true))
create_benchmark(COF())
create_benchmark(COF(parallel=true))
create_benchmark(ABOD())
create_benchmark(ABOD(parallel=true))

# Run the benchmark suite and generate a markdown report
run_benchmarks("benchmark")
generate_report_single("benchmark")
