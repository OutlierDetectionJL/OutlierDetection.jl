using JLD
using OutlierDetection
using BenchmarkTools
using Random: MersenneTwister

# Include utilities to print benchmark reports
include("utils.jl")

# Create a benchmark suite
const SUITE = BenchmarkGroup()

# Create corresponding benchmarks
create_benchmark(KNN())
create_benchmark(KNN(parallel=true))

# Run the benchmark suite and generate a markdown report
run_benchmarks("benchmark")
generate_report_single("benchmark")
