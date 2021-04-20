const _knn_params = """    k::Integer
Number of neighbors (must be greater than 0).

    metric::Metric
This is one of the Metric types defined in the Distances.jl package. It is possible to define your own metrics by
creating new types that are subtypes of Metric.

    algorithm::Symbol
One of `(:kdtree, :balltree)`. In a `kdtree`, points are recursively split into groups using hyper-planes.
Therefore a KDTree only works with axis aligned metrics which are: Euclidean, Chebyshev, Minkowski and Cityblock.
A *brutetree* linearly searches all points in a brute force fashion and works with any Metric. A *balltree*
recursively splits points into groups bounded by hyper-spheres and works with any Metric.

    leafsize::Int
Determines at what number of points to stop splitting the tree further. There is a trade-off between traversing the
tree and having to evaluate the metric function for increasing number of points.

    reorder::Bool
While building the tree this will put points close in distance close in memory since this helps with cache locality.
In this case, a copy of the original data will be made so that the original data is left unmodified. This can have a
significant impact on performance and is by default set to true.

    parallel::Bool
Parallelize `transform` and `predict` using all threads available. The number of threads can be set with the
`JULIA_NUM_THREADS` environment variable. Note: `fit` is not parallel."""

function buildTree(X::AbstractArray, metric::DI.Metric, algorithm::Symbol, leafsize::Int, reorder::Bool)::NN.NNTree
    if algorithm == :kdtree
        return NN.KDTree(X, metric; leafsize, reorder)
    elseif algorithm == :balltree
        return NN.BallTree(X, metric; leafsize, reorder)
    end
end

function knn_parallel(tree::NN.NNTree, X::AbstractArray, k::Int,
    sort::Bool = false)::Tuple{AbstractVector, AbstractVector}
    # pre-allocate the result arrays (as in NearestNeighbors.jl)
    samples = size(X, 2)
    dists = [Vector{NN.get_T(eltype(X))}(undef, k) for _ in 1:samples]
    idxs = [Vector{Int}(undef, k) for _ in 1:samples]

    # get number of threads
    nThreads = Threads.nthreads()
    # partition the input array equally
    partition_size = samples ÷ nThreads + 1
    partitions = Iterators.partition(axes(X, 2), partition_size)
    Threads.@threads for idx = collect(partitions)
        @inbounds idxs[idx], dists[idx] = NN.knn(tree, view(X, :, idx), k, sort)
    end
    idxs, dists
end

function dnn_parallel(tree::NN.NNTree, X::AbstractArray, d::Real, sort::Bool = false)::AbstractVector
    # pre-allocate the result arrays (as in NearestNeighbors.jl)
    samples = size(X, 2)
    scores = Vector{Float64}(undef, samples)

    # get number of threads
    nThreads = Threads.nthreads()
    # partition the input array equally
    partition_size = samples ÷ nThreads + 1
    partitions = Iterators.partition(axes(X, 2), partition_size)
    Threads.@threads for idx = collect(partitions)
        @inbounds scores[idx] = _dnn(NN.inrange(tree, view(X, :, idx), d, sort))
    end
    scores
end

# TODO: Make a simple macro for copying lines of text to other places work with MLJ style definitions, see example
# below; this would allow us to share templates between different detectors
# macro copy(name, definition)
#     return quote
#         macro $(esc(name))()
#             esc($(Expr(:quote, definition)))
#         end
#     end
# end

# @copy KNNTemplate begin
#     k::Integer = 5::(_ > 0)
#     metric::DI.Metric = DI.Euclidean()
#     algorithm::Symbol = :kdtree::(_ in (:kdtree, :balltree))
#     leafsize::Integer = 10::(_ ≥ 0)
#     reorder::Bool = true
#     parallel::Bool = false
# end
