using Combinatorics: combinations
using LinearAlgebra: dot, norm
using Statistics: var

"""
    ABOD(k = 5,
         metric = Euclidean(),
         algorithm = :kdtree,
         leafsize = 10,
         reorder = true,
         parallel = false,
         enhanced = false)

Determine outliers based on the angles to its nearest neighbors. This implements the `FastABOD` variant described in
the paper, that is, it uses the variance of angles to its nearest neighbors, not to the whole dataset, see [1]. 

*Notice:* The scores are inverted, to conform to our notion that higher scores describe higher outlierness.

Parameters
----------
$_knn_params

    enhanced::Bool
When `enhanced=true`, it uses the enhanced ABOD (EABOD) adaptation proposed by [2].

Examples
--------
$(_transform_unsupervised("ABOD"))

References
----------
[1] Kriegel, Hans-Peter; S hubert, Matthias; Zimek, Arthur (2008): Angle-based outlier detection in high-dimensional
data.

[2] Li, Xiaojie; Lv, Jian Cheng; Cheng, Dongdong (2015): Angle-Based Outlier Detection Algorithm with More Stable
Relationships.
"""
MMI.@mlj_model mutable struct ABOD <: UnsupervisedDetector
    k::Integer = 5::(_ > 0)
    metric::DI.Metric = DI.Euclidean()
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :brutetree, :balltree))
    leafsize::Integer = 10::(_ ≥ 0)
    reorder::Bool = true
    parallel::Bool = false
    enhanced::Bool = false
end

struct ABODModel <: DetectorModel
    # We have to store the tree to efficiently retrieve the indices to the nearest neighbors. Additionally, we have to
    # store the raw training data `X` for later angle calculations.
    X::AbstractArray
    tree::NN.NNTree
end

function fit(detector::ABOD, X::Data)::Tuple{ABODModel, Scores}
    # use tree to calculate distances
    tree = buildTree(X, detector.metric, detector.algorithm, detector.leafsize, detector.reorder)
    idxs, _ = NN.knn(tree, X, detector.k)
    scores = detector.enhanced ? _eabod(X, X, idxs, detector.k) : _abod(X, X, idxs, detector.k)
    ABODModel(X, tree), scores
end

function transform(detector::ABOD, model::ABODModel, X::Data)::Scores
    # TODO: We could also paralellize the abod score calculation.
    if detector.parallel
        idxs, _ = knn_parallel(model.tree, X, detector.k)
        return detector.enhanced ? _eabod(X, model.X, idxs, detector.k) : _abod(X, model.X, idxs, detector.k)
    else
        idxs, _ = NN.knn(model.tree, X, detector.k)
        return detector.enhanced ? _eabod(X, model.X, idxs, detector.k) : _abod(X, model.X, idxs, detector.k)
    end
end

function _abod(X::AbstractArray, Xtrain::AbstractArray, idxs::AbstractVector, k::Int)::Scores
    # Calculate the ABOF for all instances in X.
    scores = Vector{Float64}(undef, length(idxs))
    for i in eachindex(idxs)
        @inbounds scores[i] = _abof(X[:, i], idxs[i], Xtrain, k)
    end
    scores
end

function _abof(p::AbstractVector, idxs::AbstractVector, X::AbstractArray, k::Int)::Real
    # Calculate the angle-based outlier factor (ABOF). The ABOF is the variance over the angles between the difference
    # vectors of a point `p` to all pairs of points in its nearest neighbors weighted by the distance of the points.

    # all two-neighbor combinations
    combs = combinations(idxs, 2)
    # we know that there are binomial(k, 2) results for all two-neighbor combinations and can thus pre-allocate
    result = Vector{Float64}(undef, binomial(k, 2))
    for (i, (idx1, idx2)) in enumerate(combs)
        neighbor1 = p .- X[:, idx1]
        neighbor2 = p .- X[:, idx2]
        @inbounds result[i] = dot(neighbor1, neighbor2) / (norm(neighbor1)^2 * norm(neighbor2)^2)
    end
    # NaN means that at least one norm was zero, we use -1, because higher scores should describe outlierness
    -1 * var(Iterators.filter(!isnan, result))
end

function _eabod(X::AbstractArray, Xtrain::AbstractArray, idxs::AbstractVector, k::Int)::Scores
    # Calculate the EABOF for all instances in X.
    scores = Vector{Float64}(undef, length(idxs))
    for i in eachindex(idxs)
        @inbounds scores[i] = _eabof(X[:, i], idxs[i], Xtrain, k)
    end
    scores
end

function _eabof(p::AbstractVector, idxs::AbstractVector, X::AbstractArray, k::Int)::Real
    # Calculate the enhanced angle-based outlier factor (EABOF).

    # all two-neighbor combinations
    combs = combinations(idxs, 2)
    # we know that there are binomial(k, 2) results for all two-neighbor combinations and can thus pre-allocate
    result = Vector{Float64}(undef, binomial(k, 2))
    for (i, (idx1, idx2)) in enumerate(combs)
        neighbor1 = p .- X[:, idx1]
        neighbor2 = p .- X[:, idx2]
        norm1 = norm(neighbor1)
        norm2 = norm(neighbor2)
        @inbounds result[i] = (1 / (norm1 + norm2)) * (dot(neighbor1, neighbor2) / (norm1^2 * norm2^2))
    end
    # NaN means that at least one norm was zero, we use -1, because higher scores should describe outlierness
    -1 * var(Iterators.filter(!isnan, result))
end
