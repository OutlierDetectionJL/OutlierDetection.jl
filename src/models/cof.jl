"""
    COF(k = 5,
        metric = Euclidean(),
        algorithm = :kdtree,
        leafsize = 10,
        reorder = true,
        parallel = false)

Local outlier density based on chaining distance between graphs of neighbors, as described in [1].

Parameters
----------
$_k_param

$_knn_shared

Examples
--------
$(_score_unsupervised("COF"))

References
----------
[1] Tang, Jian; Chen, Zhixiang; Fu, Ada Wai-Chee; Cheung, David Wai-Lok (2002): Enhancing Effectiveness of Outlier
Detections for Low Density Patterns.
"""
MMI.@mlj_model mutable struct COF <: UnsupervisedDetector
    k::Integer = 5::(_ > 0)
    metric::DI.Metric = DI.Euclidean()
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :balltree))
    leafsize::Integer = 10::(_ ≥ 0)
    reorder::Bool = true
    parallel::Bool = false
end

struct COFModel <: Model
    # An efficient COF prediction requires us to store the full pairwise distance matrix of the training examples in
    # addition to the learned tree as well as the ACDs of the training examples.
    tree::NN.NNTree
    pdists::AbstractArray
    acds::AbstractVector
end

function fit(detector::COF, X::Data)::Fit
    # calculate pairwise distances in addition to building the tree; we could remove this once NearestNeighbors.jl
    # exports something like `allpairs`
    pdists = DI.pairwise(detector.metric, X, dims=2)

    # use tree to calculate distances
    tree = buildTree(X, detector.metric, detector.algorithm, detector.leafsize, detector.reorder)

    # We need k + 1 neighbors to calculate the chaining distance and have to make sure the indices are sorted 
    idxs, _ = knn_others(tree, X, detector.k + 1)
    acds = _calc_acds(idxs, pdists, detector.k)
    scores = _cof(idxs, acds, detector.k)
    Fit(COFModel(tree, pdists, acds), scores)
end

@score function score(detector::COF, model::Fit, X::Data)::Result
    if detector.parallel
        idxs, _ = knn_parallel(model.tree, X, detector.k + 1, true)
        return _cof(idxs, model.pdists, model.acds, detector.k)
    else
        idxs, _ = NN.knn(model.tree, X, detector.k + 1, true)
        return _cof(idxs, model.pdists, model.acds, detector.k)
    end
end

function _cof(idxs::AbstractVector{<:AbstractVector}, acds::AbstractVector, k:: Int)::Scores
    # Calculate the connectivity-based outlier factor from given acds
    cof = Vector{Float64}(undef, length(idxs))
    for (i, idx) in enumerate(idxs)
        @inbounds cof[i] = acds[i] * k / sum(acds[@view idx[2:end]])
    end
    cof
end

function _cof(idxs::AbstractVector{<:AbstractVector}, pdists::AbstractMatrix, acds:: AbstractVector, k::Int)::Scores
    # Calculate the connectivity-based outlier factor for test examples with given training distances and acds.
    cof = Vector{Float64}(undef, length(idxs))
    acdsTest = _calc_acds(idxs, pdists, k)
    for (i, idx) in enumerate(idxs)
        @inbounds cof[i] = acdsTest[i] * k / sum(acds[@view idx[2:end]])
    end
    cof
end

function _calc_acds(idxs::AbstractVector{<:AbstractVector}, pdists::AbstractMatrix, k::Int)::AbstractVector
    kplus1 = k + 1
    acds = zeros(length(idxs))
    for (i, idx) in enumerate(idxs)
        for j in 1:k
            # calculate the minimum distance (from all reachable points). That is, we sort the distances of a specific
            # point (given by idx[j+1]) according to the order of the current idx, where idx[1] specifies the idx of the
            # nearest neighbors and idx[k] specifies the idx of the k-th neighbor. We then restrict this so-called
            # set-based nearest path (SBN) to the points that are reachable with [begin:j]
            cost = minimum(pdists[idx, idx[j + 1]][begin:j])
            @inbounds acds[i] += ((2 * (kplus1 - j)) / (k * kplus1)) * cost
        end
    end
    acds
end

# """
# _cof_unused(distances, k)
# Unused version that does not rely on a pre-calculated tree, but is slower.
# """
# function _cof_unused(distances::AbstractMatrix, k:: Int)::AbstractVector
#     # pre allocate arrays
#     samples = size(distances, 2)
#     acds = zeros(samples)
#     cof = Vector{Float64}(undef, samples)
#     sbns = Array{Int, 2}(undef, (k, samples))
#     kplus1 = k + 1
#     # calculate sbns and acds
#     for i in 1:samples
#         # calculate the set based nearest path (nearest neighbors of current)
#         sbn_path = sortperm(distances[:, i], alg=PartialQuickSort(kplus1))[begin:kplus1]
#         sbns[:, i] = sbn_path[2:end]
#         for j in 1:k
#             # calculate the minimum distance (from all reachable points)
#             # e.g. point sbn_path[3] has two possible reachable points, sbn_path[1] and sbn_path[2]
#             cost = minimum(distances[sbn_path, sbn_path[j + 1]][begin:j])
#             acd = ((2 * (kplus1 - j)) / (k * kplus1)) * cost
#             acds[i] += acd
#         end
#     end
#     # calculate cof
#     for i in 1:samples
#         cof[i] = acds[i] * k / sum(acds[sbns[:, i]])
#     end
#     cof
# end
