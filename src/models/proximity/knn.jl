"""
    KNN(k=5,
        metric=Euclidean,
        algorithm=:kdtree,
        leafsize=10,
        reorder=true,
        reduction=:maximum)

Calculate the anomaly score of an instance based on the distance to its k-nearest neighbors.

Parameters
----------
$_knn_params

    reduction::Symbol
One of `(:maximum, :median, :mean)`. (`reduction=:maximum`) was proposed by [1]. Angiulli et al. [2] proposed sum to
reduce the distances, but mean has been implemented for numerical stability.

Examples
--------
$(_transform_unsupervised("KNN"))

References
----------
[1] Ramaswamy, Sridhar; Rastogi, Rajeev; Shim, Kyuseok (2000): Efficient Algorithms for Mining Outliers from Large Data
Sets.

[2] Angiulli, Fabrizio; Pizzuti, Clara (2002): Fast Outlier Detection in High Dimensional Spaces.
"""
MMI.@mlj_model mutable struct KNN <: UnsupervisedDetector
    k::Integer = 5::(_ > 0)
    metric::DI.Metric = DI.Euclidean()
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :brutetree, :balltree))
    leafsize::Integer = 10::(_ ≥ 0)
    reorder::Bool = true
    parallel::Bool = false
    reduction::Symbol = :maximum::(_ in (:maximum, :median, :mean))
end

struct KNNModel <: DetectorModel
    tree::NN.NNTree
end

function fit(detector::KNN, X::Data)::Tuple{KNNModel, Scores}
    # create the specified tree
    tree = buildTree(X, detector.metric, detector.algorithm, detector.leafsize, detector.reorder)

    # use tree to calculate distances
    idxs, dists = NN.knn(tree, X, detector.k)

    # reduce distances to outlier score
    scores = _knn(dists, detector.reduction)
    KNNModel(tree), scores
end

function transform(detector::KNN, model::KNNModel, X::Data)::Scores
    if detector.parallel
        idxs, dists = knn_parallel(model.tree, X, detector.k)
        return _knn(dists, detector.reduction)
    else
        idxs, dists = NN.knn(model.tree, X, detector.k)
        return _knn(dists, detector.reduction)
    end
end

@inline function _knn(distances::AbstractVector{<:AbstractVector}, reduction::Symbol)::Scores
    # Helper function to reduce `k` distances to a single distance.
    if reduction == :maximum
        return maximum.(distances)
    elseif reduction == :median
        return median.(distances)
    elseif reduction == :mean
        return mean.(distances)
    end
end