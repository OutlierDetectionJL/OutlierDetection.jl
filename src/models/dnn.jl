"""
    DNN(d = 0,
        metric = Euclidean(),
        algorithm = :kdtree,
        leafsize = 10,
        reorder = true,
        parallel = false)

Anomaly score based on the number of neighbors in a hypersphere of radius `d`. Knorr et al. [1] directly converted the
resulting outlier scores to labels, thus this implementation does not fully reflect the approach from the paper.

Parameters
----------
    d::Real
The hypersphere radius used to calculate the global density of an instance.

$_knn_shared

Examples
--------
$(_score_unsupervised("DNN"))

References
----------
[1] Knorr, Edwin M.; Ng, Raymond T. (1998): Algorithms for Mining Distance-Based Outliers in Large Datasets.
"""
@detector_model mutable struct DNN <: UnsupervisedDetector
    metric::DI.Metric = DI.Euclidean()
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :balltree))
    leafsize::Integer = 10::(_ ≥ 0)
    reorder::Bool = true
    parallel::Bool = false
    d::Real = 0::(_ > 0) # warns if `d` is not set
end

struct DNNModel <: Model
    tree::NN.NNTree
end

function fit(detector::DNN, X::Data)::Fit
    # create the specified tree
    tree = buildTree(X, detector.metric, detector.algorithm, detector.leafsize, detector.reorder)

    # use tree to calculate distances
    scores = dnn_others(NN.inrange(tree, X, detector.d))
    Fit(DNNModel(tree), scores)
end

function score(detector::DNN, fitresult::Fit, X::Data)::Score
    model = fitresult.model
    if detector.parallel
        # already returns scores
        return dnn_parallel(model.tree, X, detector.d)
    else
        return dnn(NN.inrange(model.tree, X, detector.d))
    end
end

@inline function dnn(idxs::AbstractVector{<:AbstractVector})::Score
    # Helper function to reduce the instances to a global density score.
    1 ./ (length.(idxs) .+ 0.1) # min score = 0, max_score = 10
end

@inline function dnn_others(idxs::AbstractVector{<:AbstractVector})::Score
    # Remove the (self) point previously added when fitting the tree, otherwise during `fit`, that point would always
    # be included in the density estimation
    1 ./ (length.(idxs) .- 0.9) # 1 - 0.1
end
