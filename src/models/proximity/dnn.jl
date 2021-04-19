"""
    DNN(metric = Euclidean()
    algorithm = :kdtree
    leafsize = 10
    reorder = true
    parallel = false)

Anomaly score based on the number of neighbors in a hypersphere of radius `d`. Knorr et al. [1] directly converted the
resulting outlier scores to labels, thus this implementation does not fully reflect the approach from the paper.

Parameters
----------
$_knn_params

    d::Real
The hypersphere radius used to calculate the global density of an instance.

Examples
--------
$(_transform_unsupervised("DNN"))

References
----------
[1] Knorr, Edwin M.; Ng, Raymond T. (1998): Algorithms for Mining Distance-Based Outliers in Large Datasets.
"""
MMI.@mlj_model mutable struct DNN <: UnsupervisedDetector
    metric::DI.Metric = DI.Euclidean()
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :brutetree, :balltree))
    leafsize::Integer = 10::(_ ≥ 0)
    reorder::Bool = true
    parallel::Bool = false
    d::Real = 0::(_ > 0) # warns if `d` is not set
end

struct DNNModel <: DetectorModel
    tree::NN.NNTree
end

function fit(detector::DNN, X::Data)::Tuple{DNNModel, Scores}
    # create the specified tree
    tree = buildTree(X, detector.metric, detector.algorithm, detector.leafsize, detector.reorder)

    # use tree to calculate distances
    scores = _dnn(NN.inrange(tree, X, detector.d))
    DNNModel(tree), scores
end

function transform(detector::DNN, model::DNNModel, X::Data)::Scores
    if detector.parallel
        # already returns scores
        return dnn_parallel(model.tree, X, detector.d)
    else
        return _dnn(NN.inrange(model.tree, X, detector.d))
    end
end

@inline function _dnn(idxs:: AbstractVector{<:AbstractVector})::Scores
    # Helper function to reduce the instances to a global density score.
    1 ./ (length.(idxs) .+ 1e-10)
end
