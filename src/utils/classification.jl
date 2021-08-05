using Statistics:quantile

"""
    classify(threshold,
             scores_train,
             scores_test)

Convert an array of scores to an array of classes with `1` indicating normal data and `-1` indicating outliers. The
conversion is based on percentiles of the training data, i.e. all datapoints above the `threshold` percentile
are considered outliers.

Parameters
----------
    threshold::Real
The score threshold (number between 0 and 1) used to classify the samples into inliers and outliers.

    scores_train::AbstractVector{<:Real}
A vector of training scores, typically the result of [`fit`](@ref) with a detector.

    scores_test::AbstractVector{<:Real}
A vector of test scores, typically the result of [`score`](@ref) using a previously fitted detector.

Returns
----------
    classes::AbstractVector{<:Integer}
The vector of classes consisting of `-1` (outlier) and `1` (inlier) elements.

Examples
----------
    scores_train, scores_test = ([1, 2, 3], [4, 3, 2, 1, 0])
    classify(0.3, scores_train, scores_test) # [-1, -1, 1, 1, 1]
    classify(0.3, scores_train) # [1, 1, -1]
"""
function classify(threshold::Real, scores_train::Score, scores_test::Score)::Labels
    @assert 0 < threshold < 1
    ifelse.(scores_test .> quantile(scores_train, threshold), CLASS_OUTLIER, CLASS_NORMAL)
end
classify(threshold::Real, scores::Tuple{Score, Score}) = classify(threshold, scores...)
classify(threshold::Real, scores::Score) = classify(threshold, scores, scores)
