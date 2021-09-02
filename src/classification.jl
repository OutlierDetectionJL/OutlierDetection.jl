using Statistics:quantile

"""
    classify_percentile(threshold)

Create a percentile-based classifiction function that converts `scores_train::Scores` and `scores_test::Scores` to an
array of classes with `"normal"` indicating normal data and `"outlier"` indicating outliers. The conversion is based on
percentiles of the training data, i.e. all datapoints above the `threshold` percentile are considered outliers.

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
    classes::AbstractVector{String}
The vector of classes consisting of `"outlier"` and `"normal"` elements.

Examples
----------
    classify = classify_percentile(0.9)
    scores_train, scores_test = ([1, 2, 3], [4, 3, 2, 1, 0])
    classify(scores_train, scores_train) # [1, 1, -1]
    classify(scores_train, scores_test) # [-1, -1, 1, 1, 1]
"""
function classify_percentile(threshold::Real)
    function percentile(scores_train::Scores, scores_test::Scores)::Tuple{Labels, Labels}
        @assert 0 < threshold < 1
        t = quantile(scores_train, threshold)
        f = scores -> ifelse.(scores .> t, CLASS_OUTLIER, CLASS_NORMAL)
        f(scores_train), f(scores_test)
    end
    return percentile
end
