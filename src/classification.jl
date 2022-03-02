using Statistics:quantile

"""
    classify_quantile(threshold)

Create a percentile-based classifiction function that converts `scores_train::Scores` and `scores_test::Scores` to an
array of classes with `"normal"` indicating normal data and `"outlier"` indicating outliers. The conversion is based on
percentiles of the training data, i.e. all datapoints above the `threshold` percentile are considered outliers.

Parameters
----------
    threshold::Real
The score threshold (number between 0 and 1) used to classify the samples into inliers and outliers.

    scores::Tuple{Scores, Scores}
A tuple consisting of two vectors representing training and test scores.

Returns
----------
    classes::Tuple{Vector{String}, Vector{String}}
The vector of classes consisting of `"outlier"` and `"normal"` elements.

Examples
----------
    classify = classify_quantile(0.9)
    scores_train, scores_test = ([1, 2, 3], [4, 3, 2])
    classify(scores_train, scores_train) # ["inlier", "inlier", "outlier"]
    classify(scores_train, scores_test) # ["outlier", "outlier", "inlier"]
"""
function classify_quantile(threshold::Real)
    function percentile(scores::Tuple{Scores, Scores})::Tuple{Vector{String}, Vector{String}}
        scores_train, scores_test = scores
        @assert 0 < threshold < 1
        t = quantile(scores_train, threshold)
        f = scores -> ifelse.(scores .> t, CLASS_OUTLIER, CLASS_NORMAL)
        f(scores_train), f(scores_test)
    end
    return percentile
end
