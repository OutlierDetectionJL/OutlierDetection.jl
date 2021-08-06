using Statistics:mean, median

# TODO: Add AOM/MOA/LSCP combination strategies 

"""
    combine_mean(scores_mat)

Combination method to merge outlier scores from multiple detectors using the mean value of scores.

Parameters
----------
    scores_mat::AbstractMatrix{T}
A matrix, with each row representing the scores for a specific instance and each column representing a detector.

Returns
----------
    combined_scores::AbstractVector{T}
    The combined scores, i.e. column-wise mean.

Examples
----------
    scores = [1 2; 3 4; 5 6]
    combine_mean(scores) # [1.5, 3.5, 5.5]
"""
combine_mean(scores_mat::AbstractMatrix{<:Real}) = dropdims(mean(scores_mat, dims=2), dims=2)
combine_mean(scores::Score...) = combine_mean(hcat(scores...))

"""
    combine_median(scores_mat)

Combination method to merge outlier scores from multiple detectors using the median value of scores.

Parameters
----------
    scores_mat::AbstractMatrix{T}
A matrix, with each row representing the scores for a specific instance and each column representing a detector.

Returns
----------
    combined_scores::AbstractVector{T}
The combined scores, i.e. column-wise median.

Examples
----------
    scores = [1 2; 3 4; 5 6]
    combine_median(scores) # [1.5, 3.5, 5.5]
"""
combine_median(scores_mat::AbstractMatrix{<:Real}) = dropdims(median(scores_mat, dims=2), dims=2)
combine_median(scores::Score...) = combine_median(hcat(scores...))

"""
    combine_max(scores_mat)

Combination method to merge outlier scores from multiple detectors using the maximum value of scores.

Parameters
----------
    scores_mat::AbstractMatrix{T}
A matrix, with each row representing the scores for a specific instance and each column representing a detector.

Returns
----------
    combined_scores::AbstractVector{T}
The combined scores, i.e. column-wise maximum.

Examples
----------
    scores = [1 2; 3 4; 5 6]
    combine_max(scores) # [2, 4, 6]
"""
combine_max(scores_mat::AbstractMatrix{<:Real}) = dropdims(maximum(scores_mat, dims=2), dims=2)
combine_max(scores::Score...) = combine_max(hcat(scores...))

_class_params = """    outlier_fraction::Float64
The fraction of outliers (number between 0 and 1) in the data is used to determine the score threshold to classify the
samples into inliers and outliers.

    classify::Union{Function, Nothing}
A function to transform a vector of scores to a vector of bits, where 1 represents an outlier and 0 represents a normal
instance. *Hint:* Sometimes you don't want to transform your scores to classes, e.g. in ROC AUC evaluation, where you
can use `no_classify` to pass along the reduced (raw) scores. See [`classify`](@ref) for a specific implementation."""

_score_params = """    combine::Function
A function to reduce a matrix, where each row represents an instance and each column represents a score of specific
detector, to a vector of scores for each instance. See `combine` for a specific implementation. *Note:* This function
is not called if the input to the evaluator consists of a single train/test scores tuple.

    normalize::Union{Function, Nothing}
A function to reduce a matrix, where each row represents an instance and each column a score of specific detector, to a
vector of scores for each instance. See [`normalize`](@ref) for a specific implementation."""

"""
    Score(combine = combine,           
          normalize = normalize)
Transform the results of a single or multiple outlier detection models to combined and normalized scores.

Parameters
----------
$_score_params

Examples
----------
```julia
using OutlierDetection: Score, KNN, fit, score
detector = KNN()
X = rand(10, 100)
model = fit(detector, X)
train_scores, test_scores = score(detector, model, X)
ŷ = detect(Score(), train_scores, test_scores)
```
"""
MMI.@mlj_model mutable struct Scores <: MMI.Static
    normalize::Function = normalize
    combine::Function = combine_mean
end

function to_scores(normalize::Function, combine::Function, scores::Tuple{Score, Score}...)::Tuple{Score, Score}
    # make sure that we have at least one tuple of train and test scores
    n_scores = length(scores)
    @assert n_scores > 0

    # normalize all scores (might be identity function)
    scores = map(normalize, scores)

    # [(train1, test1), (train2, test2)] -> matrix of scores for train [train1 train2]' and test [test1 test2]'
    # Note: The matrices contain one observation per row!
    reduce_cat = idx -> reduce(hcat, getfield.(scores, idx))
    scores_train, scores_test = combine(reduce_cat(1)), combine(reduce_cat(2))
    scores_train, scores_test
end

"""
    Class(threshold = 0.9,
          normalize = normalize,
          combine = combine,
          classify = classify)
A flexible, quantile-thresholding classifier that maps the outlier scores of a single or multiple outlier detection
models to binary classes, where `1` represents inliers and `-1` represents outliers.

Parameters
----------
$_class_params

$_score_params

Examples
----------
TODO
"""
MMI.@mlj_model mutable struct Labels <: MMI.Static
    threshold::Float64 = 0.9::(0 < _ < 1)
    normalize::Function = normalize
    combine::Function = combine_mean
    classify::Function = classify
end

function to_labels(threshold::Float64, normalize::Function, combine::Function, classify::Function,
                    scores::Tuple{Score, Score}...)::Label
    scores_train, scores_test = to_scores(normalize, combine, scores)
    classify(threshold, scores_train, scores_test)
end
