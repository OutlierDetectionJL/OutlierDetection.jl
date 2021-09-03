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
combine_mean(scores::Scores...) = combine_mean(hcat(scores...))

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
combine_median(scores::Scores...) = combine_median(hcat(scores...))

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
combine_max(scores::Scores...) = combine_max(hcat(scores...))

_class_params = """    outlier_fraction::Float64
The fraction of outliers (number between 0 and 1) in the data is used to determine the score threshold to classify the
samples into inliers and outliers.

    classify::Union{Function, Nothing}
A function to transform a vector of scores to a vector of classes, where `"outlier"` represents an outlier and
`"normal"` represents a normal instance. See [`classify`](@ref) for a specific implementation."""

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
using OutlierDetection: Score, KNNDetector, fit, score
detector = KNNDetector()
X = rand(10, 100)
model = fit(detector, X)
train_scores, test_scores = score(detector, model, X)
ŷ = detect(Score(), train_scores, test_scores)
```
"""
@detector_model mutable struct Score <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
end

function to_scores(normalize::Function, combine::Function, scores::Tuple{Scores, Scores}...)::Tuple{Scores, Scores}
    # make sure that we have at least one tuple of train and test scores
    n_scores = length(scores)
    @assert n_scores > 0

    # normalize all scores (might be identity function)
    scores = map(score_tuple -> normalize(score_tuple...), scores)

    # [(train1, test1), (train2, test2)] -> matrix of scores for train [train1 train2]' and test [test1 test2]'
    # Note: The matrices contain one observation per row!
    reduce_cat = idx -> reduce(hcat, getfield.(scores, idx))
    scores_train, scores_test = combine(reduce_cat(1)), combine(reduce_cat(2))
    scores_train, scores_test
end

"""
    Class(normalize = scale_minmax,
          combine = combine_mean,
          classify = classify_percentile(0.9))
A flexible, percentile-thresholding classifier that maps the outlier scores of a single or multiple outlier detection
models to binary classes, where `"normal"` represents inliers and `"outlier"` represents outliers.

Parameters
----------
$_class_params

$_score_params

Examples
----------
TODO
"""
default_percentile_threshold = classify_percentile(DEFAULT_THRESHOLD)
@detector_model mutable struct Class <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
    classify::Function = default_percentile_threshold
end

function to_classes(normalize::Function,
                    combine::Function,
                    classify::Function,
                    scores::Tuple{Scores, Scores}...)::Tuple{Labels, Labels}
    scores_train, scores_test = to_scores(normalize, combine, scores...)
    classify(scores_train, scores_test)
end

function MLJ.transform(ev::Score, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    _, scores_test = to_scores(ev.normalize, ev.combine, scores...)
    to_univariate_finite(scores_test)
end

function MLJ.transform(ev::Class, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    _, classes_test = to_classes(ev.normalize, ev.combine, ev.classify, scores...)
    to_categorical(classes_test)
end
