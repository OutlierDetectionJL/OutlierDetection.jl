CLASS_PARAMS = """    outlier_fraction::Float64
The fraction of outliers (number between 0 and 1) in the data is used to determine the score threshold to classify the
samples into inliers and outliers.

    classify::Union{Function, Nothing}
A function to transform a vector of scores to a vector of classes, where `"outlier"` represents an outlier and
`"normal"` represents a normal instance. See [`classify`](@ref) for a specific implementation."""

SCORE_PARAMS = """    combine::Function
A function to reduce a matrix, where each row represents an instance and each column represents a score of specific
detector, to a vector of scores for each instance. See `combine` for a specific implementation. *Note:* This function
is not called if the input to the evaluator consists of a single train/test scores tuple.

    normalize::Union{Function, Nothing}
A function to reduce a matrix, where each row represents an instance and each column a score of specific detector, to a
vector of scores for each instance. See [`normalize`](@ref) for a specific implementation."""

"""
    ScoreTransformer(combine = combine,           
                     normalize = normalize)

Transform the results of a single or multiple outlier detection models to combined and normalized scores.

Parameters
----------
$SCORE_PARAMS
```
"""
@detector mutable struct ScoreTransformer <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
end

"""
    ClassTransformer(normalize = scale_minmax,
                     combine = combine_mean,
                     classify = classify_percentile(0.9))

Transform the results of a single or multiple outlier detection models to binary classes, where `"normal"` represents
inliers and `"outlier"` represents outliers.

Parameters
----------
$CLASS_PARAMS

$SCORE_PARAMS
"""
default_percentile_threshold = classify_percentile(DEFAULT_THRESHOLD)
@detector mutable struct ClassTransformer <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
    classify::Function = default_percentile_threshold
end

function MLJ.transform(ev::ScoreTransformer, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    _, scores_test = to_scores(ev.normalize, ev.combine, scores...)
    to_univariate_finite(scores_test)
end

function MLJ.transform(ev::ClassTransformer, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    _, classes_test = to_classes(ev.normalize, ev.combine, ev.classify, scores...)
    to_categorical(classes_test)
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

function to_classes(normalize::Function, combine::Function, classify::Function,
                    scores::Tuple{Scores, Scores}...)::Tuple{Labels, Labels}
    scores_train, scores_test = to_scores(normalize, combine, scores...)
    classify(scores_train, scores_test)
end
