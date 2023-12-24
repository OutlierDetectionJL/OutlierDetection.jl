SCORE_PARAMS = """  normalize::Function
A function to reduce a matrix, where each row represents an instance and each column a score of specific detector, to a
vector of scores for each instance. See [`scale_minmax`](@ref) for a specific implementation.

    combine::Function
A function to reduce a matrix, where each row represents an instance and each column represents the score of specific
detector, to a vector of scores for each instance. See [`combine_mean`](@ref) for a specific implementation."""

"""
    ScoreTransformer(combine = combine,           
                     normalize = normalize)

Transform the results of a single or multiple outlier detection models to combined and normalized scores.

Parameters
----------
$SCORE_PARAMS
"""
@detector mutable struct ScoreTransformer <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
end

"""
    ProbabilisticTransformer(combine = combine,           
                             normalize = normalize)

Transform the results of a single or multiple outlier detection models to combined univariate finite distributions.

Parameters
----------
$SCORE_PARAMS
"""
@detector mutable struct ProbabilisticTransformer <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
end

default_percentile_threshold = classify_quantile(DEFAULT_THRESHOLD)
"""
    DeterministicTransformer(combine = combine,           
                             normalize = normalize,
                             classify = classify_quantile(DEFAULT_THRESHOLD))

Transform the results of a single or multiple outlier detection models to combined categorical values.

Parameters
----------
$SCORE_PARAMS
"""
@detector mutable struct DeterministicTransformer <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
    classify::Function = default_percentile_threshold
end

const StaticTransformer = Union{
    ScoreTransformer,
    DeterministicTransformer,
    ProbabilisticTransformer
}

# returns the augmented train/test scores
function MLJ.transform(ev::StaticTransformer, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    to_scores(ev.normalize, ev.combine, scores...)
end

function MLJ.predict(ev::ProbabilisticTransformer, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    _, scores_test = to_scores(ev.normalize, ev.combine, scores...)
    to_univariate_finite(scores_test)
end

function MLJ.predict(ev::DeterministicTransformer, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    scores = to_scores(ev.normalize, ev.combine, scores...)
    _, classes_test = ev.classify(scores)
    to_categorical(classes_test)
end

function to_scores(normalize::Function, combine::Function, scores::Tuple{Scores, Scores}...)::Tuple{Scores, Scores}
    # make sure that we have at least one tuple of train and test scores
    @assert length(scores) > 0

    # [(train1, test1), (train2, test2)] -> matrix of scores for train [train1 train2]' and test [test1 test2]'
    # Note: The matrices contain one observation per row!
    scores_train, scores_test = combine(map(normalize, scores)...)
    scores_train, scores_test
end
