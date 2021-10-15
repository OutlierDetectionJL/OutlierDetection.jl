"""
    ScoreTransformer(combine = combine,           
                     normalize = normalize)

Transform the results of a single or multiple outlier detection models to combined and normalized scores.

Parameters
----------
    normalize::Function
A function to reduce a matrix, where each row represents an instance and each column a score of specific detector, to a
vector of scores for each instance. See [`scale_minmax`](@ref) for a specific implementation.

    combine::Function
A function to reduce a matrix, where each row represents an instance and each column represents the score of specific
detector, to a vector of scores for each instance. See [`combine_mean`](@ref) for a specific implementation.
"""
@detector mutable struct ScoreTransformer <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
end

# returns the augmented train/test scores
function MLJ.transform(ev::ScoreTransformer, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    scores_train, scores_test = to_scores(ev.normalize, ev.combine, scores...)
    scores_train, scores_test
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
