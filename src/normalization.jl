using Statistics:mean, std
using SpecialFunctions:erf

"""
    scale_minmax(scores)

Transform an array of scores into a range between [0,1] using min-max scaling.

Parameters
----------
      scores::Tuple{Scores, Scores}
A tuple consisting of two vectors representing training and test scores.

Returns
----------
    normalized_scores::Tuple{Scores, Scores}
The normalized train and test scores.

Examples
----------
    scores_train, scores_test = ([1, 2, 3], [4, 3, 2, 1, 0])
    scale_minmax(scores_train, scores_test) # ([0.0, 0.5, 1.0], [1.0, 1.0, 0.5, 0.0, 0.0])
"""
function scale_minmax(scores::Tuple{Scores, Scores})::Tuple{Scores, Scores}
    scores_train, scores_test = scores
    minTrain, maxTrain = extrema(scores_train)
    @assert minTrain < maxTrain "Cannot normalize scores if they are all equal"
    f = scores -> clamp.((scores .- minTrain) ./ (maxTrain - minTrain), 0, 1)
    f(scores_train), f(scores_test)
end

"""
    scale_unify(scores)

Transform an array of scores into a range between [0,1] using unifying scores as described in [1].

Parameters
----------
    scores::Tuple{Scores, Scores}
A tuple consisting of two vectors representing training and test scores.

Returns
----------
    unified_scores::Tuple{Scores, Scores}
The unified train and test scores.

Examples
----------
    scores_train, scores_test = ([1, 2, 3], [4, 3, 2, 1, 0])
    unify(scores_train, scores_test) # ([0.0, 0.0, 0.68..], [0.95.., 0.68.., 0.0, 0.0, 0.0])

References
----------
Kriegel, Hans-Peter; Kroger, Peer; Schubert, Erich; Zimek, Arthur (2011): Interpreting and Unifying Outlier Scores.
"""
function scale_unify(scores::Tuple{Scores, Scores})::Tuple{Scores, Scores}
    scores_train, scores_test = scores
    μ, σ = mean(scores_train), std(scores_train)
    @assert σ > 0 "Cannot normalize scores if they are all equal"
    f = scores -> clamp.(erf.((scores .- μ) ./ (σ * √2)), 0, 1)
    f(scores_train), f(scores_test)
end
