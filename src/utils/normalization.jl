using Statistics:mean, std
using SpecialFunctions:erf

"""
normalize(scoresTrain,
          scoresTest)

Transform an array of scores into a range between [0,1] using min-max scaling.

Parameters
----------
      scores_train::AbstractVector{<:Real}
A vector of training scores, typically the result of [`fit`](@ref) with a detector.

    scores_test::AbstractVector{<:Real}
A vector of test scores, typically the result of [`score`](@ref) using a previously fitted detector.

Returns
----------
normalized_scores::Tuple{AbstractVector{<:Real}, AbstractVector{<:Real}}
The normalized train and test scores.

Examples
----------
scores_train, scores_test = ([1, 2, 3], [4, 3, 2, 1, 0])
normalize(scores_train, scores_test) # ([0.0, 0.5, 1.0], [1.0, 1.0, 0.5, 0.0, 0.0])
normalize(scores_train) # [0.0, 0.5, 1.0]
"""
function normalize(scores_train::Score, scores_test::Score)::Tuple{Score, Score}
    minTrain = minimum(scores_train)
    maxTrain = maximum(scores_train)
    @assert minTrain < maxTrain "Cannot normalize scores if they are all equal"
    f = scores -> clamp.((scores .- minTrain) ./ (maxTrain - minTrain), 0, 1)
    f(scores_train), f(scores_test)
end
normalize(scores::Tuple{Score, Score}) = normalize(scores...)
normalize(scores::Score) = normalize(scores, scores)[1]  # only extract scores train because they are the same

"""
unify(scores_train,
      scores_test)

Transform an array of scores into a range between [0,1] using unifying scores as described in [1].

Parameters
----------
    scores_train::AbstractVector{<:Real}
A vector of training scores, typically the result of [`fit`](@ref) with a detector.

    scores_test::AbstractVector{<:Real}
A vector of test scores, typically the result of [`score`](@ref) using a previously fitted detector.

Returns
----------
unified_scores::Tuple{AbstractVector{<:Real}, AbstractVector{<:Real}}
The unified train and test scores.

Examples
----------
scores_train, scores_test = ([1, 2, 3], [4, 3, 2, 1, 0])
unify(scores_train, scores_test) # ([0.0, 0.0, 0.68..], [0.95.., 0.68.., 0.0, 0.0, 0.0])
unify(scores_train) # [0.0, 0.0, 0.68..]

References
----------
Kriegel, Hans-Peter; Kroger, Peer; Schubert, Erich; Zimek, Arthur (2011): Interpreting and Unifying Outlier Scores.
"""
function unify(scores_train::Score, scores_test::Score)::Tuple{Score, Score}
    μ = mean(scores_train)
    σ = std(scores_train)
    @assert σ > 0 "Cannot normalize scores if they are all equal"
    f = scores -> clamp.(erf.((scores .- μ) ./ (σ * √2)), 0, 1)
    f(scores_train), f(scores_test)
end
unify(scores::Tuple{Score, Score}) = unify(scores...)
unify(scores) = unify(scores, scores)[1] # only extract scores train because they are the same
