using Statistics:quantile, mean, median, std
using SpecialFunctions:erf

const _shared_params = """    scores_train::AbstractVector{<:Real}
A vector of training scores, typically the result of [`fit`](@ref) with a detector.

    scores_test::AbstractVector{<:Real}
A vector of test scores, typically the result of [`transform`](@ref) using a previously fitted detector."""

"""
    combine(scores_mat,
            strategy = :mean)

Combination method to merge outlier scores from multiple detectors by using a combination strateg. This function is
typically used within a [`Classifier`](@ref). TODO: Add AOM/MOA combination strategies 

Parameters
----------
    scores_mat::AbstractMatrix{T}
A row-major matrix, with each row representing the scores for a specific instance and each column representing a
specific detector.

    strategy::Symbol=:mean
Determines how to combine the scores of multiple detectors, e.g. maximum or mean of all scores.

Returns
----------
    combined_scores::AbstractVector{T}
The combined scores.e.g. the maximum of scores from different detectors.

Examples
----------
    scores = [1 2; 3 4; 5 6]
    combine(scores) # [1.5, 3.5, 5.5]
"""
function combine(scores_mat::AbstractMatrix{<:Real}, strategy::Symbol=:mean)::Scores
    @assert strategy in (:mean, :maximum, :median)

    if strategy == :mean
        return  dropdims(mean(scores_mat, dims=2), dims=2)
    elseif strategy == :maximum
        return dropdims(maximum(scores_mat, dims=2), dims=2)
    elseif strategy == :median
        return dropdims(median(scores_mat, dims=2), dims=2)
    end
end

# Combine scores in nested array format
combine(scores::AbstractVector{AbstractVector}, strategy::Symbol=:mean) = combine(reduce(hcat, scores), strategy)

"""
    normalize(scoresTrain,
              scoresTest)

Transform an array of scores into a range between [0,1] using min-max scaling.

Parameters
----------
$_shared_params

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
function normalize(scores_train::Scores, scores_test::Scores)::Tuple{Scores, Scores}
    minTrain = minimum(scores_train)
    maxTrain = maximum(scores_train)
    @assert minTrain < maxTrain # otherwise all scores are equal
    f = scores -> clamp.((scores .- minTrain) ./ (maxTrain - minTrain), 0, 1)
    f(scores_train), f(scores_test)
end
normalize(scores) = normalize(scores, scores)[1]  # only extract scores train because they are the same

"""
    unify(scores_train,
          scores_test)

Transform an array of scores into a range between [0,1] using unifying scores as described in [1].

Parameters
----------
$_shared_params

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
function unify(scores_train::Scores, scores_test::Scores)::Tuple{Scores, Scores}
    μ = mean(scores_train)
    σ = std(scores_train)
    @assert σ > 0 # otherwise all scores are equal
    f = scores -> clamp.(erf.((scores .- μ) ./ (σ * √2)), 0, 1)
    f(scores_train), f(scores_test)
end
unify(scores) = unify(scores, scores)[1] # only extract scores train because they are the same

"""
    classify(outlier_fraction,
             scores_train,
             scores_test)

Convert an array of scores to an array of classes with `1` indicating normal data and `-1` indicating outliers. The
conversion is based on percentiles of the training data, i.e. all datapoints above the `1 - outlier_fraction` percentile
are considered outliers.

Parameters
----------
    outlier_fraction::Real
The fraction of outliers (number between 0 and 1) in the data used to determine the score threshold to classify the
samples into inliers and outliers.

$_shared_params

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
function classify(outlier_fraction::Real, scores_train::Scores, scores_test::Scores)::Labels
    @assert 0 < outlier_fraction < 1
    ifelse.(scores_test .> quantile(scores_train, 1 - outlier_fraction), -1, 1)
end
classify(outlier_fraction, scores) = classify(outlier_fraction, scores, scores)
