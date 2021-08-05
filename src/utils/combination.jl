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
