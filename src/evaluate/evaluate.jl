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
    Class(outlier_fraction = 0.1,
          classify = classify,
          combine = combine,           
          normalize = normalize)

A flexible, quantile-thresholding classifier that maps the outlier scores of a single or multiple outlier detection
models to binary classes, where `1` represents inliers and `-1` represents outliers.

Parameters
----------
$_class_params

$_score_params

Examples
----------
$_evaluator
"""
MMI.@mlj_model mutable struct Class <: Evaluator
    outlier_fraction::Float64 = 0.1::(0 < _ < 1)
    classify::Function = classify
    combine::Function = combine
    normalize::Union{Nothing, Function} = normalize
end

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
MMI.@mlj_model mutable struct Score <: Evaluator
    combine::Function = combine
    normalize::Union{Nothing, Function} = normalize
end

function detect(ev::Class, scores::Result...)::Labels
    _detect_helper(ev.outlier_fraction, ev.classify, ev.normalize, ev.combine, scores...)
end

function detect(ev::Score, scores::Result...)::Scores
    _detect_helper(nothing, nothing, ev.normalize, ev.combine, scores...)
end

function _detect_helper(outlier_fraction::Union{Real,Nothing},
                        classify::Union{Function,Nothing},
                        normalize::Union{Function,Nothing},
                        combine::Function,
                        scores::Result...)::Scores
    # transforms a variable number of equal-length scores into classes, where each input tuple represents the
    # train scores and test scores of a detector.

    # make sure that we have at least one tuple of train and test scores
    n_scores = length(scores)
    @assert n_scores > 0

    # conditionally normalize all scores if not nothing
    scores = isnothing(normalize) ? scores : map(tup -> normalize(tup...), scores)

    # [(train1, test1), (train2, test2)] -> matrix of scores for train [train1 train2]' and test [test1 test2]'
    # Note: The matrices contain one observation per row!
    reduce_cat = idx -> reduce(hcat, getfield.(scores, idx))
    scores_train, scores_test = reduce_cat(1), reduce_cat(2)

    # return an identity function that returns the existing scores if classify is nothing
    classify = isnothing(classify) ? (_, _, scores) -> scores : classify

    # scores_train and scores_test can be either a vector (if n_scores == 1) or a matrix (if n_scores > 1)
    n_scores == 1 ? classify(outlier_fraction, scores_train, scores_test) :
                    classify(outlier_fraction, combine(scores_train), combine(scores_test))
end
