using Tables:matrix

"""
    UnsupervisedDetector

This abstract type forms the basis for all implemented unsupervised outlier detection algorithms. To implement a new
`UnsupervisedDetector` yourself, you have to implement the `fit(detector, X)::Fit` and
`score(detector, model, X)::Score` methods. 
"""
abstract type UnsupervisedDetector <: MMI.Unsupervised end

"""
    SupervisedDetector

This abstract type forms the basis for all implemented supervised outlier detection algorithms. To implement a new
`SupervisedDetector` yourself, you have to implement the `fit(detector, X, y)::Fit` and
`score(detector, model, X)::Score` methods. 
"""
abstract type SupervisedDetector <: MMI.Deterministic end

"""
    Detector::Union{<:SupervisedDetector, <:UnsupervisedDetector}

The union type of all implemented detectors, including supervised, semi-supervised and unsupervised detectors. *Note:* A
semi-supervised detector can be seen as a supervised detector with a specific class representing unlabeled data.
"""
const Detector = Union{<:SupervisedDetector,<:UnsupervisedDetector}

"""
    Model

A `Model` represents the learned behaviour for specific [`Detector`](@ref). This might include parameters in parametric
models or other repesentations of the learned data in nonparametric models. In essence, it includes everything required
to transform an instance to an outlier score.
"""
abstract type Model end

"""
    Score::AbstractVector{<:Real}

Scores are continuous values, where the range depends on the specific detector yielding the scores. *Note:* All
detectors return increasing scores and higher scores are associated with higher outlierness.
"""
const Score = AbstractVector{<:Real}

"""
    Labels::AbstractVector{<:Integer}

Labels are used for supervision and evaluation and are defined as an `AbstractArray{<:Integer}`. The convention for
labels is that `-1` indicates outliers, `1` indicates inliers and `0` indicates unlabeled data in semi-supervised tasks.
"""
const Labels = AbstractVector{<:Integer}

"""
    Data::AbstractArray{<:Real}

The raw input data for every detector is defined as`AbstractArray{<:Real}` and should be a one observation per last axis
in an n-dimensional array. It represents the input data used to [`fit`](@ref) a [`Detector`](@ref) and 
[`score`](@ref) [`Data`](@ref).
"""
const Data = AbstractArray{<:Real}

"""
    Fit

A `Fit` bundles a [`Model`](@ref) and [`Scores`](@ref) achieved when fitting a [`Detector`](@ref). The model
is used directly in later [`score`](@ref) calls and the (train-) scores are forwarded in [`score`](@ref).
"""
struct Fit
    model::Model
    scores::Score
end

const _input_data = """    X::Union{AbstractMatrix, Tables.jl-compatible}
Either a matrix or a [Tables.jl-compatible](https://github.com/JuliaData/Tables.jl) data source, with one observation
per row and a number of feature columns."""

const _label_data = """    y::AbstractVector{<:Integer}
A vector of labels with `-1` indicating an outlier and `1` indicating an inlier."""

const _detector = """    detector::Detector
Any [`UnsupervisedDetector`](@ref) or [`SupervisedDetector`](@ref) implementation."""

_score_unsupervised(name::String) = """
```julia
using OutlierDetection: $name, fit, score
detector = $name()
X = rand(10, 100)
model = fit(detector, X)
train_scores, test_scores = score(detector, model, X)
```"""

_score_supervised(name::String) = """
```julia
using OutlierDetection: $name, fit, score
detector = $name()
X = rand(10, 100)
y = rand([-1,1], 100)
model = fit(detector, X, y)
train_scores, test_scores = score(detector, model, X)
```"""

const _result = """
```julia
using OutlierDetection: Class, KNN, fit, score
detector = KNN()
X = rand(10, 100)
model = fit(detector, X)
train_scores, test_scores = score(detector, model, X)
ŷ = detect(Class(), train_scores, test_scores)
# or, if using multiple detectors
# ŷ = detect(Class(), (train1, test1), (train2, test2), ...)
```"""

"""
    fit(detector,
        X,
        y)

Fit a specified unsupervised, supervised or semi-supervised outlier detector. That is, learn a `Model` from input data
`X` and, in the supervised and semi-supervised setting, labels `y`. In a supervised setting, the label `-1` represents
outliers and `1` inliers. In a semi-supervised setting, the label `0` additionally represents unlabeled data. *Note:*
Unsupervised detectors can be fitted without specifying `y`.

Parameters
----------
$_detector

$_input_data

Returns
----------
    fit::Fit
The learned model of the given detector, which contains all the necessary information for later prediction and the
achieved outlier scores of the given input data `X`.

Examples
--------
$(_score_unsupervised("KNN"))
""" # those definitions apply when X is not already a (transposed) abstract array
fit(detector::UnsupervisedDetector, X) = fit(detector, matrix(X; transpose=true)) # unsupervised call syntax
fit(detector::SupervisedDetector, X, y) = fit(detector, matrix(X; transpose=true), y)

"""
    score(detector,
          model,
          X)

Transform input data `X` to outlier scores using an [`UnsupervisedDetector`](@ref) or [`SupervisedDetector`](@ref) and
a corresponding [`Model`](@ref).

Parameters
----------
$_detector

    model::Model
The model learned from using [`fit`](@ref) with a supervised or unsupervised [`Detector`](@ref)

$_input_data

Returns
----------
    result::Score
Tuple of the achieved outlier scores of the given train and test data.

Examples
--------
$(_score_unsupervised("KNN"))
""" # definition applies when X is not already a (transposed) abstract array
score(detector::Detector, fitresult::Fit, X) = score(detector, fitresult, matrix(X; transpose=true))
