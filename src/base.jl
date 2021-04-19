"""
    UnsupervisedDetector

This abstract type forms the basis for all implemented unsupervised outlier detection algorithms. To implement a new
`UnsupervisedDetector` yourself, you have to implement the `fit(detector, X)::DetectorModel` and
`transform(detector, model, X)::Scores` methods. 
"""
abstract type UnsupervisedDetector <: MMI.Unsupervised end

"""
    SupervisedDetector

This abstract type forms the basis for all implemented supervised outlier detection algorithms. To implement a new
`SupervisedDetector` yourself, you have to implement the `fit(detector, X, y)::DetectorModel` and
`transform(detector, model, X)::Scores` methods. 
"""
abstract type SupervisedDetector <: MMI.Unsupervised end

"""
    Detector::Union{<:SupervisedDetector, <:UnsupervisedDetector}

The union type of all implemented detectors, including supervised, semi-supervised and unsupervised detectors. *Note:* A
semi-supervised detector can be seen as a supervised detector with a specific class representing unlabeled data.
"""
const Detector = Union{<:SupervisedDetector, <:UnsupervisedDetector}

"""
    DetectorModel

A `DetectorModel` represents the learned behaviour for specific detector. This might include parameters in parametric
models or other repesentations of the learned data in nonparametric models. In essence, it includes everything required
to transform an instance to an outlier score.
"""
abstract type DetectorModel end

"""
    Scores::AbstractVector{<:Real}

Scores are continuous values, where the range depends on the specific detector yielding the scores. *Note:* All
detectors return increasing scores and higher scores are associated with higher outlierness. Concretely, scores are
defined as an `AbstractVector{<:Real}`.
"""
const Scores = AbstractVector{<:Real}

"""
    Data::AbstractArray{<:Real}

The raw input data for every detector is defined as`AbstractArray{<:Real}` and should be a column-major n-dimensional
array. The input data used to [`fit`](@ref) a [`Detector`](@ref) and [`transform`](@ref) [`Data`](@ref).
"""
const Data = AbstractArray{<:Real}

"""
    Labels::AbstractArray{<:Integer}

Labels are used for supervision and evaluation and are defined as an `AbstractArray{<:Integer}`. The convention for
labels is that `-1` indicates outliers, `1` indicates inliers and `0` indicates unlabeled data in semi-supervised tasks.
"""
const Labels = AbstractArray{<:Integer}

const _input_data = """    X::Union{AbstractMatrix, Tables.jl-compatible}
Either a column-major matrix or a row-major [Tables.jl-compatible](https://github.com/JuliaData/Tables.jl) source."""

const _label_data = """    y::AbstractVector{<:Integer}
A vector of labels with `-1` indicating an outlier and `1` indicating an inlier."""

const _detector = """    detector::Detector
Any [`UnsupervisedDetector`](@ref) or [`SupervisedDetector`](@ref) implementation."""

_transform_unsupervised(name::String) = """
```julia
using OutlierDetection: $name, fit, transform
detector = $name()
X = rand(10, 100)
model, scores = fit(detector, X)
transform(detector, model, X)
```"""

_transform_supervised(name::String) = """
```julia
using OutlierDetection: $name, fit, transform
detector = $name()
X = rand(10, 100)
y = rand([-1,1], 100)
model, scores = fit(detector, X, y)
transform(detector, model, X)
```"""

"""
    fit(detector,
        X,
        y)

Fit a specified unsupervised, supervised or semi-supervised outlier detector. That is, learn a `DetectorModel` from
input data `X` and, in the supervised and semi-supervised setting, labels `y`. In a supervised setting, the label `-1`
represents outliers and `1` inliers. In a semi-supervised setting, the label `0` additionally represents unlabeled data.
*Note:* Unsupervised detectors can be fitted without specifying `y`, otherwise `y` is simply ignore.

Parameters
----------
$_detector

$_input_data

Returns
----------
    model::DetectorModel
The learned model of the given detector, which contains all the necessary information for later prediction.

    scores::Scores
The achieved outlier scores of the given training data `X`.

Examples
--------
$(_transform_unsupervised("KNN"))
"""
fit(detector::UnsupervisedDetector, X) = fit(detector, MMI.matrix(X; transpose=true)) # unsupervised call syntax
fit(detector::UnsupervisedDetector, X, _) = fit(detector, MMI.matrix(X; transpose=true)) # supervised call syntax
fit(detector::UnsupervisedDetector, X::Data, _) = fit(detector, X) # supervised call syntax for raw input data
fit(detector::SupervisedDetector, X, y) = fit(detector, MMI.matrix(X; transpose=true), y)

"""
    transform(detector,
              model,
              X)

Transform input data `X` to outlier scores using an [`UnsupervisedDetector`](@ref) or [`SupervisedDetector`](@ref) and
a corresponding [`DetectorModel`](@ref).

Parameters
----------
$_detector

    model::DetectorModel
The model learned from using [`fit`](@ref) with a supervised or unsupervised [`Detector`](@ref)

$_input_data

Returns
----------
    scores::Scores
The achieved outlier scores of the given test data `X`.

Examples
--------
$(_transform_unsupervised("KNN"))
"""
transform(detector::Detector, X) = fit(detector, MMI.matrix(X; transpose=true))
