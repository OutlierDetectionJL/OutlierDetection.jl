using MacroTools

"""
    UnsupervisedDetector

This abstract type forms the basis for all implemented unsupervised outlier detection algorithms. To implement a new
`UnsupervisedDetector` yourself, you have to implement the `fit(detector, X)::Fit` and
`transform(detector, model, X)::Result` methods. 
"""
abstract type UnsupervisedDetector <: MMI.Unsupervised end

"""
    SupervisedDetector

This abstract type forms the basis for all implemented supervised outlier detection algorithms. To implement a new
`SupervisedDetector` yourself, you have to implement the `fit(detector, X, y)::Fit` and
`transform(detector, model, X)::Result` methods. 
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

A `Model` represents the learned behaviour for specific  [`Detector`](@ref). This might include parameters in parametric
models or other repesentations of the learned data in nonparametric models. In essence, it includes everything required
to transform an instance to an outlier score.
"""
abstract type Model end

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

"""
    Fit

A `Fit` bundles a [`Model`](@ref) and [`Scores`](@ref) achieved when fitting a [`Detector`](@ref). The model
is used directly in later [`transform`](@ref) calls and the (train-) scores are forwarded in [`transform`](@ref).
"""
struct Fit
    model::Model
    scores::Scores
end

"""
    Result::Tuple{Scores, Scores}

Describes the result of using [`transform`](@ref) with a [`Detector`](@ref) and is a tuple containing the train scores
achieved with [`fit`](@ref) and the test scores achieved with [`transform`](@ref). We return both train and test scores
because this gives us the greatest flexibility in later score combination or classification.
"""
const Result = Tuple{Scores,Scores}

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

Fit a specified unsupervised, supervised or semi-supervised outlier detector. That is, learn a `Model` from input data
`X` and, in the supervised and semi-supervised setting, labels `y`. In a supervised setting, the label `-1` represents
outliers and `1` inliers. In a semi-supervised setting, the label `0` additionally represents unlabeled data. *Note:*
Unsupervised detectors can be fitted without specifying `y`, otherwise `y` is simply ignore.

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
a corresponding [`Model`](@ref).

Parameters
----------
$_detector

    model::Model
The model learned from using [`fit`](@ref) with a supervised or unsupervised [`Detector`](@ref)

$_input_data

Returns
----------
    result::Result
Tuple of the achieved outlier scores of the given train and test data.

Examples
--------
$(_transform_unsupervised("KNN"))
"""
transform(detector::Detector, X) = fit(detector, MMI.matrix(X; transpose=true))

"""
    @unscorify

Helps with the definition of [`transform`](@ref) for detectors, by unpacking the `.model` field of the second argument
directly into the argument and implicitly returns the values of the `.scores` field as the first tuple element of the
returned expression.
"""
macro unscorify(fn)
    fn = MacroTools.longdef(fn)
    @capture(fn, function f_(detector_, result_::Fit, X_::Data)::Result body_ end) || error("Expected a function with
    three parameters f(`detector<:Detector`, `result::Fit`, `X::Data`)::Result and fully specified types.") 
    copy_result = :copy_result_unique_name

    # implicitly return the scores as the first tuple element for all return
    return_count = 0
    body = MacroTools.postwalk(body) do x
        @capture(x, ret_return) || return x
        # count the return
        return_count += 1

        # get the return expression
        ret_expr = ret.args[1]
        return :(($copy_result.scores, $(ret_expr)))
    end

    # check if there are any returns were found, if not, use the last body expression
    if return_count == 0
        body = :(($copy_result.scores, $(body.args[end])))
    end

    :(function $f($detector, $result, $X)
        $copy_result = $result;
        $result = $result.model;
        $body 
    end)
end
