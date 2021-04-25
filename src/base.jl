using MacroTools

"""
    UnsupervisedDetector

This abstract type forms the basis for all implemented unsupervised outlier detection algorithms. To implement a new
`UnsupervisedDetector` yourself, you have to implement the `fit(detector, X)::Fit` and
`score(detector, model, X)::Result` methods. 
"""
abstract type UnsupervisedDetector <: MMI.Unsupervised end

"""
    SupervisedDetector

This abstract type forms the basis for all implemented supervised outlier detection algorithms. To implement a new
`SupervisedDetector` yourself, you have to implement the `fit(detector, X, y)::Fit` and
`score(detector, model, X)::Result` methods. 
"""
abstract type SupervisedDetector <: MMI.Deterministic end

"""
    Detector::Union{<:SupervisedDetector, <:UnsupervisedDetector}

The union type of all implemented detectors, including supervised, semi-supervised and unsupervised detectors. *Note:* A
semi-supervised detector can be seen as a supervised detector with a specific class representing unlabeled data.
"""
const Detector = Union{<:SupervisedDetector,<:UnsupervisedDetector}

"""
    Classifier

A classifier uses one or more detector scores produced by [`score`](@ref) and transforms them into labels, typically
with two classes describing inliers (`1`) and outliers (`-1`).
"""
abstract type Classifier <: MMI.Static end

"""
    Model

A `Model` represents the learned behaviour for specific [`Detector`](@ref). This might include parameters in parametric
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

The raw input data for every detector is defined as`AbstractArray{<:Real}` and should be a one observation per column
n-dimensional array. The input data used to [`fit`](@ref) a [`Detector`](@ref) and [`score`](@ref) [`Data`](@ref).
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
is used directly in later [`score`](@ref) calls and the (train-) scores are forwarded in [`score`](@ref).
"""
struct Fit
    model::Model
    scores::Scores
end

"""
    Result::Tuple{Scores, Scores}

Describes the result of using [`score`](@ref) with a [`Detector`](@ref) and is a tuple containing the train scores
achieved with [`fit`](@ref) and the test scores achieved with [`score`](@ref). We return both train and test scores
because this gives us the greatest flexibility in later score combination or classification.
"""
const Result = Tuple{Scores,Scores}

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

const _classifier = """
```julia
using OutlierDetection: Binarize, KNN, fit, score
detector = KNN()
X = rand(10, 100)
model = fit(detector, X)
train_scores, test_scores = score(detector, model, X)
clf = Binarize()
ŷ = detect(clf, train_scores, test_scores)
# or, if using multiple detectors
# ŷ = detect(clf, (train1, test1), (train2, test2), ...)
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
$(_score_unsupervised("KNN"))
""" # those definitions apply when X is not already a (transposed) abstract array
fit(detector::UnsupervisedDetector, X) = fit(detector, MMI.matrix(X; transpose=true)) # unsupervised call syntax
fit(detector::SupervisedDetector, X, y) = fit(detector, MMI.matrix(X; transpose=true), y)

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
    result::Result
Tuple of the achieved outlier scores of the given train and test data.

Examples
--------
$(_score_unsupervised("KNN"))
""" # definition applies when X is not already a (transposed) abstract array
score(detector::Detector, fitresult::Fit, X) = score(detector, fitresult, MMI.matrix(X; transpose=true))

"""
    detect(classifier,
           result...)

Convert a number of scores into inlier (`1`) / outlier (`-1`) classes, typically by using on a outlier-threshold on the
achieved scores.

Parameters
----------
    classifier::Classifier
A [`Classifier`](@ref) that implements the [`detect`](@ref) method.

    result::Result...
One or more [`score`](@ref) results (tuples) or alternatively a single vector of scores or two vectors, where the first
vector represents train scores and the second vector test scores.

Returns
----------
    result::Labels
A vector containing the binary inlier and outlier labels.

Examples
--------
$_classifier
""" # transforms single scores, or combination of train-test scores into a tuple
detect(classifier::Classifier, scores_train::Scores) = detect(classifier, (scores_train, scores_train))
detect(classifier::Classifier, scores_train::Scores, scores_test::Scores) =
    detect(classifier, (scores_train, scores_test))

"""
    @score

Helps with the definition of [`score`](@ref) for detectors, by unpacking the `.model` field of the second argument
directly into the argument and implicitly returning the values of the `.scores` field as the first tuple element of the
returned expression.
"""
macro score(fn)
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
