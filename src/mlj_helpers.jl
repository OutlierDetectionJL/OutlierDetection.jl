"""
    to_univariate_finite(scores::Scores)

Convert normalized scores to a vector of univariate finite distributions. 

Parameters
----------
    scores::[`Scores`](@ref)
Raw vector of scores.

Returns
----------
    scores::UnivariateFiniteVector{OrderedFactor{2}}
Univariate finite vector of scores.
"""
function to_univariate_finite(scores::Scores)
    MLJ.UnivariateFinite([CLASS_NORMAL, CLASS_OUTLIER], scores; augment=true, pool=missing, ordered=true)
end
to_univariate_finite(scores::MLJ.AbstractNode) = MLJ.node(to_univariate_finite, scores)

"""
    to_categorical(classes::AbstractVector{String})

Convert a vector of classes (with possible missing values) to a categorical vector.

Parameters
----------
    classes::[`Labels`](@ref)
A vector of classes.

Returns
----------
    classes::CategoricalVector{Union{Missing,String}, UInt32}
A categorical vector of classes.
"""
function to_categorical(classes::AbstractVector{String})
    levels = [CLASS_NORMAL, CLASS_OUTLIER]

    for class in unique(skipmissing(classes))
        @assert class in levels "Class $class must be in $levels"
    end

    # explicit cast to Vector{Union{String, Missing}} in case only missing values are passed
    c = Vector{Union{String,Missing}}(classes)
    # we cast to string if no missing values are present
    MLJ.categorical(try Vector{String}(c) catch c end, ordered=true, levels=levels)
end
to_categorical(classes::MLJ.AbstractNode) = MLJ.node(to_categorical, classes)

"""
    from_univariate_finite(scores)

Extract the raw scores from a vector of univariate finite distributions.

Parameters
----------
    scores::MLJ.UnivariateFiniteVector
A vector of univariate finite distributions.

Returns
----------
    scores::[`Scores`](@ref)
A vector of raw scores.
"""
from_univariate_finite(scores) = MLJ.pdf.(scores, CLASS_OUTLIER)
from_univariate_finite(scores::MLJ.Node) = MLJ.node(from_univariate_finite, scores)

"""
    from_categorical(classes)

Extract the raw classes from categorical arrays.

Parameters
----------
    classes::MLJ.CategoricalVector
A vector of categorical values.

Returns
----------
    classes::[`Labels`](@ref)
A vector of raw classes.
"""
from_categorical(categorical) = MLJ.unwrap.(categorical)
from_categorical(categorical::MLJ.Node) = MLJ.node(from_categorical, categorical)

# transform a fitresult (containing only the model) back to a Fit containing the model and training scores
to_fitresult(mach::MLJ.Machine{<:OD.Detector})::Fit = (mach.fitresult, mach.report.scores)

# this includes all composites defined in mlj_wrappers.jl
const DetectorComposites = Union{
    MLJ.Machine{<:MLJ.SupervisedDetectorComposite},
    MLJ.Machine{<:MLJ.UnsupervisedDetectorComposite},
    MLJ.Machine{<:MLJ.ProbabilisticUnsupervisedDetectorComposite},
    MLJ.Machine{<:MLJ.DeterministicUnsupervisedDetectorComposite},
    MLJ.Machine{<:MLJ.ProbabilisticSupervisedDetectorComposite},
    MLJ.Machine{<:MLJ.DeterministicSupervisedDetectorComposite}
}

const DetectorSurrogates = Union{
    MLJ.Machine{<:MLJ.SupervisedDetectorSurrogate},
    MLJ.Machine{<:MLJ.UnsupervisedDetectorSurrogate},
    MLJ.Machine{<:MLJ.ProbabilisticUnsupervisedDetectorSurrogate},
    MLJ.Machine{<:MLJ.DeterministicUnsupervisedDetectorSurrogate},
    MLJ.Machine{<:MLJ.ProbabilisticSupervisedDetectorSurrogate},
    MLJ.Machine{<:MLJ.DeterministicSupervisedDetectorSurrogate}
}

function check_mach(mach)
    # catch deserialized machine with no data:
    isempty(mach.args) && MLJ._err_serialized(augmented_transform)
    # catch not-yet-trained machine:
    mach.state > 0 || error("$mach has not been trained.")
end

function _augmented_transform(detector::Detector, fitresult::Fit, X)
    model, scores_train = fitresult
    scores_test = MLJ.transform(detector, model, X)
    return scores_train, scores_test
end

# 0. augmented_transform given rows:
"""
    augmented_transform(mach; rows=:)

Extends `transform` by additionally returning the training scores from detectors as a train/test score tuple.

Parameters
----------
    mach::MLJ.Machine{<:OD.Detector}
A fitted machine with a detector model.

    rows
Test data specified as rows machine-bound data (as in `transform`), but could also provide new test data `X`.

Returns
----------
    augmented_scores::Tuple{AbstractVector{<:Real}, AbstractVector{<:Real}}
A tuple of raw training and test scores.
"""
function augmented_transform(mach::MLJ.Machine{<:OD.Detector}; rows=:)
    check_mach(mach)
    return _augmented_transform(mach.model, to_fitresult(mach), selectrows(mach.model, rows, mach.data[1])...)
end

function augmented_transform(mach::DetectorComposites; rows=:)
    check_mach(mach)
    scores_train = mach.report.scores
    scores_test = mach.fitresult.transform(selectrows(mach.model, rows, mach.data[1])...)
    return scores_train, scores_test
end

function augmented_transform(mach::DetectorSurrogates; rows=:)
    check_mach(mach)
    scores_train = mach.report.scores
    scores_test = mach.fitresult.transform(rows=rows)
    return scores_train, scores_test
end

# 1. augmented_transform on machines, given *concrete* data:
function augmented_transform(mach::MLJ.Machine{<:OD.Detector}, X)
    check_mach(mach)
    return _augmented_transform(mach.model, to_fitresult(mach), reformat(mach.model, X)...)
end

function augmented_transform(mach::DetectorComposites, X)
    check_mach(mach)
    scores_train = mach.report.scores
    scores_test = mach.fitresult.transform(X)
    return scores_train, scores_test
end

# 2. operations on machines, given *dynamic* data (nodes):
function augmented_transform(mach::MLJ.Machine{<:OD.Detector}, X::MLJ.AbstractNode)
    MLJ.node(augmented_transform, mach, X)
end

function augmented_transform(mach::DetectorComposites, X::MLJ.AbstractNode)
    MLJ.node(augmented_transform, mach, X)
end
