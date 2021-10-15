"""
    to_univariate_finite(scores::Scores)

Convert normalized scores to a vector of univariate finite distributions. 

Parameters
----------
    scores::[`Scores`](@ref)

Returns
----------
    fit::UnivariateFiniteVector{OrderedFactor{2}}
The learned model of the given detector, which contains all the necessary information for later prediction and the
achieved outlier scores of the given input data `X`.
"""
function to_univariate_finite(scores::Scores)
    MLJ.UnivariateFinite([CLASS_NORMAL, CLASS_OUTLIER], scores; augment=true, pool=missing, ordered=true)
end
to_univariate_finite(scores::MLJ.AbstractNode) = MLJ.node(to_univariate_finite, scores)

"""
    to_categorical(classes::Labels)

Convert a vector of classes (with possible missing values) to a categorical vector.

Parameters
----------
    classes::[`Labels`](@ref)
A vector of classes.

Returns
----------
    fit::CategoricalVector{Union{Missing, String},UInt32}
The learned model of the given detector, which contains all the necessary information for later prediction and the
achieved outlier scores of the given input data `X`.
"""
function to_categorical(classes::Labels)
    # explicit cast to Vector{Union{String, Missing}} in case only missing values are passed
    MLJ.categorical(Vector{Union{String, Missing}}(classes), ordered=true, levels=[CLASS_NORMAL, CLASS_OUTLIER])
end
to_categorical(classes::MLJ.AbstractNode) = MLJ.node(to_categorical, classes)

"""
    raw_scores(scores)

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
    raw_scores(scores)

Extract the raw classes from categorical arrays.

Parameters
----------
    scores::MLJ.CategoricalVector
A vector of categorical values.

Returns
----------
    scores::[`Labels`](@ref)
A vector of raw classes.
"""
from_categorical(categorical) = MLJ.unwrap.(categorical)
from_categorical(categorical::MLJ.Node) = MLJ.node(from_categorical, categorical)
