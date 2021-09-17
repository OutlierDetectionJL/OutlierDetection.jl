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
