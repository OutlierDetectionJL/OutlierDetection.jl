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
