
"""
    @surrogate(fn, name)

Create a surrogate model from a learning network, implicitly defining a composite
struct using `name` and a `prefit` function using `fn`.

Parameters
----------
    fn::Function
A function to reduce a matrix, where each row represents an instance and each column represents the score of specific
detector, to a vector of scores for each instance. See [`combine_mean`](@ref) for a specific implementation.
    name::Symbol
The name of the resulting composite model (the surrogate model).    
"""
macro surrogate(fn, name)
    esc(
        quote
            mutable struct $name <: $MLJ.AnnotatorNetworkComposite end
            function $MLJ.prefit(::$name, ::Integer, data...)
                $fn(map($MLJ.source, data)...)
            end
        end
    )
end
