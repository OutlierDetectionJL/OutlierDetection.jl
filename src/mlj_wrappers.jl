# Determine the input scitype of an array of detectors
to_input_scitype(detectors) = MLJ._glb(MLJ.input_scitype.(detectors)...)

# Determine the supported composite detector type
ex_to_eltype(type_symbol) = type_symbol == :Unsupervised ? UnsupervisedDetector :
                            Union{SupervisedDetector,UnsupervisedDetector}

# transform an expression specifying the type (unsupervised or supervised) to the corresponding composite types
# e.g. type_ex = Unsupervised, target_type = :"" would lead to a struct called :UnsupervisedCompositeDetector
# that is a subtype of :UnsupervisedDetectorComposite, if you would add target_type = :Probabilistic, the resulting
# struct would be called :ProbabilisticUnsupervisedCompositeDetector
function ex_to_types(type_ex, target_type::Symbol)
    type_symbol = Symbol(type_ex)
    detector_type = ex_to_eltype(type_symbol)
    composite_type = Symbol(target_type, type_symbol, :Detector, :NetworkComposite)
    type_composite = Symbol(target_type, type_symbol, :Composite, :Detector)
    detector_type, composite_type, type_composite
end

# create the struct expression for a scoring composite model from the input symbols
function score_composite(detector_type, composite_type, type_composite)
    return quote
        mutable struct $type_composite{detector_names,input_scitype} <: MLJ.$composite_type
            detectors::Vector{<:$detector_type}
            normalize::Function
            combine::Function

            function $type_composite(detectors, detector_names, normalize, combine)
                input_scitypes = to_input_scitype(detectors)
                new{detector_names,input_scitypes}(detectors, normalize, combine)
            end
        end
    end
end

# create the struct expression for a classifying composite model from the input symbols
function class_composite(detector_type, composite_type, type_composite)
    return quote
        mutable struct $type_composite{detector_names,input_scitype} <: MLJ.$composite_type
            detectors::Vector{<:$detector_type}
            normalize::Function
            combine::Function
            classify::Function

            function $type_composite(detectors, detector_names, normalize, combine, classify)
                input_scitypes = to_input_scitype(detectors)
                new{detector_names,input_scitypes}(detectors, normalize, combine, classify)
            end
        end
    end
end

macro CompositeDetector(type_ex)
    detector_type, composite_type, type_composite = ex_to_types(type_ex, Symbol())
    score_composite(detector_type, composite_type, type_composite)
end

macro ProbabilisticDetector(type_ex)
    detector_type, composite_type, type_composite = ex_to_types(type_ex, :Probabilistic)
    score_composite(detector_type, composite_type, type_composite)
end

macro DeterministicDetector(type_ex)
    detector_type, composite_type, type_composite = ex_to_types(type_ex, :Deterministic)
    class_composite(detector_type, composite_type, type_composite)
end

@CompositeDetector Supervised
@CompositeDetector Unsupervised
@ProbabilisticDetector Supervised
@ProbabilisticDetector Unsupervised
@DeterministicDetector Supervised
@DeterministicDetector Unsupervised

# TODO: enable programmatic construction of multiple detectors without names
const ERR_SPECIFY_NAMES = ArgumentError(
    "When specifying more than one detector, also specify names, as in " *
    "`ProbabilisiticDetector(detector1=KNNDetector(k=1), detector2=KNNDetector(k=5))`. ")
warn_ignore_detector_names(detector, named_detectors) =
    "Wrapping the single detector `$detector`. Ignoring $named_detectors. "

# check and transform the arguments given to composite detectors
function extract_detector_args(args, named_args)
    nt = NamedTuple(named_args)
    length(args) < 2 || throw(ERR_SPECIFY_NAMES)
    if length(args) == 1
        detector_names = (:detector,)
        detectors = [only(args),]
        isempty(nt) || @warn warn_ignore_detector_names(only(args), nt)
    else
        detector_names = keys(nt)
        detectors = collect(nt)
    end
    (detector_names, detectors)
end

# fails if it cannot convert to a vector of detectors
function return_composite(target_symbol::Symbol, detectors, args)
    detector_type = eltype(detectors)
    unsupervised = eval(Symbol(target_symbol, :Unsupervised, :Composite, :Detector))
    supervised = eval(Symbol(target_symbol, :Supervised, :Composite, :Detector))
    return detector_type <: UnsupervisedDetector ? unsupervised(detectors, args...) :
           detector_type <: SupervisedDetector ? supervised(detectors, args...) :
           supervised(convert(Vector{OD.Detector}, detectors), args...) # mixed case
end

"""
    ProbabilisticDetector(unnamed_detectors...;
                          normalize,
                          combine,
                          named_detectors...)

Transform one or more raw detectors into a single probabilistic detector (that returns outlier probabilities).
"""
function ProbabilisticDetector(args...; normalize=scale_minmax, combine=combine_mean, named_detectors...)
    detector_names, detectors = extract_detector_args(args, named_detectors)
    args = (detector_names, normalize, combine)
    return_composite(:Probabilistic, detectors, args)
end

"""
    DeterministicDetector(unnamed_detectors...;
                          normalize,
                          combine,
                          named_detectors...)

Transform one or more raw detectors into a single deterministic detector (that returns inlier and outlier classes).
"""
function DeterministicDetector(args...; normalize=scale_minmax, combine=combine_mean,
    classify=classify_quantile(DEFAULT_THRESHOLD), named_detectors...)
    detector_names, detectors = extract_detector_args(args, named_detectors)
    args = (detector_names, normalize, combine, classify)
    return_composite(:Deterministic, detectors, args)
end

"""
    CompositeDetector(unnamed_detectors...;
                      normalize,
                      combine,
                      named_detectors...)

Transform one or more raw detectors into a single composite detector (that returns raw outlier scores).
"""
function CompositeDetector(args...; normalize=scale_minmax, combine=combine_mean, named_detectors...)
    detector_names, detectors = extract_detector_args(args, named_detectors)
    args = (detector_names, normalize, combine)
    return_composite(Symbol(), detectors, args)
end

function unsupervised_transformer(model, Xs)
    detectors = getfield(model, :detectors)
    scores = map(detector -> MLJ.transform(MLJ.machine(detector, Xs), Xs), detectors)
    normalized_scores = map(node -> MLJ.node(n -> model.normalize(n), node), scores)
    return MLJ.node(model.combine, normalized_scores...)
end

function supervised_transformer(model, Xs, ys)
    detectors = getfield(model, :detectors)
    scores = map(detector -> MLJ.transform(MLJ.machine(detector, Xs, ys), Xs), detectors)
    normalized_scores = map(node -> MLJ.node(n -> model.normalize(n), node), scores)
    return MLJ.node(model.combine, normalized_scores...)
end

function MLJ.prefit(model::ProbabilisticUnsupervisedCompositeDetector, verbosity::Integer, X)
    Xs = MLJ.source(X)
    scores = unsupervised_transformer(model, Xs)
    return (;
        predict=to_univariate_finite(last(scores)),
        transform=scores,
        report=(;scores=first(scores))
   )
end

function MLJ.prefit(model::DeterministicUnsupervisedCompositeDetector, verbosity::Integer, X)
    Xs = MLJ.source(X)
    scores = unsupervised_transformer(model, Xs)
    classes = MLJ.node(model.classify, scores) |> last
    return (;
        predict=to_categorical(classes),
        transform=scores,
        report=(;scores=first(scores))
   )
end

function MLJ.prefit(model::ProbabilisticSupervisedCompositeDetector, verbosity::Integer, X, y)
    Xs, ys = MLJ.source(X), MLJ.source(y)
    scores = supervised_transformer(model, Xs, ys)
    return (;
        predict=to_univariate_finite(last(scores)),
        transform=scores,
        report=(;scores=first(scores))
   )
end

function MLJ.prefit(model::DeterministicSupervisedCompositeDetector, verbosity::Integer, X, y)
    Xs, ys = MLJ.source(X), MLJ.source(y)
    scores = supervised_transformer(model, Xs, ys)
    classes = MLJ.node(model.classify, scores) |> last
    return (;
        predict=to_categorical(classes),
        transform=scores,
        report=(;scores=first(scores))
   )
end

function MLJ.prefit(model::UnsupervisedCompositeDetector, verbosity::Integer, X)
    Xs = MLJ.source(X)
    scores = unsupervised_transformer(model, Xs)
    return (;
        transform=scores,
        report=(;scores=first(scores))
   )
end

function MLJ.prefit(model::SupervisedCompositeDetector, verbosity::Integer, X, y)
    Xs, ys = MLJ.source(X), MLJ.source(y)
    scores = supervised_transformer(model, Xs, ys)
    return (;
        transform=scores,
        report=(;scores=first(scores))
   )
end

# allow prefit of unsupervised detectors with supervised call
function MLJ.prefit(
    model::Union{UnsupervisedCompositeDetector,
        ProbabilisticUnsupervisedCompositeDetector,
        DeterministicUnsupervisedCompositeDetector},
        verbosity::Integer, X, y)
    return MLJ.prefit(model, verbosity, X)
end

ProbabilisticDetectorUnion{detector_names} = Union{
    ProbabilisticUnsupervisedCompositeDetector{detector_names},
    ProbabilisticSupervisedCompositeDetector{detector_names}}

DeterministicDetectorUnion{detector_names} = Union{
    DeterministicUnsupervisedCompositeDetector{detector_names},
    DeterministicSupervisedCompositeDetector{detector_names}}

CompositeDetectorUnion{detector_names} = Union{
    UnsupervisedCompositeDetector{detector_names},
    SupervisedCompositeDetector{detector_names}}

MLJ.input_scitype(::Type{<:Union{
    ProbabilisticUnsupervisedCompositeDetector{N,I},
    DeterministicUnsupervisedCompositeDetector{N,I},
    ProbabilisticSupervisedCompositeDetector{N,I},
    DeterministicSupervisedCompositeDetector{N,I},
    UnsupervisedCompositeDetector{N,I},
    SupervisedCompositeDetector{N,I}
}}) where {N,I} = I

Base.propertynames(::Union{ProbabilisticDetectorUnion{detector_names},
    CompositeDetectorUnion{detector_names}}) where {detector_names} =
    tuple(:normalize, :combine, detector_names...)

Base.propertynames(::DeterministicDetectorUnion{detector_names}) where {detector_names} =
    tuple(:normalize, :combine, :classify, detector_names...)

ERR_NO_PROPERTY(name) = error("CompositeDetector has no property $name")

function Base.getproperty(model::Union{ProbabilisticDetectorUnion{detector_names},
        CompositeDetectorUnion{detector_names}}, name::Symbol) where {detector_names}
    name === :normalize && return getfield(model, :normalize)
    name === :combine && return getfield(model, :combine)
    models = getfield(model, :detectors)
    for j in eachindex(detector_names)
        name === detector_names[j] && return models[j]
    end
    ERR_NO_PROPERTY(name)
end

function Base.getproperty(model::DeterministicDetectorUnion{detector_names}, name::Symbol) where {detector_names}
    name === :normalize && return getfield(model, :normalize)
    name === :combine && return getfield(model, :combine)
    name === :classify && return getfield(model, :classify)
    models = getfield(model, :detectors)
    for j in eachindex(detector_names)
        name === detector_names[j] && return models[j]
    end
    ERR_NO_PROPERTY(name)
end

function Base.setproperty!(model::Union{ProbabilisticDetectorUnion{detector_names},
        CompositeDetectorUnion{detector_names}}, name::Symbol, val) where {detector_names}
    name === :normalize && return setfield!(model, :normalize, val)
    name === :combine && return setfield!(model, :combine, val)
    idx = findfirst(==(name), detector_names)
    idx isa Nothing || return getfield(model, :detectors)[idx] = val
    ERR_NO_PROPERTY(name)
end

function Base.setproperty!(model::DeterministicDetectorUnion{detector_names}, name::Symbol, val) where {detector_names}
    name === :normalize && return setfield!(model, :normalize, val)
    name === :combine && return setfield!(model, :combine, val)
    name === :classify && return setfield!(model, :classify, val)
    idx = findfirst(==(name), detector_names)
    idx isa Nothing || return getfield(model, :detectors)[idx] = val
    ERR_NO_PROPERTY(name)
end
