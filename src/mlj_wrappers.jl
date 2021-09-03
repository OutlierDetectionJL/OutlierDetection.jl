# Determine the input scitype of an array of detectors
to_input_scitype(detectors) = MLJ.glb(MLJ.input_scitype.(detectors)...)

mutable struct ProbabilisticUnsupervisedDetector{detector_names, input_scitype} <:
    MLJ.UnsupervisedProbabilisticComposite

    detectors::Vector{<:UnsupervisedDetector}
    normalize::Function
    combine::Function

    function ProbabilisticUnsupervisedDetector(detector_names, detectors, normalize, combine)
        input_scitypes = to_input_scitype(detectors)
        new{detector_names, input_scitypes}(detectors, normalize, combine)
    end
end

mutable struct ProbabilisticSupervisedDetector{detector_names, input_scitype} <: MLJ.ProbabilisticComposite

    detectors::Vector{<:Union{UnsupervisedDetector, SupervisedDetector}}
    normalize::Function
    combine::Function

    function ProbabilisticSupervisedDetector(detector_names, detectors, normalize, combine)
        input_scitypes = to_input_scitype(detectors)
        new{detector_names, input_scitypes}(detectors, normalize, combine)
    end
end

mutable struct DeterministicUnsupervisedDetector{detector_names, input_scitype} <:
    MLJ.UnsupervisedDeterministicComposite

    detectors::Vector{<:UnsupervisedDetector}
    normalize::Function
    combine::Function
    classify::Function

    function DeterministicUnsupervisedDetector(detector_names, detectors, normalize, combine, classify)
        input_scitypes = to_input_scitype(detectors)
        new{detector_names, input_scitypes}(detectors, normalize, combine, classify)
    end
end

mutable struct DeterministicSupervisedDetector{detector_names, input_scitype} <: MLJ.DeterministicComposite

    detectors::Vector{<:Union{UnsupervisedDetector, SupervisedDetector}}
    normalize::Function
    combine::Function
    classify::Function

    function DeterministicSupervisedDetector(detector_names, detectors, normalize, combine, classify)
        input_scitypes = to_input_scitype(detectors)
        new{detector_names, input_scitypes}(detectors, normalize, combine, classify)
    end
end

const ERR_DETECTOR_UNSUPPORTED = ArgumentError(
    "All detectors must subtype from `UnsupervisedDetector` or `SupervisedDetector`")

# TODO: enable programmatic construction of multiple detectors without names
const ERR_SPECIFY_NAMES = ArgumentError(
    "When specifying more than one detector, also specify names, as in "*
    "`ProbabilisiticDetector(detector1=KNNDetector(k=1), detector2=KNNDetector(k=5))`. ")
warn_ignore_detector_names(detector, named_detectors) =
    "Wrapping the single detector `$detector`. Ignoring $named_detectors. "

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

function ProbabilisticDetector(args...; normalize=scale_minmax, combine=combine_mean, named_detectors...)
    detector_names, detectors = extract_detector_args(args, named_detectors)
    args = (detector_names, detectors, normalize, combine)
    LUB = eltype(detectors)
    if LUB <: UnsupervisedDetector
        ProbabilisticUnsupervisedDetector(args...)
    elseif LUB <: Union{UnsupervisedDetector, SupervisedDetector}
        ProbabilisticSupervisedDetector(args...)
    else
        throw(ERR_DETECTOR_UNSUPPORTED)
    end
end

function DeterministicDetector(args...; normalize=scale_minmax, combine=combine_mean,
                               classify=classify_percentile(DEFAULT_THRESHOLD), named_detectors...)
    detector_names, detectors = extract_detector_args(args, named_detectors)
    args = (detector_names, detectors, normalize, combine, classify)
    LUB = eltype(detectors)
    if LUB <: UnsupervisedDetector
        DeterministicUnsupervisedDetector(args...)
    elseif LUB <: Union{UnsupervisedDetector, SupervisedDetector}
        DeterministicSupervisedDetector(args...)
    else
        throw(ERR_DETECTOR_UNSUPPORTED)
    end
end

# extract the raw scores from univariate finite distributions
raw_scores(dist) = MLJ.pdf.(dist, CLASS_OUTLIER)
raw_scores(dist::MLJ.Node) = MLJ.node(raw_scores, dist)

# extract the raw classes from categorical arrays
raw_classes(categorical) = MLJ.unwrap.(categorical)
raw_classes(categorical::MLJ.Node) = MLJ.node(raw_classes, categorical)

# augment the test scores with the training scores from the fit result (report)
augment(Xs) = mach -> MLJ.node((mach, Xs) -> (MLJ.report(mach).scores, MLJ.transform(mach, Xs)), mach, Xs)
augment_scores(model, Xs) = augment(Xs).(map(d -> MLJ.machine(d, Xs), getfield(model, :detectors)))
augment_scores(model, Xs, ys) = augment(Xs).(map(d -> MLJ.machine(d, Xs, ys), getfield(model, :detectors)))
transform_augmented(transformer, augmented_scores) = MLJ.transform(MLJ.machine(transformer), augmented_scores...)

function MLJ.fit(model::ProbabilisticUnsupervisedDetector, verbosity, X)
    Xs = MLJ.source(X)
    augmented_scores = augment_scores(model, Xs)
    transformer = Score(normalize=model.normalize, combine=model.combine)
    probs = transform_augmented(transformer, augmented_scores)
    network_mach = MLJ.machine(UnsupervisedProbabilistic(), Xs, predict=probs, transform=raw_scores(probs))
    MLJ.return!(network_mach, model, verbosity)
end

function MLJ.fit(model::DeterministicUnsupervisedDetector, verbosity, X)
    Xs = MLJ.source(X)
    augmented_scores = augment_scores(model, Xs)
    transformer = Class(normalize=model.normalize, combine=model.combine, classify=model.classify)
    classes = transform_augmented(transformer, augmented_scores)
    network_mach = MLJ.machine(UnsupervisedDeterministic(), Xs, predict=classes, transform=raw_classes(classes))
    MLJ.return!(network_mach, model, verbosity)
end

function MLJ.fit(model::ProbabilisticSupervisedDetector, verbosity, X, y)
    Xs, ys = MLJ.source(X), MLJ.source(y)
    augmented_scores = augment_scores(model, Xs, ys)
    transformer = Score(normalize=model.normalize, combine=model.combine)
    probs = transform_augmented(transformer, augmented_scores)
    network_mach = MLJ.machine(Probabilistic(), Xs, ys, predict=probs, transform=raw_scores(probs))
    MLJ.return!(network_mach, model, verbosity)
end

function MLJ.fit(model::DeterministicSupervisedDetector, verbosity, X, y)
    Xs, ys = MLJ.source(X), MLJ.source(y)
    augmented_scores = augment_scores(model, Xs, ys)
    transformer = Class(normalize=model.normalize, combine=model.combine, classify=model.classify)
    classes = transform_augmented(transformer, augmented_scores)
    network_mach = MLJ.machine(Deterministic(), Xs, ys, predict=classes, transform=raw_classes(classes))
    MLJ.return!(network_mach, model, verbosity)
end

DetectorUnion{N,I} = Union{
    ProbabilisticUnsupervisedDetector{N,I},
    DeterministicUnsupervisedDetector{N,I},
    ProbabilisticSupervisedDetector{N,I},
    DeterministicSupervisedDetector{N,I}
}

ProbabilisticDetectorUnion{detector_names} = Union{
    ProbabilisticUnsupervisedDetector{detector_names},
    ProbabilisticSupervisedDetector{detector_names}}

DeterministicDetectorUnion{detector_names} = Union{
    DeterministicUnsupervisedDetector{detector_names},
    DeterministicSupervisedDetector{detector_names}}

MLJ.input_scitype(::Type{<:DetectorUnion{N,I}}) where {N,I} = I
MLJ.output_scitype(::Type{<:DetectorUnion}) = AbstractVector{<:MLJ.Continuous}
MLJ.target_scitype(::Type{<:DetectorUnion}) = AbstractVector{<:Union{Missing,MLJ.OrderedFactor{2}}}

Base.propertynames(::ProbabilisticDetectorUnion{detector_names}) where detector_names =
    tuple(:normalize, :combine, detector_names...)

Base.propertynames(::DeterministicDetectorUnion{detector_names}) where detector_names =
    tuple(:normalize, :combine, :classify, detector_names...)

function Base.getproperty(model::ProbabilisticDetectorUnion{detector_names}, name::Symbol) where detector_names
    name === :normalize && return getfield(model, :normalize)
    name === :combine && return getfield(model, :combine)
    models = getfield(model, :detectors)
    for j in eachindex(detector_names)
        name === detector_names[j] && return models[j]
    end
    error("ProbabilisticDetector has no property $name")
end

function Base.getproperty(model::DeterministicDetectorUnion{detector_names}, name::Symbol) where detector_names
    name === :normalize && return getfield(model, :normalize)
    name === :combine && return getfield(model, :combine)
    name === :classify && return getfield(model, :classify)
    models = getfield(model, :detectors)
    for j in eachindex(detector_names)
        name === detector_names[j] && return models[j]
    end
    error("DeterministicDetector has no property $name")
end

function Base.setproperty!(model::ProbabilisticDetectorUnion{detector_names}, name::Symbol, val) where detector_names
    name === :normalize && return setfield!(model, :normalize, val)
    name === :combine && return setfield!(model, :combine, val)
    idx = findfirst(==(name), detector_names)
    idx isa Nothing || return getfield(model, :detectors)[idx] = val
    error("type ProbabilisticDetector has no property $name")
end

function Base.setproperty!(model::DeterministicDetectorUnion{detector_names}, name::Symbol, val) where detector_names
    name === :normalize && return setfield!(model, :normalize, val)
    name === :combine && return setfield!(model, :combine, val)
    name === :classify && return setfield!(model, :classify, val)
    idx = findfirst(==(name), detector_names)
    idx isa Nothing || return getfield(model, :detectors)[idx] = val
    error("type ProbabilisticDetector has no property $name")
end
