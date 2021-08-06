function MMI.fit(detector::UnsupervisedDetector, verbosity::Int, X)
    model = fit(detector, X)
    model, nothing, (scores = model.scores,)
end

function MMI.fit(detector::SupervisedDetector, verbosity::Int, X, y)
    model = fit(detector, X, y)
    model, nothing, (scores = model.scores,)
end

function MMI.transform(detector::Detector, fitresult::Fit, X)
    score(detector, fitresult, X)
end

function MMI.predict(detector::Detector, fitresult::Fit, X)
    _, scores_test = detector.normalize(fitresult.scores, score(detector, fitresult, X))
    univariate_finite(scores_test)
end

function MMI.predict_mode(detector::Detector, fitresult::Fit, X)
    scores_train, scores_test = detector.normalize(fitresult.scores, score(detector, fitresult, X))
    MMI.categorical(detector.classify(detector.threshold, scores_train, scores_test))
end

function MMI.transform(ev::Scores, _, scores::Tuple{Score, Score}...) # _ because there is no fitresult
    _, scores_test = to_scores(ev.normalize, ev.combine, scores...)
    univariate_finite(scores_test)
end

function MMI.transform(ev::Labels, _, scores::Tuple{Score, Score}...) # _ because there is no fitresult
    labels = to_labels(ev.threshold, ev.normalize, ev.combine, ev.classify, scores...)
    MMI.categorical(labels)
end

# extend evaluate to enable evaluation of unsupervised models
function MMI.evaluate(model::T, Xs, ys, measure; args...) where {T <: MMI.Unsupervised}
    ptype = MMI.prediction_type(measure)
    @assert ptype in (:probabilistic, :deterministic)
    # Xs, ys = MMI.source(X), MMI.source(y)
    ypred = MMI.transform(machine(model, Xs), Xs)
    # transform unsupervised model to supervised surrogate
    mach = ptype == :probabilistic ?
        machine(MMI.Probabilistic(), Xs, ys, predict=ypred) :
        machine(MMI.Deterministic(), Xs, ys, predict=ypred)
    MMI.evaluate!(mach; measure=measure, args...)
end

# helper to convert scores to univariate finite
function univariate_finite(scores::Score)
    MMI.UnivariateFinite([CLASS_NORMAL, CLASS_OUTLIER], scores; augment=true, pool=missing, ordered=false)
end

# specify scitypes
MMI.input_scitype(::Type{<:Detector}) = Union{MMI.Table(MMI.Continuous), AbstractMatrix{MMI.Continuous}}
MMI.output_scitype(::Type{<:Detector}) = AbstractVector{<:MMI.Continuous}
MMI.target_scitype(::Type{<:Detector}) = AbstractVector{<:MMI.Finite}
MMI.output_scitype(::Type{<:Scores}) = AbstractVector{<:MMI.Finite}
MMI.output_scitype(::Type{<:Labels}) = AbstractVector{<:MMI.Binary}

# data front-end for fit (supervised):
MMI.reformat(::SupervisedDetector, X, y) = (MMI.matrix(X, transpose=true), y)
MMI.reformat(::SupervisedDetector, X, y, w) = (MMI.matrix(X, transpose=true), y, w) 
MMI.selectrows(::SupervisedDetector, I, Xmatrix, y) = (view(Xmatrix, :, I), view(y, I))
MMI.selectrows(::SupervisedDetector, I, Xmatrix, y, w) = (view(Xmatrix, :, I), view(y, I), view(w, I))

# data front-end for fit (unsupervised)/predict/transform
MMI.reformat(::Detector, X) = (MMI.matrix(X, transpose=true),)
MMI.selectrows(::Detector, I, Xmatrix) = (view(Xmatrix, :, I),)

MODELS = (ABOD, COF, DNN, KNN, LOF, AE, DeepSAD, ESAD)
MMI.metadata_pkg.(MODELS,
    package_name="OutlierDetection.jl",
    package_uuid="262411bb-c475-4342-ba9e-03b8c0183ca6",
    package_url="https://github.com/davnn/OutlierDetection.jl",
    is_pure_julia=true,
    package_license="MIT",
    is_wrapper=false)
