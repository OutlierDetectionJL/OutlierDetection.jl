function MMI.fit(detector::UnsupervisedDetector, verbosity::Int, X)
    model = fit(detector, X)
    model, nothing, (scores=model.scores,)
end

function MMI.fit(detector::SupervisedDetector, verbosity::Int, X, y)
    model = fit(detector, X, y)
    model, nothing, (scores=model.scores,)
end

function MMI.transform(detector::Detector, fitresult::Fit, X)
    score(detector, fitresult, X)
end

function MMI.predict(detector::Detector, fitresult::Fit, X)
    score(detector, fitresult, X)
end

function MMI.transform(clf::Classifier, _, scores::Result...) # _ because there is no fitresult
    detect(clf, scores...)
end

# specify scitypes
MMI.input_scitype(::Type{<:Detector}) = Union{MMI.Table(MMI.Continuous), AbstractMatrix{MMI.Continuous}}
MMI.output_scitype(::Type{<:Detector}) = Tuple{AbstractVector{<:MMI.Continuous}, AbstractVector{<:MMI.Continuous}}

# data front-end for fit (supervised):
MMI.reformat(::SupervisedDetector, X, y) = (MMI.matrix(X, transpose=true), y)
MMI.reformat(::SupervisedDetector, X, y, w) = (MMI.matrix(X, transpose=true), y, w) 
MMI.selectrows(::SupervisedDetector, I, Xmatrix, y) = (view(Xmatrix, :, I), view(y, I))
MMI.selectrows(::SupervisedDetector, I, Xmatrix, y, w) = (view(Xmatrix, :, I), view(y, I), view(w, I))

# data front-end for fit (unsupervised)/predict/transform
MMI.reformat(::Detector, X) = (MMI.matrix(X, transpose=true),)
MMI.selectrows(::Detector, I, Xmatrix) = (view(Xmatrix, :, I),)

MODELS = (ABOD, COF, DNN, KNN, LOF, AE, DeepSAD, ESAD, Classifier)
MMI.metadata_pkg.(MODELS,
    package_name="OutlierDetection.jl",
    package_uuid="262411bb-c475-4342-ba9e-03b8c0183ca6",
    package_url="https://github.com/davnn/OutlierDetection.jl",
    is_pure_julia=true,
    package_license="MIT",
    is_wrapper=false)
