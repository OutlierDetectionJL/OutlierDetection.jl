function MMI.fit(detector::UnsupervisedDetector, verbosity::Int, X)
    model, scores = fit(detector, X)
    model, nothing, scores
end

function MMI.fit(detector::SupervisedDetector, verbosity::Int, X, y)
    model, scores = fit(detector, X, y)
    model, nothing, scores
end

function MMI.transform(detector::Detector, model::DetectorModel, X)
    transform(detector, model, X)
end

function MMI.transform(clf::Classifier, scores::Tuple{Scores, Scores}...)
    transform(clf, scores...)
end

MMI.input_scitype(::Type{<:Detector}) = MMI.Table(MMI.Continuous)
MMI.output_scitype(::Type{<:Detector}) = AbstractVector{<:MMI.Continuous}

# for fit (supervised):
MMI.reformat(::SupervisedDetector, X, y) = (MMI.matrix(X, transpose=true), y)
MMI.reformat(::SupervisedDetector, X, y, w) = (MMI.matrix(X, transpose=true), y, w)
MMI.selectrows(::SupervisedDetector, I, Xmatrix, y) = (view(Xmatrix, :, I), view(y, I))
MMI.selectrows(::SupervisedDetector, I, Xmatrix, y, w) = (view(Xmatrix, :, I), view(y, I), view(w, I))

# for fit (unsupervised) (allow supervised call syntax with unused y and w):
MMI.reformat(::UnsupervisedDetector, X) = (MMI.matrix(X, transpose=true),)
MMI.reformat(::UnsupervisedDetector, X, _) = (MMI.matrix(X, transpose=true),)
MMI.reformat(::UnsupervisedDetector, X, _, _) = (MMI.matrix(X, transpose=true),)
MMI.selectrows(::UnsupervisedDetector, I, Xmatrix) = (view(Xmatrix, :, I),)
MMI.selectrows(::UnsupervisedDetector, I, Xmatrix, _) = (view(Xmatrix, :, I),)
MMI.selectrows(::UnsupervisedDetector, I, Xmatrix, _, _) = (view(Xmatrix, :, I),)

# for transform/predict:
MMI.reformat(::Detector, X) = (MMI.matrix(X, transpose=true),)
MMI.selectrows(::Detector, I, Xmatrix) = view(Xmatrix, I)

MODELS = (ABOD, COF, DNN, KNN, LOF, AE, DeepSAD, ESAD)
MMI.metadata_pkg.(MODELS,
    package_name="OutlierDetection.jl",
    package_uuid="262411bb-c475-4342-ba9e-03b8c0183ca6",
    package_url="https://github.com/davnn/OutlierDetection.jl",
    is_pure_julia=true,
    package_license="MIT",
    is_wrapper=false)