module OutlierDetection
    using NearestNeighbors
    using MacroTools
    using Distances
    using MLJModelInterface
    using Requires:@require
    const MMI = MLJModelInterface
    const NN = NearestNeighbors
    const DI = Distances
    const CLASS_NORMAL = 1
    const CLASS_OUTLIER = -1
    const CLASS_UNKNOWN = 0

    # base
    export Detector,
           UnsupervisedDetector,
           SupervisedDetector,
           Model,
           Label,
           Data,
           Fit,
           fit,
           Score,
           score

    # models
    export DNN,
           DNNModel,
           KNN,
           KNNModel,
           LDF,
           LDFModel,
           LOF,
           LOFModel,
           COF,
           COFModel,
           ABOD,
           ABODModel,
           AE,
           AEModel,
           DeepSAD,
           DeepSADModel,
           ESAD,
           ESADModel

    export PyABOD,
           PyCBLOF,
           PyCOF,
           PyCOPOD,
           PyHBOS,
           PyIForest,
           PyKNN,
           PyLMDD,
           PyLODA,
           PyLOF,
           PyLOCI,
           PyMCD,
           PyOCSVM,
           PyPCA,
           PyROD,
           PySOD,
           PySOS,
           PyModel

    # evaluation
    export classify,
           normalize,
           unify,
           combine_mean,
           combine_median,
           combine_max,
           Scores,
           Labels

    # basics
    include("base.jl")

    # utilities
    include("utils/normalization.jl")
    include("utils/classification.jl")
    include("utils/combination.jl")
    include("utils/neighbors.jl")
    include("utils/neural.jl")

    # macros
    include("macros.jl")

    # detectors
    include("models/abod.jl")
    include("models/ae.jl")
    include("models/cof.jl")
    include("models/deepsad.jl")
    include("models/dnn.jl") 
    include("models/esad.jl")
    include("models/knn.jl")
    include("models/ldf.jl")
    include("models/lof.jl")
    include("pymodels/utils.jl")
    include("pymodels/detectors.jl")

    # examples
    include("examples/examples.jl")

    # extension
    include("extension/mlj.jl")

#     function __init__()
#        @require MLJ="add582a8-e3ab-11e8-2d5e-e98b27df1bc7" begin
#            include("extension/mlj.jl")
#            include("extension/mlj_extra.jl")
#        end
#     end
end
