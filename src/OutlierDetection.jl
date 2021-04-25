module OutlierDetection
    import MLJModelInterface
    import NearestNeighbors
    import Distances
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
           Scores,
           Data,
           Labels,
           Fit,
           Result,
           fit,
           score,
           detect

    # models
    export DNN,
           DNNModel,
           KNN,
           KNNModel,
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
    export roc,
           roc_auc,
           classify,
           normalize,
           unify,
           combine,
           Classifier

    # basic types
    include("base.jl")

    # utilities
    include("utils/neighbors.jl")
    include("utils/neural.jl")

    # detectors
    include("models/abod.jl")
    include("models/ae.jl")
    include("models/cof.jl")
    include("models/deepsad.jl")
    include("models/dnn.jl") 
    include("models/esad.jl")
    include("models/knn.jl")
    include("models/lof.jl")
    include("pymodels/utils.jl")
    include("pymodels/detectors.jl")

    # evaluation
    include("evaluate/evaluate.jl")
    include("evaluate/classifier.jl")
    include("evaluate/roc.jl")

    # examples
    include("examples/examples.jl")

    # extension
    include("extension/mlj.jl")
end
