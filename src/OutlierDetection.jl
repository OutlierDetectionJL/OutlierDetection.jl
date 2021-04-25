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

    # probabilistic models


    # distance-based models
    include("models/proximity/utils.jl")
    include("models/proximity/dnn.jl") 
    include("models/proximity/knn.jl")
    include("models/proximity/lof.jl")
    include("models/proximity/cof.jl")
    include("models/proximity/abod.jl")

    # kernel-based models
    

    # neural network models
    include("models/neural/utils.jl")
    include("models/neural/ae.jl")
    include("models/neural/deepsad.jl")
    include("models/neural/esad.jl")

    # python models
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
