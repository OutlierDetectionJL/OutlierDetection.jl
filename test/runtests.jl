using OutlierDetection
using OutlierDetection.Examples
using Random:MersenneTwister
using MLJBase
using Test

include("classifier.jl")
include("detector.jl")

# Number of dimensions for training
const trainDim = 100
encoder, decoder = MLPAutoEncoder(trainDim, 5, [50,20]; bias=false);

# julia
test_detector(DNN(d=0.1))
test_detector(DNN(d=0.1, parallel=true))
test_detector(KNN())
test_detector(KNN(parallel=true, reduction=:mean, algorithm=:balltree))
test_detector(KNN(parallel=true, reduction=:median))
test_detector(LOF())
test_detector(LOF(parallel=true))
test_detector(COF())
test_detector(COF(parallel=true))
test_detector(ABOD())
test_detector(ABOD(parallel=true, enhanced=true))
test_detector(AE(encoder=encoder, decoder=decoder))
test_detector(DeepSAD(encoder=encoder, decoder=decoder))
test_detector(ESAD(encoder=encoder, decoder=decoder))

# python
test_detector(PyABOD())
test_detector(PyCBLOF(random_state=0))
test_detector(PyCOF())
test_detector(PyCOPOD())
test_detector(PyHBOS())
test_detector(PyIForest(random_state=0))
test_detector(PyKNN())
test_detector(PyLMDD(random_state=0))
test_detector(PyLODA())
test_detector(PyLOF())
test_detector(PyLOCI())
test_detector(PyMCD(random_state=0))
test_detector(PyOCSVM())
test_detector(PyPCA(random_state=0))
# The following detector is only supported in the latest version of PyOD and currently fails
# test_detector(PyROD())
test_detector(PySOD())
test_detector(PySOS())
