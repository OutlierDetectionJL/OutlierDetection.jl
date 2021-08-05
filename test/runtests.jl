using MLJ
using OutlierDetection
using OutlierDetection.Examples
using CategoricalArrays:unwrap
using Random:MersenneTwister
using Test

include("normalization.jl")
include("detector.jl")

# specify test parameters
const rng, fraction_train, n_samples, dim = MersenneTwister(0), 0.5, 100, 10
const n_train = Int(n_samples * fraction_train)
const n_test = n_samples - n_train

# specify test data
const X_raw = rand(rng, dim, n_samples)
const X_mat = collect(X_raw')
const X_df = table(X_mat)
const y = rand(rng, (-1, 1), n_samples)
const train, test  = partition(eachindex(y), fraction_train, rng=rng);

# Number of dimensions for training
const encoder, decoder = MLPAutoEncoder(dim, 5, [50,20]; bias=false);

# Test the Julia detectors
test_detector(ABOD())
test_detector(ABOD(parallel=true, enhanced=true))
test_detector(AE(encoder=encoder, decoder=decoder))
test_detector(COF())
test_detector(COF(parallel=true))
test_detector(DeepSAD(encoder=encoder, decoder=decoder))
test_detector(DNN(d=1))
test_detector(DNN(d=1, parallel=true))
test_detector(ESAD(encoder=encoder, decoder=decoder))
test_detector(KNN())
test_detector(KNN(parallel=true, reduction=:mean, algorithm=:balltree))
test_detector(KNN(parallel=true, reduction=:median))
test_detector(LOF())
test_detector(LOF(parallel=true))

# Test the Python detectors
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
