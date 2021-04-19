using OutlierDetection
using OutlierDetection.Examples
using Test

include("classifier.jl")
include("detector.jl")

# Number of dimensions for training
const trainDim = 100
encoder, decoder = MLPAutoEncoder(trainDim, 5, [50,20]; bias=false);

test_detector(DNN(d=0.1))
test_detector(DNN(d=0.1, parallel=true))
test_detector(KNN())
test_detector(KNN(parallel=true))
test_detector(LOF())
test_detector(LOF(parallel=true))
test_detector(COF())
test_detector(COF(parallel=true))
test_detector(ABOD())
test_detector(ABOD(parallel=true))
test_detector(AE(encoder=encoder, decoder=decoder))
test_detector(DeepSAD(encoder=encoder, decoder=decoder))
test_detector(ESAD(encoder=encoder, decoder=decoder))
