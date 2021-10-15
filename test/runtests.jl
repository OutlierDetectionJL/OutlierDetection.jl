using MLJBase
using OutlierDetection
using Statistics
using Test

import OutlierDetectionInterface
const OD = OutlierDetectionInterface

struct MinimalDetectorModel <: OD.DetectorModel end
struct MinimalUnsupervisedDetector <: OD.UnsupervisedDetector end
struct MinimalSupervisedDetector <: OD.SupervisedDetector end

score(X) = dropdims(mean(X, dims=1), dims=1)
OD.fit(::MinimalUnsupervisedDetector, X::OD.Data; verbosity)::OD.Fit = MinimalDetectorModel(), score(X)
OD.fit(::MinimalSupervisedDetector, X::OD.Data, y::OD.Labels; verbosity)::OD.Fit = MinimalDetectorModel(), score(X)
OD.transform(::Union{MinimalSupervisedDetector, MinimalUnsupervisedDetector},
             model::MinimalDetectorModel, X::OD.Data)::OD.Scores = score(X)

include("tests.jl")
