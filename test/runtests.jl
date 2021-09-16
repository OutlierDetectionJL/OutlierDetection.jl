using MLJBase
using OutlierDetection
using Statistics
using Test

import OutlierDetectionInterface
const OD = OutlierDetectionInterface

struct MinimalUnsupervised <: OD.UnsupervisedDetector end
struct MinimalSupervised <: OD.SupervisedDetector end
struct MinimalModel <: OD.DetectorModel end

score(X) = dropdims(mean(X, dims=1), dims=1)
OD.fit(::MinimalUnsupervised, X::OD.Data; verbosity)::OD.Fit = MinimalModel(), score(X)
OD.fit(::MinimalSupervised, X::OD.Data, y::OD.Labels; verbosity)::OD.Fit = MinimalModel(), score(X)
OD.transform(::Union{MinimalSupervised, MinimalUnsupervised}, model::MinimalModel, X::OD.Data)::OD.Scores = score(X)

include("tests.jl")
