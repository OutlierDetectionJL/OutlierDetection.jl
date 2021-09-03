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
OD.fit(_::MinimalUnsupervised, X::OD.Data)::OD.Fit = MinimalModel(), score(X)
OD.fit(_::MinimalSupervised, X::OD.Data, y::OD.Labels)::OD.Fit = MinimalModel(), score(X)
OD.transform(_::Union{MinimalSupervised, MinimalUnsupervised}, model::MinimalModel, X::OD.Data)::OD.Scores = score(X)

include("tests.jl")
