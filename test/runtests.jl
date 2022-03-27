using MLJBase
using OutlierDetection
using Statistics
using Test

import OutlierDetectionInterface
const OD = OutlierDetectionInterface
const MMI = OutlierDetectionInterface.MLJModelInterface

# make sure the OD detector interface works
struct MinimalDetectorModel <: OD.DetectorModel end
struct ODUnsupervisedDetector <: OD.UnsupervisedDetector end
struct ODSupervisedDetector <: OD.SupervisedDetector end

# make sure the MMI detector interface works
struct MMIDetectorModel end
struct MMIUnsupervisedDetector <: MMI.UnsupervisedDetector end
struct MMISupervisedDetector <: MMI.SupervisedDetector end

score(X) = dropdims(mean(X, dims=1), dims=1)
table_score(X) = score(MLJBase.matrix(X, transpose=true))
OD.fit(::ODUnsupervisedDetector, X::OD.Data; verbosity)::OD.Fit = MinimalDetectorModel(), score(X)
OD.fit(::ODSupervisedDetector, X::OD.Data, y::OD.Labels; verbosity)::OD.Fit = MinimalDetectorModel(), score(X)
OD.transform(::Union{ODSupervisedDetector,ODUnsupervisedDetector},
    model::MinimalDetectorModel, X::OD.Data)::OD.Scores = score(X)

MMI.fit(::MMIUnsupervisedDetector, verbosity, X) = MMIDetectorModel(), nothing, (scores=table_score(X),)
MMI.fit(::MMISupervisedDetector, verbosity, X, y) = MMIDetectorModel(), nothing, (scores=table_score(X),)
MMI.transform(::Union{MMIUnsupervisedDetector,MMISupervisedDetector}, model::MMIDetectorModel, X) = table_score(X)

OD.@default_frontend(ODUnsupervisedDetector)
OD.@default_frontend(ODSupervisedDetector)

include("tests.jl")
