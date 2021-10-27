SCORE_PARAMS = """  normalize::Function
A function to reduce a matrix, where each row represents an instance and each column a score of specific detector, to a
vector of scores for each instance. See [`scale_minmax`](@ref) for a specific implementation.

    combine::Function
A function to reduce a matrix, where each row represents an instance and each column represents the score of specific
detector, to a vector of scores for each instance. See [`combine_mean`](@ref) for a specific implementation."""

"""
    ScoreTransformer(combine = combine,           
                     normalize = normalize)

Transform the results of a single or multiple outlier detection models to combined and normalized scores.

Parameters
----------
$SCORE_PARAMS
"""
@detector mutable struct ScoreTransformer <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
end

"""
    ProbabilisticTransformer(combine = combine,           
                             normalize = normalize)

Transform the results of a single or multiple outlier detection models to combined univariate finite distributions.

Parameters
----------
$SCORE_PARAMS
"""
@detector mutable struct ProbabilisticTransformer <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
end

"""
    DeterministicTransformer(combine = combine,           
                             normalize = normalize,
                             classify = classify_percentile(DEFAULT_THRESHOLD))

Transform the results of a single or multiple outlier detection models to combined categorical values.

Parameters
----------
$SCORE_PARAMS
"""
default_percentile_threshold = classify_percentile(DEFAULT_THRESHOLD)
@detector mutable struct DeterministicTransformer <: MLJ.Static
    normalize::Function = scale_minmax
    combine::Function = combine_mean
    classify::Function = default_percentile_threshold
end

const StaticTransformer = Union{
    ScoreTransformer,
    DeterministicTransformer,
    ProbabilisticTransformer
}

# returns the augmented train/test scores
function MLJ.transform(ev::StaticTransformer, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    _, scores_test = to_scores(ev.normalize, ev.combine, scores...)
    scores_test
end

function MLJ.predict(ev::ProbabilisticTransformer, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    _, scores_test = to_scores(ev.normalize, ev.combine, scores...)
    to_univariate_finite(scores_test)
end

function MLJ.predict(ev::DeterministicTransformer, _, scores::Tuple{Scores, Scores}...) # _ because there is no fitresult
    scores_train, scores_test = to_scores(ev.normalize, ev.combine, scores...)
    _, classes_test = ev.classify(scores_train, scores_test)
    to_categorical(classes_test)
end

function to_scores(normalize::Function, combine::Function, scores::Tuple{Scores, Scores}...)::Tuple{Scores, Scores}
    # make sure that we have at least one tuple of train and test scores
    n_scores = length(scores)
    @assert n_scores > 0

    # normalize all scores (might be identity function)
    scores = map(score_tuple -> normalize(score_tuple...), scores)

    # [(train1, test1), (train2, test2)] -> matrix of scores for train [train1 train2]' and test [test1 test2]'
    # Note: The matrices contain one observation per row!
    reduce_cat = idx -> reduce(hcat, getfield.(scores, idx))
    scores_train, scores_test = combine(reduce_cat(1)), combine(reduce_cat(2))
    scores_train, scores_test
end

to_fitresult(mach::MLJ.Machine{<:OD.Detector})::Fit = (mach.fitresult, mach.report.scores)

# this includes all composites defined in mlj_wrappers.jl
const DetectorComposites = Union{
    MLJ.Machine{<:MLJ.SupervisedDetectorComposite},
    MLJ.Machine{<:MLJ.UnsupervisedDetectorComposite},
    MLJ.Machine{<:MLJ.ProbabilisticUnsupervisedDetectorComposite},
    MLJ.Machine{<:MLJ.DeterministicUnsupervisedDetectorComposite},
    MLJ.Machine{<:MLJ.ProbabilisticSupervisedDetectorComposite},
    MLJ.Machine{<:MLJ.DeterministicSupervisedDetectorComposite}
}

function check_mach(mach)
    # catch deserialized machine with no data:
    isempty(mach.args) && MLJ._err_serialized(augmented_transform)
    # catch not-yet-trained machine:
    mach.state > 0 || error("$mach has not been trained.")
end

function _augmented_transform(detector::Detector, fitresult::Fit, X)
    model, scores_train = fitresult
    scores_test = MLJ.transform(detector, model, X)
    return scores_train, scores_test
end

# 0. augmented_transform given rows:
function augmented_transform(mach::MLJ.Machine{<:OD.Detector}; rows=:)
    check_mach(mach)
    return _augmented_transform(mach.model, to_fitresult(mach), selectrows(mach.model, rows, mach.data[1])...)
end

function augmented_transform(mach::DetectorComposites; rows=:)
    check_mach(mach)
    scores_train = mach.report.scores
    scores_test = mach.fitresult.transform(selectrows(mach.model, rows, mach.data[1])...)
    return scores_train, scores_test
end

# 1. augmented_transform on machines, given *concrete* data:
function augmented_transform(mach::MLJ.Machine{<:OD.Detector}, X)
    check_mach(mach)
    return _augmented_transform(mach.model, to_fitresult(mach), reformat(mach.model, X)...)
end

function augmented_transform(mach::DetectorComposites, X)
    check_mach(mach)
    scores_train = mach.report.scores
    scores_test = mach.fitresult.transform(X)
    return scores_train, scores_test
end

# 2. operations on machines, given *dynamic* data (nodes):
function augmented_transform(mach::MLJ.Machine{<:OD.Detector}, X::MLJ.AbstractNode)
    MLJ.node(augmented_transform, mach, X)
end
