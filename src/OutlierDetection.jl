module OutlierDetection
    using OutlierDetectionInterface
    const OD = OutlierDetectionInterface

    import MLJBase
    const MLJ = MLJBase

    # re-export from OutlierDetectionInterface
    export CLASS_NORMAL, CLASS_OUTLIER, DEFAULT_THRESHOLD

    export augmented_transform,
           scale_minmax,
           scale_unify,
           combine_mean,
           combine_median,
           combine_max,
           classify_percentile,
           to_categorical,
           to_univariate_finite,
           from_categorical,
           from_univariate_finite

    export ProbabilisticDetector,
           DeterministicDetector,
           CompositeDetector,
           ProbabilisticTransformer,
           DeterministicTransformer,
           ScoreTransformer

    # utilities
    include("normalization.jl")
    include("classification.jl")
    include("combination.jl")

    # extension
    include("mlj_helpers.jl")
    include("mlj_transformers.jl")
    include("mlj_wrappers.jl")
end
