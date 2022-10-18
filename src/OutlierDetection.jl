module OutlierDetection
    using OutlierDetectionInterface
    const OD = OutlierDetectionInterface

    import MLJBase
    const MLJ = MLJBase
    const MMI = OD.MLJModelInterface

    # re-export from OutlierDetectionInterface
    export CLASS_NORMAL, CLASS_OUTLIER, DEFAULT_THRESHOLD

    # combination.jl
    export combine_mean,
           combine_median,
           combine_max

    # normalization.jl
    export scale_minmax,
           scale_unify

    # classification.jl
    export classify_quantile

    # helpers.jl
    export normal_count,
           outlier_count,
           normal_fraction,
           outlier_fraction 

    # mlj_helpers.jl
    export augmented_transform,
           to_categorical,
           to_univariate_finite,
           from_categorical,
           from_univariate_finite

    # mlj_transformers.jl
    export ProbabilisticTransformer,
           DeterministicTransformer,
           ScoreTransformer

    # mlj_wrappers.jl
    export ProbabilisticDetector,
           DeterministicDetector,
           CompositeDetector

    # utilities
    include("normalization.jl")
    include("classification.jl")
    include("combination.jl")
    include("helpers.jl")

    # extension
    include("mlj_helpers.jl")
    include("mlj_transformers.jl")
    include("mlj_wrappers.jl")
end
