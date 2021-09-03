module OutlierDetection
    using OutlierDetectionInterface

    import MLJBase
    const MLJ = MLJBase

    # re-export from OutlierDetectionInterface
    export CLASS_NORMAL, CLASS_OUTLIER, DEFAULT_THRESHOLD

    export ScoreTransformer,
           ClassTransformer,
           scale_minmax,
           scale_unify,
           combine_mean,
           combine_median,
           combine_max,
           classify_percentile,
           to_categorical,
           to_univariate_finite,
           probabilistic,
           deterministic

    # utilities
    include("normalization.jl")
    include("classification.jl")
    include("combination.jl")

    # extension
    include("mlj_helpers.jl")
    include("mlj_transformers.jl")
    include("mlj_wrappers.jl")
end
