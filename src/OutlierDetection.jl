module OutlierDetection
    using OutlierDetectionInterface

    import MLJModelInterface
    const MMI = MLJModelInterface

    import MLJBase
    const MLJ = MLJBase

    # re-export from OutlierDetectionInterface
    export to_categorical, to_univariate_finite, CLASS_NORMAL, CLASS_OUTLIER, DEFAULT_THRESHOLD

    export Score,
           Class,
           scale_minmax,
           scale_unify,
           combine_mean,
           combine_median,
           combine_max,
           classify_percentile,
           ProbabilisticDetector,
           DeterministicDetector

    # utilities
    include("normalization.jl")
    include("classification.jl")
    include("combination.jl")

    # extension
    include("mlj_wrappers.jl")
end
