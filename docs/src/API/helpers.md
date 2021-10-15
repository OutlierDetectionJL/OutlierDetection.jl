# Helpers

`OutlierDetection.jl` provides many useful helper functions to work with outlier detection models. The goal of these helpers is to normalize, combine and classify raw outlier scores and turn them into evaluateable probabilities or classes.

## Wrappers

Wrappers take one or more detectors and transform the (combined) raw scores to probabilities ([`OutlierDetection.ProbabilisticDetector`](@ref)) or classes ([`OutlierDetection.DeterministicDetector`](@ref)). Using wrappers, you can easily evaluate outlier detection models with MLJ.

### CompositeDetector

```@docs
OutlierDetection.CompositeDetector
```

### ProbabilisticDetector

```@docs
OutlierDetection.ProbabilisticDetector
```

### DeterministicDetector

```@docs
OutlierDetection.DeterministicDetector
```

## Transformers

A transformer simply takes the training and test scores of one or more detectors, normalizes the scores and transforms them into some new representation.

### ScoreTransformer

```@docs
OutlierDetection.ScoreTransformer
```

## Normalization

These functions may be used as an input for the `normalize` keyword argument present in wrappers and transformers, they transform `train_scores` and `test_scores`.

```@docs
OutlierDetection.scale_minmax
```

```@docs
OutlierDetection.scale_unify
```

## Combination

These functions may be used as an input for the `combine` keyword argument present in wrappers and transformers.

```@docs
OutlierDetection.combine_mean
```

```@docs
OutlierDetection.combine_median
```

```@docs
OutlierDetection.combine_max
```

## Classification

These functions may be used as an input for the `classify` keyword argument present in wrappers and transformers.

```@docs
OutlierDetection.classify_percentile
```

## Output helpers

```@docs
OutlierDetection.to_univariate_finite
```

```@docs
OutlierDetection.to_categorical
```

```@docs
OutlierDetection.from_univariate_finite
```

```@docs
OutlierDetection.from_categorical
```
