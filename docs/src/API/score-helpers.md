# Score Helpers

`OutlierDetection.jl` provides many useful helper functions to work with outlier scores. The goal of these helpers is to normalize, combine and classify raw outlier scores. The main design philosophy behind all of these functions is that they transform a tuple of train/test scores into some different train/test tuple representation, e.g. train/test classes.

## Transformers

In order to normalize scores or classify them, both the training and testing scores are necessary. We thus provide a helper function called [`augmented_transform`](@ref) that returns a tuple of training and test scores. Transformers can make use of one or more such train/test tuples to convert them into normalized scores, probabilities or classes.

### `augmented_transform`

```@docs
OutlierDetection.augmented_transform
```

### `ScoreTransformer`

```@docs
OutlierDetection.ScoreTransformer
```

### `ProbabilisticTransformer`

```@docs
OutlierDetection.ProbabilisticTransformer
```

### `DeterministicTransformer`

```@docs
OutlierDetection.DeterministicTransformer
```

## Wrappers

Wrappers take one or more detectors and transform the (combined) raw scores to probabilities ([`ProbabilisticDetector`](@ref)) or classes ([`DeterministicDetector`](@ref)). Using wrappers, you can easily evaluate outlier detection models with MLJ.

### `CompositeDetector`

```@docs
OutlierDetection.CompositeDetector
```

### `ProbabilisticDetector`

```@docs
OutlierDetection.ProbabilisticDetector
```

### `DeterministicDetector`

```@docs
OutlierDetection.DeterministicDetector
```

## Normalization

These functions may be used as an input for the `normalize` keyword argument present in wrappers and transformers, they transform a tuple of train/test scores into a tuple of normalized train/test scores.

### `scale_minmax`

```@docs
OutlierDetection.scale_minmax
```

### `scale_unify`

```@docs
OutlierDetection.scale_unify
```

## Combination

These functions may be used as an input for the `combine` keyword argument present in wrappers and transformers. The input for the combine functions are one or more train/test score tuples or alternatively a matrix where the first columns represents train scores and the second column test scores.

### `combine_mean`

```@docs
OutlierDetection.combine_mean
```

### `combine_median`

```@docs
OutlierDetection.combine_median
```

### `combine_max`

```@docs
OutlierDetection.combine_max
```

## Classification

These functions may be used as an input for the `classify` keyword argument present in wrappers and transformers, they transform a tuple of train/test scores into a tuple of train/test classes.

### `classify_quantile`

```@docs
OutlierDetection.classify_quantile
```

## Output helpers

### `to_univariate_finite`

```@docs
OutlierDetection.to_univariate_finite
```

### `to_categorical`

```@docs
OutlierDetection.to_categorical
```

### `from_univariate_finite`

```@docs
OutlierDetection.from_univariate_finite
```

### `from_categorical`

```@docs
OutlierDetection.from_categorical
```

## Label helpers

### `normal_fraction`

```@docs
OutlierDetection.normal_fraction
```

### `outlier_fraction`

```@docs
OutlierDetection.outlier_fraction
```

### `normal_count`

```@docs
OutlierDetection.normal_count
```

### `outlier_count`

```@docs
OutlierDetection.outlier_count
```
