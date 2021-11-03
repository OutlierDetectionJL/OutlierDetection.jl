# Interface

Here we define the abstract supertypes that all outlier detectors share as well as useful datatypes use throughout `OutlierDetectionJL` and the [`fit`](@ref) and [`transform`](@ref) methods, that have to be implemented for each detector.

## Detectors

### `Detector`

```@docs
OutlierDetectionInterface.Detector
```

### `SupervisedDetector`

```@docs
OutlierDetectionInterface.SupervisedDetector
```

### `UnsupervisedDetector`

```@docs
OutlierDetectionInterface.UnsupervisedDetector
```

## Data types

### `DetectorModel`

```@docs
OutlierDetectionInterface.DetectorModel
```

### `Scores`

```@docs
OutlierDetectionInterface.Scores
```

### `Data`

```@docs
OutlierDetectionInterface.Data
```

### `Label`

```@docs
OutlierDetectionInterface.Labels
```

### `Fit`

```@docs
OutlierDetectionInterface.Fit
```

## Functions

### `fit`

```@docs
OutlierDetectionInterface.fit
```

### `transform`

```@docs
OutlierDetectionInterface.transform
```

## Macros

### `@detector`

```@docs
OutlierDetectionInterface.@detector
```

### `@default_frontend`

```@docs
OutlierDetectionInterface.@default_frontend
```

### `@default_metadata`

```@docs
OutlierDetectionInterface.@default_metadata
```
