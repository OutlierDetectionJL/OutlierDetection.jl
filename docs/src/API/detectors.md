# Detectors

A [`Detector`](@ref) is just a collection of hyperparameters. Each detector implements a [`fit`](@ref) and [`transform`](@ref) method, where *fit* refers to learning a model from training data and *transform* refers to using a learned model to calculate outlier scores of new data. Detectors typically do not classify samples into inliers and outliers; that's a [`DeterministicDetector`](@ref) wrapper is used to convert the raw scores into binary labels.

## Neighbor-based

### `ABODDetector`

```@docs
OutlierDetectionNeighbors.ABODDetector
```

### `COFDetector`

```@docs
OutlierDetectionNeighbors.COFDetector
```

### `DNNDetector`

```@docs
OutlierDetectionNeighbors.DNNDetector
```

### `KNNDetector`

```@docs
OutlierDetectionNeighbors.KNNDetector
```

### `LOFDetector`

```@docs
OutlierDetectionNeighbors.LOFDetector
```

## Network-based

!!! warning
    The neural-network detectors are *experimental* and subject to change.

### `AEDetector`

```@docs
OutlierDetectionNetworks.AEDetector
```

### `DSADDetector`

```@docs
OutlierDetectionNetworks.DSADDetector
```

### `ESADDetector`

```@docs
OutlierDetectionNetworks.ESADDetector
```

## Python-based

Using [PyCall](https://github.com/JuliaPy/PyCall.jl), we can easily integrate existing python outlier detection algorithms. Currently, almost every [PyOD](https://github.com/yzhao062/pyod) algorithm is integrated and can thus be easily used directly from Julia.

### `ABODDetector`

```@docs
OutlierDetectionPython.ABODDetector
```

### `CBLOFDetector`

```@docs
OutlierDetectionPython.CBLOFDetector
```

### `CDDetector`

```@docs
OutlierDetectionPython.CDDetector
```

### `COFDetector`

```@docs
OutlierDetectionPython.COFDetector
```

### `COPODDetector`

```@docs
OutlierDetectionPython.COPODDetector
```

### `ECODDetector`

```@docs
OutlierDetectionPython.ECODDetector
```

### `GMMDetector`

```@docs
OutlierDetectionPython.GMMDetector
```

### `HBOSDetector`

```@docs
OutlierDetectionPython.HBOSDetector
```

### `IForestDetector`

```@docs
OutlierDetectionPython.IForestDetector
```

### `INNEDetector`

```@docs
OutlierDetectionPython.INNEDetector
```

### `KDEDetector`

```@docs
OutlierDetectionPython.KDEDetector
```

### `KNNDetector`

```@docs
OutlierDetectionPython.KNNDetector
```

### `LMDDDetector`

```@docs
OutlierDetectionPython.LMDDDetector
```

### `LODADetector`

```@docs
OutlierDetectionPython.LODADetector
```

### `LOFDetector`

```@docs
OutlierDetectionPython.LOFDetector
```

### `LOCIDetector`

```@docs
OutlierDetectionPython.LOCIDetector
```

### `MCDDetector`

```@docs
OutlierDetectionPython.MCDDetector
```

### `OCSVMDetector`

```@docs
OutlierDetectionPython.OCSVMDetector
```

### `PCADetector`

```@docs
OutlierDetectionPython.PCADetector
```

### `RODDetector`

```@docs
OutlierDetectionPython.RODDetector
```

### `SODDetector`

```@docs
OutlierDetectionPython.SODDetector
```

### `SOSDetector`

```@docs
OutlierDetectionPython.SOSDetector
```
