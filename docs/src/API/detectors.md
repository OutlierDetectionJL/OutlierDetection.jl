# Detectors

All models, both supervised and unsupervised, define a [`Detector`](@ref), which is just a mutable collection of hyperparameters. Each detector implements a [`fit`](@ref) and [`score`](@ref) method, where *fit* refers to learning a model from training data and *score* refers to using a learned model to calculate outlier scores of new data. Detectors typically do not classify samples; that's why a classifier might be used to convert the scores into binary labels, see [`Binarize`](@ref) for example.

## Proximity Models

### ABOD

```@docs
ABOD
```

### COF

```@docs
COF
```

### DNN

```@docs
DNN
```

### KNN

```@docs
KNN
```

### LOF

```@docs
LOF
```

## Neural Models

### AE

```@docs
AE
```

### DeepSAD

```@docs
DeepSAD
```

### ESAD

```@docs
ESAD
```
