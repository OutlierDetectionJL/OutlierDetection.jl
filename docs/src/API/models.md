# Models

All models, both supervised and unsupervised, define a [`Detector`](@ref), which is just a mutable collection of hyperparameters. Each detector implements a [`fit`](@ref) and [`transform`](@ref) method, where *fitting refers to learning a model from training data* and *transform refers to using a learned model to calculate outlier scores of test data*.

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
