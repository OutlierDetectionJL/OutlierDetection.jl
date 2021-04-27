# Detectors

A [`Detector`](@ref) is just a mutable collection of hyperparameters. Each detector implements a [`fit`](@ref) and [`score`](@ref) method, where *fit* refers to learning a model from training data and *score* refers to using a learned model to calculate outlier scores of new data. Detectors typically do not classify samples; that's why a classifier is used to convert the scores into binary labels, see [`Binarize`](@ref) for example.

## ABOD

```@docs
ABOD
```

## AE

```@docs
AE
```

## COF

```@docs
COF
```

## DeepSAD

```@docs
DeepSAD
```

## DNN

```@docs
DNN
```

## ESAD

```@docs
ESAD
```

## KNN

```@docs
KNN
```

## LOF

```@docs
LOF
```
