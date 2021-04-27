# Guide

This guide should provide you the necessary knowledge to work with `OutlierDetection.jl` and understand the concepts behind the library design.

!!! note
    Outlier detection is predominantly an *unsupervised learning task*, transforming each data point to an *outlier score* quantifying the level of "outlierness". This very general form of output retains all the information provided by a specific algorithm.

## Key Concepts

The key design choice of OutlierDetection.jl is promoting the usage of *outlier scores*, not labels. The main data type, a [`Detector`](@ref), has to implement two methods: [`fit`](@ref) and [`score`](@ref).

- [`Detector`](@ref): A `struct` defining the hyperparameters for an outlier detection algorithm, just like an estimator in [scikit-learn](https://scikit-learn.org/stable/developers/develop.html).
- [`fit`](@ref): Learn a [`Model`](@ref) for a specific detector from input data `X` and labels `y` (if supervised), for example the weights of a neural network.
- [`score`](@ref): Using a detector and a learned model, transform unseen data into outlier scores.

Transforming the outlier scores to labels is seen as the last step of an outlier detection task. A [`Classifier`](@ref) simply turns scores into labels, typically with two classes describing inliers `(1)` and outliers `(-1)`. A classifier has to implement a single method: `detect`.

- [`detect`](@ref): Transform outlier scores to inlier and outlier classes.

A convention used in OutlierDetection.jl is that *higher scores always imply higher outlierness*.

!!! note
    A peculiarity of working with outlier scores is the distinction between *train scores* and *test scores*. Train scores result from fitting a detector ([`fit`](@ref)), and test scores result from using predicting unseen data ([`score`](@ref)). Classifying an instance as an inlier or outlier always requires a comparison to the train scores.

Let's see how the data looks like in a typical outlier detection task. We use the following naming conventions for the data we are working with:

```julia
Data::AbstractArray{<:Real}
Scores::AbstractVector{<:Real}
Labels::AbstractVector{<:Integer}
Result::Tuple{Scores, Scores}
```

Because train scores are essential in classification, we often work with tuples of training and test scores and call such a tuple a `Result`. One last previously unmentioned structure is the `Fit` result, a `struct` that bundles the learned model and training scores. Let's now looks how the methods defined by OutlierDetection.jl transform the mentioned data structures.

```julia
fit(::UnsupervisedDetector, ::Data)::Fit
fit(::SupervisedDetector, ::Data, ::Labels)::Fit
score(::Detector, ::Fit, ::Data)::Result
detect(::Classifier, ::Result...)::Labels
```

One last thing to not is that there are many convenience data transformations implemented. You can use any [Tables.jl](https://github.com/JuliaData/Tables.jl) compatible data source and the framework will make sure that the detectors receive the data in the suitable form. Also, note that `detect` can work with arbitrarily many results, which is very convenient if you want to combine the results of different detectors.

!!! warning
    If you are using native Julia arrays `AbstractArray{<:Real}` as input data, we expect the data to be formatted using the columns-as-observations convention for improved performance with Julia's column-major data. Every other input data will be transposed and converted to an array implicitly.

## Interoperation with MLJ

One of the exciting features of OutlierDetection.jl is it's interoperability with the rest of Julia's machine learning ecosystem. You might want to preprocess your data, cluster it, detect outliers, classify, and so forth.

In MLJ, we bind data to a detector using a [`machine`](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/). This data binding allows greater flexibility in later usage; for example, if you use cross-validation for evaluation, the data is split automatically behind the scenes. A machine further enables us to *implicitly* pass data and learned models to the `fit` and `score` methods.

From a usage perspective, the main differences are:

- A `Detector` is bound to data, either through `machine(::UnsupervisedDetector, X)`, or `machine(::SupervisedDetector, X, y)`.
- `fit(::Detector, X, [y])` becomes `fit!(machine)`
- `score(::Detector, ::Fit, X)` becomes `transform(machine)`
- `detect(::Classifier, ::Results...)` becomes `transform(machine(::Classifier), ::Results...)`

Take a look at [Using MLJ](../../documentation/using-mlj/) to learn more.
