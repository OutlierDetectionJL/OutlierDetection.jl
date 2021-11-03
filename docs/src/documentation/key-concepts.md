# Key Concepts

This guide should provide you the necessary knowledge to work with `OutlierDetection.jl` and understand the concepts behind the library design.

!!! note
    Outlier detection is predominantly an *unsupervised learning task*, transforming each data point to an *outlier score* quantifying the level of "outlierness". This very general form of output retains all the information provided by a specific algorithm.

The key design choice of OutlierDetection.jl is promoting the usage of *outlier scores*, not labels. The main data type, a [`Detector`](@ref), has to implement two methods: [`fit`](@ref) and [`transform`](@ref).

- [`Detector`](@ref): A `struct` defining the hyperparameters for an outlier detection algorithm, just like an estimator in [scikit-learn](https://scikit-learn.org/stable/developers/develop.html) or a model in [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/glossary/#model-(object-of-abstract-type-Model)). A detector actually *is* an `MLJModelInterface.Model` (subtype).
- [`fit`](@ref): Learn a [`DetectorModel`](@ref) for a specific detector from input data `X` and labels `y` (if supervised), for example the weights of a neural network.
- [`transform`](@ref): Using a detector and a learned model, transform unseen data into outlier scores.

Transforming the outlier scores to classes is seen as the last step of an outlier detection task. A [Wrapper](../../API/helpers#Wrappers) or [Transformer](../../API/helpers#Transformers) turns scores into probabilities or labels, typically with two classes describing inliers `"normal"` and outliers `"outlier"`. 

A convention used in OutlierDetection.jl is that *higher scores imply higher outlierness*.

!!! note
    A peculiarity of working with outlier scores is the distinction between *train scores* and *test scores*. Train scores result from fitting a detector ([`fit`](@ref)), and test scores result from predicting unseen data ([`transform`](@ref)). Classifying an instance as an inlier or outlier always requires a comparison to the train scores.

Let's see how the data types look like in a typical outlier detection task. We use the following naming conventions for the data we are working with: 

- the input data [`OutlierDetectionInterface.Data`](@ref)
- the raw scores [`OutlierDetectionInterface.Scores`](@ref)
- the labels [`OutlierDetectionInterface.Labels`](@ref)

One last unmentioned structure is the [`Fit`](@ref) result, a `struct` that bundles the learned model and training scores. Let's now looks how the methods defined by OutlierDetection.jl transform the mentioned data structures.

```julia
fit(::UnsupervisedDetector, ::Data; verbosity::Integer)::Fit
fit(::SupervisedDetector, ::Data, ::Labels; verbosity::Integer)::Fit
transform(::Detector, ::Fit, ::Data)::Scores
```

A new outlier detection algorithm can be implemented in *OutlierDetection.jl* easily by implementing above [`fit`](@ref) and [`transform`](@ref) methods.

!!! warning
    We expect the data to be formatted using the columns-as-observations convention for improved performance with Julia's column-major data.

## Integration with MLJ

One of the exciting features of *OutlierDetection.jl* is it's interoperability with the rest of Julia's machine learning ecosystem. You might want to preprocess your data, cluster it, detect outliers, classify, and so forth.

*OutlierDetection.jl* defines an interface for MLJ such the implemented `OutlierDetection.jl` detectors can be used directly with MLJ.

- A `Detector` is bound to data, either through `machine(::UnsupervisedDetector, X)`, or `machine(::SupervisedDetector, X, y)`.
- `fit(::Detector, X, [y]; verbosity)` becomes `fit!(machine)`, which calls [`fit`](@ref) under the hood
- `transform(::Detector, ::Fit, X)` becomes `transform(machine)`, which calls [`transform`](@ref) under the hood

Additionally, *OutlierDetection.jl* defines a data front-end for MLJ, which ensures that [`fit`](@ref) and [`transform`](@ref) are always called with Julia arrays in column-major format, even though `machine(::Detector, X, y)` also accepts data from any [Tables.jl](https://github.com/JuliaData/Tables.jl)-compatible data source.

Take a look at our [Simple Usage](../../documentation/simple-usage/) to learn more.
