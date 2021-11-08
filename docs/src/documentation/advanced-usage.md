# Advanced Usage

The simple usage guide covered how you can use and optimize an existing outlier detection model, however, sometimes it is necessary to combine the results of multiple models or create entirely new models.

## Working with scores

An outlier detection model, whether supervised or unsupervised, typically assigns an *outlier score* to each datapoint. We further differentiate between outier scores achieved during *training* or *testing*. Because both train and test scores are essential for further score processing, e.g. converting the scores to classes, we provide an [`augmented_transform`](@ref) that returns a tuple of train and test scores.

```@example advanced
using MLJ, OutlierDetection
using OutlierDetectionData: ODDS

X, y = ODDS.load("annthyroid")
train, test = partition(eachindex(y), 0.5, shuffle=true, stratify=y, rng=0)
KNN = @iload KNNDetector pkg=OutlierDetectionNeighbors verbosity=0
knn = KNN()
```

Let's bind the detector to data and perform an `augmented_transform`.

```@example advanced
mach = machine(knn, X, y)
fit!(mach, rows=train)
scores = augmented_transform(mach, rows=test)
scores_train, scores_test = scores
```

We split the into 50% train and 50% test data, thus `scores_train` and `scores_test` should return an equal amount of scores.

```@example advanced
scores_train
```

```@example advanced
scores_test
```

OutlierDetection.jl provides many helper functions to work with scores, see [score helpers](/API/score-helpers). The fundamental datatype to work with scores is a tuple of train/test scores and all helper functions work with this datatype. An example for such a helper function is [`scale_minmax`](@ref), which scales the scores to lie between 0 and 1 using min-max scaling.

```@example advanced
last(scores |> scale_minmax)
```

Another exemplary helper function is [`classify_quantile`](@ref), which is used to transform scores to classes. We only display the test scores using the `last` element of the tuple.

```@example advanced
last(scores |> classify_quantile(0.9))
```

Sometimes it's also necessary to combine scores from multiple detectors, which can, for example, be achieved with [`combine_mean`](@ref).

```@example advanced
combine_mean(scores, scores) == scores
```

We can see that `combine_mean` can work with multiple train/test tuples and combines them into one final tuple. In this case the resulting tuple consists of the means of the individual train and test score vectors.

## Combining models

We typically want to deal with probabilistic or deterministic predictions instead of raw scores. Using a [`ProbabilisticDetector`](@ref) or [`DeterministicDetector`](@ref), we can simply wrap a detector to enable such predictions. Both wrappers, however, are designed such that they can work with multiple models and combine them into one probabilistic or deterministic result. When using multiple models, we have to provide them as keyword arguments as follows.

```@example advanced
knn = ProbabilisticDetector(knn1=KNN(k=5), knn2=KNN(k=10),
                            normalize=scale_minmax,
                            combine=combine_mean)
```

As you can see, we additionally provided explicit arguments to `normalize` and `combine`, which take function arguments and are used for score normalization and combination. Those are the default, thus we could have also just left them unspecified and achieved the same result. The scores are always normalized *before* they are combined. Notice that any function that maps a train/test score tuple to a score tuple with values in the range `[0,1]` works for `normalization`. For example, if the scores are already in the range `[0,1]` we could just pass the `identity` function. Let's see the predictions of the defined detector.

```@example advanced
mach = machine(knn, X, y)
fit!(mach, rows=train)
predict(mach, rows=test)
```

Pretty simple, huh?

## Learning networks

Sometimes we need more flexibility to define outlier models. Unfortunately MLJ's [linear pipelines](https://alan-turing-institute.github.io/MLJ.jl/dev/linear_pipelines/#Linear-Pipelines) are not yet usable for outlier detection models, thus we need to define our learning networks manually. Let's, for example, create a machine that standardizes the input features before applying the detector.

```@example advanced
Xs, ys = source(X), source(y)
Xstd = transform(machine(Standardizer(), Xs), Xs)
ŷ = predict(machine(knn, Xstd), Xstd)
knn_std = machine(ProbabilisticUnsupervisedDetector(), Xs, ys; predict=ŷ)
```

We can `fit!` and `predict` with the resulting model as usual.

```@example advanced
fit!(knn_std, rows=train)
predict(knn_std, rows=test)
```

Note that we supplied labels `ys` to an unsupervised algorithm; this is not necessary if you just want to predict, but it *is necessary if you want to evaluate the resulting learning network*. We can easily export such a learning network as a model with `@from_network`.

```@example advanced
@from_network knn_std mutable struct StandardizedKNN end
```

Furthermore, if the goal is to create a standalone model from a network, we could use empty sources (`source()`) for `Xs` and `ys`. The standalone model can be bound to data again like any other model.

```@example advanced
knn_std = machine(StandardizedKNN(), X, y)
fit!(knn_std, rows=train)
predict(knn_std, rows=test)
```

There might be occasions, where our [`ProbabilisticDetector`](@ref) or [`DeterministicDetector`](@ref) wrappers are not flexible enough. In such cases we can directly use [`augmented_transform`](@ref) in our learning networks and use a [`ProbabilisticTransformer`](@ref) or [`DeterministicTransformer`](@ref), which takes one or more train/test tuples as inputs returning probabilistic or deterministic predictions.

## Implementing models

Learning networks let us flexibly create complex combinations of *existing models*, however, sometimes it's necessary to develop new outlier detection models for specific tasks. *OutlierDetection.jl* builds on top of [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) and provides a simple interface defining how an outlier detection algorithm can be implemented. Let's first import the interface and the packages relevant to our new algorithm.

```@example advanced
import OutlierDetectionInterface
const OD = OutlierDetectionInterface

using Statistics:mean
using LinearAlgebra:norm
```

Our proposed algorithm calculates a central point from the training data and defines an outlier as a point that's far away from that center. The only hyperparameter is `p` specifying which p-norm to use to calculate the distance. Using [`@detector`](@ref), which replicates [`@mlj_model`](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/#Macro-shortcut), we can define our detector struct with macro-generated keyword arguments and default values.

```@example advanced
OD.@detector mutable struct SimpleDetector <: OD.UnsupervisedDetector
    p::Float64 = 2
end
```

Our [`DetectorModel`](@ref), then, defines the *learned parameters* of our model. In this case the only learned parameter is the center.

```@example advanced
struct SimpleModel <: OD.DetectorModel
    center::AbstractArray{<:Real}
end
```

Let's further define a helper function to calculate the distance from the center.

```@example advanced
function distances_from(center, vectors::AbstractMatrix, p)
    deviations = vectors .- center
    return [norm(deviations[:, i], p) for i in 1:size(deviations, 2)]
end
```

Finally, we can implement the two methods necessary to implement a detector, namely [`fit`](@ref) and [`transform`](@ref). Please refer to the [Key Concepts](../key-concepts) to learn more about the involved methods and types.

```@example advanced
function OD.fit(detector::SimpleDetector, X::OD.Data; verbosity)::OD.Fit
    center = mean(X, dims=2)
    training_scores = distances_from(center, X, detector.p)
    return SimpleModel(center), training_scores
end

function OD.transform(detector::SimpleDetector, model::SimpleModel, X::OD.Data)::OD.Scores
    distances_from(model.center, X, detector.p)
end
```

Using a data-frontend, we can make sure that MLJ internally transforms input data to [`Data`](@ref), which refers to column-major Julia arrays with the last dimension representing an example. Registering that frontend can be achieved with [`@default_frontend`](@ref).

```@example advanced
OD.@default_frontend SimpleDetector
```

Again, we can simply wrap our detector in a [`ProbabilisticDetector`](@ref) to enable probabilistic predictions.

```@example advanced
sd = machine(ProbabilisticDetector(SimpleDetector()), X, y)
fit!(sd, rows=train)
predict(sd, rows=test)
```

Remember: Your feedback and contributions are extremely welcome, join us on [Github](https://github.com/OutlierDetectionJL/OutlierDetection.jl) or [#outlierdetection on Slack](https://julialang.slack.com/archives/C02EXTD7WGG) and get involved.