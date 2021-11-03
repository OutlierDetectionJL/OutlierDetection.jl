# Simple Usage

Let's import the necessary packages first.

```@example simple
using MLJ
using OutlierDetection
using OutlierDetectionData: ODDS
```

## Loading data

We can list the available datasets in the imported `ODDS` dataset collection with [`list`](@ref)

```@example simple
ODDS.list()
```

We can now [`load`](@ref) a dataset by specifying its name.

```@example simple
X, y = ODDS.load("annthyroid")
```

## Data formats

Because `OutlierDetection.jl` is built upon MLJ, there are some things to know regarding the data used in outlier detection tasks. A detector can typically be instantiated with continuous data `X` satisfying the [`Tables.jl`](https://tables.juliadata.org/stable/) interface. Often we use [`DataFrames.jl`](https://dataframes.juliadata.org/stable/) to create such tables. An important distinction to know is the difference between *machine types* and *scientific types*.

- The *machine* type refers to the Julia type being used to represent the object (for instance, Float64).
- The *scientific* type is one of the types defined in [`ScientificTypes.jl`](https://juliaai.github.io/ScientificTypes.jl/stable/) reflecting how the object should be interpreted (for instance, `Continuous` or `Multiclass`).

We can examine the machine and scientific types of our loaded dataframe `X` with `ScientificTypes.schema`.

```@example simple
schema(X)
```

Fortunately, our table contains only `Continuous` data as expected. Labels in outlier detection are always encoded as a categorical vectors with classes `"normal"` and `"outlier"` and scitype `OrderedFactor{2}`. Data with type `OrderedFactor{2}` is considered to have an intrinsic "positive" class, in our case `"outlier"`. Measures, such as `true_positive` assume the second class in the ordering is the "positive" class. Using the helper [`to_categorical`](@ref), we can transform a `Vector{String}` to a categorical vector, which ensures there are only two classes and the positive class is `"outlier"`. We don't need to coerce `y` to a categorical array in our example because `load` already returns categorical vectors.

```@example simple
to_categorical(["normal", "normal", "outlier"])
```

## Loading models

Having the data ready, we can list all available detectors in MLJ. By convention, a detector is named `$(Name)Detector` in MLJ, e.g. `KNNDetector` and we can thus simply search for "Detector".

```@example simple
models("Detector")
```

Loading a detector of your choice is simple with `@load` or `@iload`, see [loading model code](https://alan-turing-institute.github.io/MLJ.jl/dev/loading_model_code/). There are multiple detectors named `KNNDetector`, thus we specify the package beforehand.

```@example simple
KNN = @iload KNNDetector pkg=OutlierDetectionNeighbors verbosity=0
```

To enable later evaluation, we wrap a raw detector (which only defines `transform` returning raw outlier scores) in a [`ProbabilisticDetector`](@ref); this enables us to `predict` outlier probabilities from the raw scores.

```@example simple
knn = ProbabilisticDetector(KNN())
```

Note that the call above assumes that you want to use the default parameters to instantiate the [`OutlierDetectionNeighbors.KNNDetector`](@ref) and [`ProbabilisticDetector`](@ref), e.g. `k=5` so on.

## Model evaluation

We can now evaluate how such a model performs. By default, a probabilistic detector is evaluated using `area_under_curve`, but there are a lot of other evaluation strategies available, see the [list of measures](https://alan-turing-institute.github.io/MLJ.jl/dev/performance_measures/#List-of-measures). We use stratified five-fold cross validation to evaluate our model, but other [resampling strategies](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/#Built-in-resampling-strategies) are possible as well.

```@example simple
cv = StratifiedCV(nfolds=5, shuffle=true, rng=0)
evaluate(knn, X, y; resampling=cv)
```

## Model optimization

As previously mentioned, we used the default parameters to create our model. However, we typically don't know an appropriate amount of neighbors (`k`) beforehand. Using MLJ's built-in model tuning we can identify the best `k` given some performance measure.

Let's first define a range of possible parameter values for `k`.

```@example simple
r = range(knn, :(detector.k), values=[1,2,3,4,5:5:100...])
```

We can then use this range, or multiple ranges, to create a tuned model by additionally specifing a [tuning-strategy](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/), which defines how to efficiently evaluate ranges. In our case we use a simple grid search to evaluate all the given parameter options.

```@example simple
t = TunedModel(model=knn, resampling=cv, tuning=Grid(), range=r, acceleration=CPUThreads())
```

We can again bind that model to data and fit it. Fitting a tuned model instigates a search for optimal model hyperparameters, within specified `range`s, and then uses all supplied data to train the best model.

```@example simple
m = machine(t, X, y) |> fit!
```

Using the machines' report, we can idenity the best evaluation results.

```@example simple
report(m).best_history_entry
```

Additionally, we can easily extract the best identified model.

```@example simple
b = report(m).best_model
```

Let's evaluate the best model again to make sure it achieves the expected performance.

```@example simple
evaluate(b, X, y, resampling=cv)
```

## Model usage

Now that we have found the best model, we can use it to determine outliers in the data. Converting scores to classes can be achieved with a [`DeterministicDetector`](@ref). Let's create some fake train/test indices and suppose we want to identify outliers in the test data.

```@example simple
train, test = partition(eachindex(y), 0.5, shuffle=true, stratify=y, rng=0)
```

Let's determine the [`outlier_fraction`](@ref) in the training data, which we then use to determine a threshold to convert the outlier scores into classes. Using [`classify_quantile`](@ref), we can create a classification function based on quantiles of the training data. In the following example we define an outlier's score to lie above the *1 - outlier_fraction* training scores' quantile.

```@example simple
threshold = classify_quantile(1 - outlier_fraction(y[train]))
final = machine(DeterministicDetector(b.detector, classify=threshold), X)
fit!(final, rows=train)
```

Using `predict` allows us to determine the outliers in the test data.

```@example simple
ŷ = predict(final, rows=test)
```

## Model persistence

Finally, we can store the model with `MLJ.save`.

```@example simple
MLJ.save("final.jlso", final)
```

Loading the model again, the machine is not bound to data anymore, but we can bind it to data if we supply `X` again.

```@example simple
final = machine("final.jlso")
```

We can still use the machine to predict, even though its not bound to data.

```@example simple
ŷ == predict(final, X[test, :])
```

If you would like to know how you can combine detectors or how to develop your own detectors, continue with the [Advanced Usage](../advanced-usage) guide.
