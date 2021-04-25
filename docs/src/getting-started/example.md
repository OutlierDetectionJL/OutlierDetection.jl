# Example

This example demonstrates using the *raw* `OutlierDetection` API to determine the outlierness of instances in the *Thyroid Disease Dataset*, which is part of the [ODDS collection](http://odds.cs.stonybrook.edu/). We use [`OutlierDetectionData.jl`](https://github.com/davnn/OutlierDetectionData.jl) to download and read the dataset. Note that the raw API uses the *columns-as-observations* convention for improved performance.

Import `OutlierDetection` and `OutlierDetectionData`

```julia
using OutlierDetection
using OutlierDetectionData: ODDS
```

Download and read the `"thyroid"` dataset from the `ODDS` collection.

```julia
X, y = ODDS.read("thyroid")
```

Create indices to split the data into 50% training and test data.

```
n_train = Int(length(y) * 0.5)
train, test = eachindex(y)[1:n_train], eachindex(y)[n_train+1:end]
```

Initialize an unsupervised [`KNN`](@ref) [`Detector`](@ref) with `k=10` neighbors.

```julia
detector = KNN(k=10)
```

Learn a model from the data `X`.

```julia
fitresult = fit(detector, X[train, :]')
```

Evaluate the resulting training data scores (stored in the fit result).

```julia
roc_auc(y[train], fitresult.scores)
```

Calculate the outlier scores for our test data. Note that we always return both the train and test scores because later used [classifiers](../../API/base/#OutlierDetection.Classifier) typically choose a threshold based on the train scores.

```julia
scores_train, scores_test = score(detector, fitresult, X[test, :]')
```

Evaluate the resulting test scores with the given labels.

```julia
roc_auc(y[test], scores_test)
```

You can easily convert the obtained scores into inlier (`1`), and outlier (`-1`) labels using a [`Classifier`](@ref), in this case, [`Binarize`](@ref).

```julia
clf = Binarize()
detect(clf, scores_train, scores_test)
```

## Using MLJ

Typically, you do not use the *raw* `OutlierDetection` API, but instead use [MLJ](https://github.com/alan-turing-institute/MLJ.jl) to interface with `OutlierDetection.jl`. The main difference between the raw API and MLJ is, besides method naming differences, the introduction of a [`machine`](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/). In the raw API, we explicitly pass the results of fitting a detector (models) to further `score` calls. Machines allow us to hide that complexity by binding data directly to detectors and automatically passing fit results to further `transform` (unsupervised) or `predict` (supervised) calls. Under the hood, `transform` and `predict` pass the input data and previous fit result to `score`.

Given that you have already imported `OutlierDetection` and loaded `X` and `y` as described before, import `MLJ`.

```julia
using MLJ # or MLJBase
```

Create a pipeline consisting of a detector and classifier.

```julia
pipe = @pipeline KNN(k=10) Binarize()
```

Bind the pipeline to data to create a machine.

```julia
mach = machine(pipe, X)
```

Fit the machine to learn from the training input data.

```julia
fit!(mach, rows=train)
```

Predict the labels for the test data with the learned machine.

```julia
transform(mach, rows=test)
```

## Learn more

To learn more about the concepts in `OutlierDetection.jl`, check out the [guide](../../documentation/guide/)!
