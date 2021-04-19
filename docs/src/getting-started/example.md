# Example

This example demonstrates using the *raw* `OutlierDetection` API to determine the outlierness of instances in the *Thyroid Disease Dataset*, which is part of the [ODDS collection](http://odds.cs.stonybrook.edu/). We use [`OutlierDetectionData.jl`](https://github.com/davnn/OutlierDetectionData.jl) to download and read the dataset.

Import `OutlierDetection` and `OutlierDetectionData`

```julia
using OutlierDetection: KNN, fit, transform, roc_auc
using OutlierDetectionData: ODDS
```

Download and read the `"thyroid"` dataset from the `ODDS` collection.

```julia
X, y = ODDS.read("thyroid");

# simply use the first 70% of the data for training
n_train = convert(Int, floor(length(y) * 0.7));

# split into train and test
X_train, X_test, y_train, y_test = X[:, begin:n_train-1], X[:, n_train:end], y[begin:n_train-1], y[n_train:end];
```

Initialize an unsupervised `KNN` detector with `k=10` neighbors.

```julia
detector = KNN(k=10);
```

Learn a model from the data `X`.

```julia
model, scores_train = fit(detector, X_train);
```

Evaluate our resulting scores on the training data.

```julia
roc_auc(y_train, scores_train)
```

    julia> 0.945...

Calculate the outlier scores for our test data.

```julia
scores_test = transform(detector, model, X_test);
```

Evaluate the result on the test data.

```julia
roc_auc(y_test, scores_test)
```

    julia> 0.967...

## Notice

Typically, you do not use the *raw* `OutlierDetection` API directly, but instead use [MLJ](https://github.com/alan-turing-institute/MLJ.jl) to interface with `OutlierDetection.jl`. This way you can easily integrate and extend Julia's machine learning ecosystem with outlier detection algorithms. To learn more about the usage of MLJ, check out the [guide](../../documentation/guide/)!
