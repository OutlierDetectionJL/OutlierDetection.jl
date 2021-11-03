# Getting Started

This example demonstrates using the `OutlierDetection` API to determine the outlierness of instances in the *Thyroid Disease Dataset*, which is part of the [ODDS collection](http://odds.cs.stonybrook.edu/). We use [`OutlierDetectionData.jl`](https://github.com/OutlierDetectionJL/OutlierDetectionData.jl) to load the dataset. 

Import `MLJ`, `OutlierDetection` and `OutlierDetectionData`.

```@example ex
using MLJ
using OutlierDetection
using OutlierDetectionData: ODDS
```

Load the `"thyroid"` dataset from the `ODDS` collection.

```@example ex
X, y = ODDS.load("thyroid")
```

Create indices to split the data into 50% training and test data.

```@example ex
train, test = partition(eachindex(y), 0.5, shuffle=true, rng=0)
```

Load a [`OutlierDetectionNeighbors.KNNDetector`](@ref) and initialize it with `k=10` neighbors. 

```@example ex
KNN = @iload KNNDetector pkg=OutlierDetectionNeighbors verbosity=0
knn = KNN(k=10)
```

Bind a raw, probabilistic and deterministic detector to data using a machine.

```@example ex
knn_raw = machine(knn, X)
knn_probabilistic = machine(ProbabilisticDetector(knn), X)
knn_deterministic = machine(DeterministicDetector(knn), X)
```

Learn models from the training data.

```@example ex
fit!(knn_raw, rows=train)
fit!(knn_probabilistic, rows=train)
fit!(knn_deterministic, rows=train)
```

Transform the test data into raw outlier scores.

```@example ex
transform(knn_raw, rows=test)
```

Predict outlier probabilities based on the test data.

```@example ex
predict(knn_probabilistic, rows=test)
```

Predict outlier classes based on the test data.

```@example ex
predict(knn_deterministic, rows=test)
```

## Learn more

To learn more about the concepts in `OutlierDetection.jl`, check out the [simple usage guide](../../documentation/simple-usage/).
