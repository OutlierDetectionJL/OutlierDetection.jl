🚧 *Experimental*, install from master branch until 0.2 is released and expect breaking changes 🚧

<h1 align="center">OutlierDetection.jl</h1>
<p align="center">
  <a href="https://discord.gg/F5MPPS9t4h">
    <img src="https://img.shields.io/badge/chat-on%20discord-7289da.svg?sanitize=true" alt="Chat">
  </a>
  <a href="https://OutlierDetectionJL.github.io/OutlierDetection.jl/stable">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Documentation (stable)">
  </a>
  <a href="https://OutlierDetectionJL.github.io/OutlierDetection.jl/dev">
    <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Documentation (dev)">
  </a>
  <a href="https://github.com/OutlierDetectionJL/OutlierDetection.jl/actions">
    <img src="https://github.com/OutlierDetectionJL/OutlierDetection.jl/workflows/CI/badge.svg" alt="Build Status">
  </a>
  <a href="https://codecov.io/gh/OutlierDetectionJL/OutlierDetection.jl">
    <img src="https://codecov.io/gh/OutlierDetectionJL/OutlierDetection.jl/branch/master/graph/badge.svg" alt="Coverage">
  </a>
  <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
  <a href="#contributors-">
    <img src="https://img.shields.io/badge/all_contributors-1-green.svg?style=flat-square" alt="All Contributors">
  </a>
  <!-- ALL-CONTRIBUTORS-BADGE:END -->  
</p>

*OutlierDetection.jl* is a Julia toolkit for detecting outlying objects, also known as *anomalies*. This package is an effort to make Julia a first-class citizen in the Outlier- and Anomaly-Detection community. *Why should you use this package?*

- Provides a unified API for outlier detection in Julia
- Provides access to state-of-the-art outlier detection algorithms
- Seamlessly integrates with Julia's existing machine learning ecosystem

## Installation

It is recommended to use [Pkg.jl](https://julialang.github.io/Pkg.jl) for installation. Follow the command below to install the latest official release or use `] add OutlierDetection` in the Julia REPL.

```julia
import Pkg;
Pkg.add("OutlierDetection")
```

If you would like to modify the package locally, you can use `Pkg.develop(OutlierDetection)` or `] dev OutlierDetection` in the Julia REPL. This fetches a full clone of the package to `~/.julia/dev/` (the path can be changed by setting the environment variable `JULIA_PKG_DEVDIR`).

## API

You typically want to interface with OutlierDetection.jl through the [MLJ-API](#mlj-api). However, it's also possible to use OutlierDetection.jl without MLJ. The main parts of the API are the functions `fit`, `score`, and `to`. Note that the raw API uses the columns-as-observations convention for improved performance, and we transpose the input data.

```julia
using OutlierDetection
using OutlierDetectionData: ODDS

# create a detector (a collection of hyperparameteres)
lof = LOF()

# download and open the thyroid benchmark dataset
X, y = ODDS.load("thyroid")

# use 50% of the data for training
n_train = Int(length(y) * 0.5)
train, test = eachindex(y)[1:n_train], eachindex(y)[n_train+1:end]

# learn a model from data
model = fit(lof, X[train, :])

# predict outlier scores with learned model
train_scores, test_scores = score(lof, model, X[test, :])

# transform scores to binary labels
ŷ = detect(Class(), train_scores, test_scores)
```

## MLJ API

The main difference between the raw API and MLJ is, besides method naming differences, the introduction of a [`machine`](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/). In the raw API, we explicitly pass the results of fitting a detector (models) to further `score` calls. Machines allow us to hide that complexity by binding data directly to detectors and automatically passing fit results to further `transform` (unsupervised) or `predict` (supervised) calls. Under the hood, `transform` and `predict` pass the input data and previous fit result to `score`.

```julia
using MLJ # or using MLJBase
using OutlierDetection
using OutlierDetectionData: ODDS

# download and open the thyroid benchmark dataset
X, y = ODDS.load("thyroid");

# use 50% of the data for training
n_train = Int(length(y) * 0.5)
train, test = eachindex(y)[1:n_train], eachindex(y)[n_train+1:end]

# create a pipeline consisting of a detector and classifier
pipe = @pipeline LOF() Class()

# create a machine by binding the pipeline to data
mach = machine(pipe, X)

# learn from data
fit!(mach, rows=train)

# predict labels with learned machine
ŷ = transform(mach, rows=test)
```

## Algorithms (also known as Detectors)

Algorithms marked with '✓' are implemented in Julia. Algorithms marked with '✓ (py)' are implemented in Python (thanks to the wonderful [PyOD library](https://github.com/yzhao062/pyod)) with an existing Julia interface through [PyCall](https://github.com/JuliaPy/PyCall.jl). If you would like to know more, open the [detector reference](https://OutlierDetectionJL.github.io/OutlierDetection.jl/dev/API/detectors/). Note: If you would like to use a Python-variant of an algorithm, prepend the algorithm name with `Py`, e.g., `PyLOF` is the Python variant of `LOF`.

| Name    | Description                                  | Year  | Status | Authors                |
| ------- | -------------------------------------------- | :---: | :----: | ---------------------- |
| LMDD    | Linear deviation-based outlier detection     | 1996  | ✓ (py) | Arning et al.          |
| KNN     | Distance-based outliers                      | 1997  |   ✓    | Knorr and Ng           |
| MCD     | Minimum covariance determinant               | 1999  | ✓ (py) | Rousseeuw and Driessen |
| KNN     | Distance to the k-th nearest neighbor        | 2000  |   ✓    | Ramaswamy              |
| LOF     | Local outlier factor                         | 2000  |   ✓    | Breunig et al.         |
| OCSVM   | One-Class support vector machine             | 2001  | ✓ (py) | Schölkopf et al.       |
| KNN     | Sum of distances to the k-nearest neighbors  | 2002  |   ✓    | Angiulli and Pizzuti   |
| COF     | Connectivity-based outlier factor            | 2002  |   ✓    | Tang et al.            |
| LOCI    | Local correlation integral                   | 2003  | ✓ (py) | Papadimitirou et al.   |
| CBLOF   | Cluster-based local outliers                 | 2003  | ✓ (py) | He et al.              |
| PCA     | Principal component analysis                 | 2003  | ✓ (py) | Shyu et al.            |
| IForest | Isolation forest                             | 2008  | ✓ (py) | Liu et al.             |
| ABOD    | Angle-based outlier detection                | 2009  |   ✓    | Kriegel et al.         |
| SOD     | Subspace outlier detection                   | 2009  | ✓ (py) | Kriegel et al.         |
| HBOS    | Histogram-based outlier score                | 2012  | ✓ (py) | Goldstein and Dengel   |
| SOS     | Stochastic outlier selection                 | 2012  | ✓ (py) | Janssens et al.        |
| AE      | Auto-encoder reconstruction loss outliers    | 2015  |   ✓    | Aggarwal               |
| ABOD    | Stable angle-based outlier detection         | 2015  |   ✓    | Li et al.              |
| LODA    | Lightweight on-line detector of anomalies    | 2016  | ✓ (py) | Pevný                  |
| DeepSAD | Deep semi-supervised anomaly detection       | 2019  |   ✓    | Ruff et al.            |
| COPOD   | Copula-based outlier detection               | 2020  | ✓ (py) | Li et al.              |
| ROD     | Rotation-based outlier detection             | 2020  | ✓ (py) | Almardeny et al.       |
| ESAD    | End-to-end semi-supervised anomaly detection | 2020  |   ✓    | Huang et al.           |

If there are already so many algorithms available in Python - *why Julia, you might ask?* Let's have some fun!

```julia
using OutlierDetection, MLJ
using BenchmarkTools: @benchmark
X = rand(100000, 10);
lof = machine(LOF(k=5, algorithm=:balltree, leafsize=30, parallel=true), X) |> fit!
pylof = machine(PyLOF(n_neighbors=5, algorithm="ball_tree", leaf_size=30, n_jobs=-1), X) |> fit!
```

Julia enables you to implement your favorite algorithm in no time and it will be fast, *blazingly fast*.

```julia
@benchmark transform(lof, X)
> median time:      807.962 ms (0.00% GC)
```

Interoperating with Python is easy!

```julia
@benchmark transform(pylof, X)
> median time:      31.077 s (0.00% GC)
```

## Contributing

OutlierDetection.jl is a community effort and your help is extremely welcome! See our [contribution guide](https://OutlierDetectionJL.github.io/OutlierDetection.jl/dev/getting-started/contributing/) for more information how to contribute to the project.

### Inclusion Guidelines

We are excited to make Julia a first-class citizen in the outlier detection community and happily accept algorithm contributions to OutlierDetection.jl.

We consider well-established algorithms for inclusion. A rule of thumb is at least two years since publication, 100+ citations, and wide use and usefulness. Algorithms that do not meet the inclusion criteria can simply extend our API. The external algorithms can also be listed in our documentation if the authors wish so.

Additionally, algorithms that implement functionality that is useful on their own should live in their own package, wrapped by OutlierDetection.jl. Algorithms that build primarily on top of existing packages can be implemented directly in OutlierDetection.jl.

## Contributors ✨

Thanks go to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://fastpaced.com"><img src="https://avatars.githubusercontent.com/u/1233304?v=4?s=100" width="100px;" alt=""/><br /><sub><b>David Muhr</b></sub></a><br /><a href="https://github.com/OutlierDetectionJL/OutlierDetection.jl/commits?author=OutlierDetectionJL" title="Code">💻</a> <a href="https://github.com/OutlierDetectionJL/OutlierDetection.jl/commits?author=OutlierDetectionJL" title="Tests">⚠️</a> <a href="https://github.com/OutlierDetectionJL/OutlierDetection.jl/commits?author=OutlierDetectionJL" title="Documentation">📖</a> <a href="#maintenance-OutlierDetectionJL" title="Maintenance">🚧</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
