<h1 align="center">OutlierDetection.jl</h1>
<p align="center">
  <a href="https://discord.gg/5ErtExMV">
    <img src="https://img.shields.io/badge/chat-on%20discord-7289da.svg?sanitize=true" alt="Chat">
  </a>
  <a href="https://davnn.github.io/OutlierDetection.jl/stable">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Documentation (stable)">
  </a>
  <a href="https://davnn.github.io/OutlierDetection.jl/dev">
    <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Documentation (dev)">
  </a>
  <a href="https://github.com/davnn/OutlierDetection.jl/actions">
    <img src="https://github.com/davnn/OutlierDetection.jl/workflows/CI/badge.svg" alt="Build Status">
  </a>
  <a href="https://codecov.io/gh/davnn/OutlierDetection.jl">
    <img src="https://codecov.io/gh/davnn/OutlierDetection.jl/branch/master/graph/badge.svg" alt="Coverage">
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

## API Demo

```julia
# train a local outlier factor model
using OutlierDetection
lof = LOF()
X_train = rand(10, 50)
X_test = rand(10, 20)
model, scores = fit(lof, X_train) # model + train scores
transform(lof, model, X_test) # test scores
```

## MLJ Demo

## Contributing

OutlierDetection.jl is a community effort and your help is extremely welcome! See our [contribution guide](https://davnn.github.io/OutlierDetection.jl/stable/contributing) for more information on how to contribute to the project.

### Inclusion Guidelines

We are excited to make Julia a first-class citizen in the outlier detection community and happily accept algorithm contributions to OutlierDetection.jl.

We consider well-established algorithms for inclusion. A rule of thumb is at least 2 years since publication, 100+ citations, and wide use and usefulness. Algorithms that do not meet the inclusion criteria can simply extend our API. External algorithm can also be listed in our documentation, if the authors wish so.

Additionally, algorithms that implement functionality that is useful on its own should live in their own package, wrapped by OutlierDetection.jl. Algorithms that build largely on top of existing packages can be implemented directly in OutlierDetection.jl.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://fastpaced.com"><img src="https://avatars.githubusercontent.com/u/1233304?v=4?s=100" width="100px;" alt=""/><br /><sub><b>David Muhr</b></sub></a><br /><a href="https://github.com/davnn/OutlierDetection.jl/commits?author=davnn" title="Code">💻</a> <a href="https://github.com/davnn/OutlierDetection.jl/commits?author=davnn" title="Tests">⚠️</a> <a href="https://github.com/davnn/OutlierDetection.jl/commits?author=davnn" title="Documentation">📖</a> <a href="#maintenance-davnn" title="Maintenance">🚧</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
