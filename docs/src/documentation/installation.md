# Installation

It is recommended to use [Pkg.jl](https://julialang.github.io/Pkg.jl) for installation. Please make sure that you are using a compatible version of Julia. A list of compatible versions can be found our [CI pipeline](https://github.com/OutlierDetectionJL/OutlierDetection.jl/blob/master/.github/workflows/ci.yml).

Follow the command below to install the latest official release or use `] add OutlierDetection` in the Julia REPL.

```julia
import Pkg;
Pkg.add("OutlierDetection")
```

A specific version can be installed by appending a version after a `@` symbol, e.g. `OutlierDetection@v0.1`. Additionally, you can directly install specific branches or commits by appending a `#` symbol and the corresponding branch name or commit SHA, e.g. `OutlierDetection#master`.

If you would like to modify the package locally, you can use `Pkg.develop(OutlierDetection)` or `] dev OutlierDetection` in the Julia REPL. This fetches a full clone of the package to `~/.julia/dev/` (the path can be changed by setting the environment variable `JULIA_PKG_DEVDIR`).
