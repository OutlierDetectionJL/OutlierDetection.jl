using Documenter

deploydocs(;
    repo="github.com/OutlierDetectionJL/OutlierDetection.jl",
    target = "site",
    push_preview = false,
    # mkdocs-material bundles all other required dependencies
    deps = Deps.pip("mkdocs-material"),
    make = () -> run(`mkdocs build`)
)
