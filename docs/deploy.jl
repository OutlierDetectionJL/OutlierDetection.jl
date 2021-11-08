using Documenter

deploydocs(;
    repo="github.com/OutlierDetectionJL/OutlierDetection.jl",
    target = "site",
    push_preview = true,
    # mkdocs-material bundles all other required dependencies
    deps = Deps.pip("mkdocs-material==7.3.6"),
    make = () -> run(`mkdocs build`)
)
