using Documenter

deploydocs(;
    repo="github.com/davnn/OutlierDetection.jl",
    target = "build",
    push_preview = true,
    deps = Deps.pip("mkdocs", "mkdocs-material", "pymdown-extensions", "pygments")
)
