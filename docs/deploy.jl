using Documenter

deploydocs(;
    repo="github.com/davnn/OutlierDetection.jl",
    target = "site",
    push_preview = true,
    deps = Deps.pip("pygments", "mkdocs", "mkdocs-material")
)
