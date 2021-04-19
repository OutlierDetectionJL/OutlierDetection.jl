using Documenter

deploydocs(;
    repo="github.com/davnn/OutlierDetection.jl",
    target = "site",
    push_preview = true,
    deps = Deps.pip("mkdocs", "mkdocs-material", "pymdown-extensions", "pygments"),
    make = () -> run(`mkdocs build`)
)
