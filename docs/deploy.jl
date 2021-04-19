using Documenter

deploydocs(;
    repo="github.com/davnn/OutlierDetection.jl",
    target = "site",
    push_preview = true,
    deps = Deps.pip("mkdocs==1.1.2", "mkdocs-material==7.1.2", "pymdown-extensions==8.1.1",
                    "pygments==2.8.1", "Jinja2==2.11.3"),
    make = () -> run(`mkdocs build`)
)
