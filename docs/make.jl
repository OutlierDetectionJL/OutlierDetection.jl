using OutlierDetection
using Documenter

makedocs(;
    modules=[OutlierDetection],
    authors="David Muhr <muhrdavid@gmail.com> and contributors",
    repo="https://github.com/davnn/OutlierDetection.jl/blob/{commit}{path}#L{line}",
    sitename="OutlierDetection.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://davnn.github.io/OutlierDetection.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/davnn/OutlierDetection.jl",
)
