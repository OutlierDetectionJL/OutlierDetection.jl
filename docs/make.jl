using OutlierDetection
using Documenter
using DocumenterMarkdown

DocMeta.setdocmeta!(OutlierDetection, :DocTestSetup, :(using OutlierDetection, OutlierDetectionData); recursive=true)
makedocs(;
    doctest = VERSION == v"1.6",
    sitename="OutlierDetection.jl",
    authors="David Muhr <muhrdavid@gmail.com> and contributors",
    repo="https://github.com/davnn/OutlierDetection.jl/blob/{commit}{path}#L{line}",
    format = Markdown(),
    clean=true
)
