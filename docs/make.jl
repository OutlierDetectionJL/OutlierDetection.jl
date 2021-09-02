using OutlierDetection
using OutlierDetectionData
using OutlierDetectionInterface
using OutlierDetectionNetworks
using OutlierDetectionNeighbors
using OutlierDetectionPy

using Documenter
using DocumenterMarkdown

# DocMeta.setdocmeta!(OutlierDetection, :DocTestSetup, :(using OutlierDetection); recursive=true)
makedocs(;
    doctest = VERSION == v"1.6",
    sitename="OutlierDetection.jl",
    authors="David Muhr <muhrdavid@gmail.com> and contributors",
    repo="https://github.com/OutlierDetectionJL/OutlierDetection.jl/blob/{commit}{path}#L{line}",
    format = Markdown(),
    modules = [
        OutlierDetection,
        OutlierDetectionData,
        OutlierDetectionInterface,
        OutlierDetectionNetworks,
        OutlierDetectionNeighbors,
        OutlierDetectionPy
    ],
    clean=true
)
