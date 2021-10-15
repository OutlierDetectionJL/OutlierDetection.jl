using OutlierDetection
using OutlierDetectionData
using OutlierDetectionInterface
using OutlierDetectionNetworks
using OutlierDetectionNeighbors
using OutlierDetectionPython

using Documenter
using DocumenterMarkdown

DocMeta.setdocmeta!(OutlierDetection, :DocTestSetup, :(); recursive=true)
makedocs(;
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
        OutlierDetectionPython
    ],
    clean=true
)
