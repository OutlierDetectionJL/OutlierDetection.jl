using ROCAnalysis: AUC
import ROCAnalysis: roc

"""
    roc(labels,
        scores,
        posclass = -1;
        laplace = false,
        collapse = true)

Utility method to create the ROC from scores and binary labels.
"""
function roc(labels::Vector{U}, scores::Vector{T}, posclass::V = -1;
    laplace::Bool=false, collapse::Bool=true) where {U <: Integer,T <: Real,V <: Integer}
    classes = unique(labels)
    if posclass ∉ classes
        throw(DomainError("posclass ∉ classes", "The specified positive class = $posclass cannot be found in labels."))
    end

    if length(labels) != length(scores)
        throw(DomainError("length(labels) != length(scores)", "Labels and scores must be of equal length."))
    end

    if length(classes) > 2
        @warn "More than two classes detected in labels, treating every label as negative except $posclass"
    end

    pos = labels .== posclass
    roc(scores[pos], scores[.!(pos)]; laplace, collapse)
end

"""
    roc_auc(labels,
            scores,
            posclass = -1)

Utility method to create the ROC AUC from scores and binary labels.
"""
function roc_auc(labels::Vector{U}, scores:: AbstractVector{T},
    posclass::V = -1; laplace::Bool=false, collapse::Bool=true) where {U <: Integer,T <: Real, V <: Integer}
    r = roc(labels, scores, posclass; laplace=laplace, collapse=collapse)
    AUC(r)
end
