"""    PyABOD(n_neighbors = 5,
              method = "fast")
$(make_docs_link("abod"))
"""
@pymodel mutable struct PyABOD <: UnsupervisedDetector
    n_neighbors::Integer = 5::(_ > 0)
    method::String = "fast"::(_ in ("default", "fast"))
end

"""    PyCBLOF(n_clusters = 8,
               alpha = 0.9,
               beta = 5,
               use_weights = false,
               random_state = nothing,
               n_jobs = 1)
$(make_docs_link("cblof"))
"""
@pymodel mutable struct PyCBLOF <: UnsupervisedDetector
    n_clusters::Integer = 8::(_ > 1)
    alpha::Real = 0.9::(0.5 < _ < 1)
    beta::Real = 5::(_ > 1)
    use_weights::Bool = false
    random_state::Union{Nothing, Integer} = nothing
    n_jobs::Integer = 1::(_ >= -1)
end

"""    PyCOF(n_neighbors = 5)
$(make_docs_link("cof"))
"""
@pymodel mutable struct PyCOF <: UnsupervisedDetector
    n_neighbors::Integer = 5::(_ > 0)
end

"""    PyCOPOD()
$(make_docs_link("copod"))
"""
@pymodel mutable struct PyCOPOD <: UnsupervisedDetector end

"""    PyHBOS(n_bins = 10,
              alpha = 0.1,
              tol = 0.5)
$(make_docs_link("hbos"))
"""
@pymodel mutable struct PyHBOS <: UnsupervisedDetector
    n_bins::Integer = 10::(_ > 1)
    alpha::Real = 0.1::(0 < _ < 1)
    tol::Real = 0.5::(0 < _ < 1)
end

"""    PyIForest(n_estimators = 100,
                 max_samples = "auto",
                 max_features = 1.0
                 bootstrap = false,
                 behaviour = "new",
                 random_state = nothing,
                 verbose = 0,
                 n_jobs = 1)
$(make_docs_link("iforest"))
"""
@pymodel mutable struct PyIForest <: UnsupervisedDetector
    n_estimators::Integer = 100::(_ > 0)
    max_samples::Union{String, Real} = "auto"
    max_features::Real = 1.0
    bootstrap::Bool = false
    behaviour::String = "new"
    random_state::Union{Nothing, Integer} = nothing
    verbose::Integer = 0::(0 <= _ <= 2)
    n_jobs::Integer = 1::(_ >= -1)
end

"""    PyKNN(n_neighbors = 5,
             method = "largest",
             radius = 1.0,
             algorithm = "auto",
             leaf_size = 30,
             metric = "minkowski",
             p = 2,
             metric_params = nothing,
             n_jobs = 1)
$(make_docs_link("knn"))
"""
@pymodel mutable struct PyKNN <: UnsupervisedDetector
    n_neighbors::Integer = 5::(_ > 0)
    method::String = "largest"::(_ in ("largest", "mean", "median"))
    radius::Real = 1.0
    algorithm::String = "auto"::(_ in ("auto", "ball_tree", "kd_tree", "brute"))
    leaf_size::Integer = 30::(_ > 0)
    metric::String = "minkowski"::(_ in ("cityblock", "cosine", "euclidean", "l1", "l2", "manhatten", "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"))
    p::Union{Nothing, Integer} = 2
    metric_params::Union{Nothing, Any} = nothing
    n_jobs::Integer = 1::(_ >= -1)
end

"""    PyLMDD(n_iter = 50,
              dis_measure = "aad",
              random_state = nothing)
$(make_docs_link("lmdd"))
"""
@pymodel mutable struct PyLMDD <: UnsupervisedDetector
    n_iter::Integer = 50::(_ > 0)
    dis_measure::String = "aad"::(_ in ("aad", "var", "iqr"))
    random_state::Union{Nothing, Integer} = nothing
end

"""    PyLODA(n_bins = 10,
              n_random_cuts = 100)
$(make_docs_link("loda"))
"""
@pymodel mutable struct PyLODA <: UnsupervisedDetector
    n_bins::Integer = 10::(_ > 1)
    n_random_cuts::Integer = 100::(_ > 0)
end

"""    PyLOF(n_neighbors = 5,
             method = "largest",
             algorithm = "auto",
             leaf_size = 30,
             metric = "minkowski",
             p = 2,
             metric_params = nothing,
             n_jobs = 1)
$(make_docs_link("lof"))
"""
@pymodel mutable struct PyLOF <: UnsupervisedDetector
    n_neighbors::Integer = 5::(_ > 0)
    algorithm::String = "auto"::(_ in ("auto", "ball_tree", "kd_tree", "brute"))
    leaf_size::Integer = 30::(_ > 0)
    metric::String = "minkowski"::(_ in ("cityblock", "cosine", "euclidean", "l1", "l2", "manhatten", "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"))
    p::Union{Nothing, Integer} = 2
    metric_params::Union{Nothing, Any} = nothing
    n_jobs::Integer = 1::(_ >= -1)
end

"""    PyLOCI(alpha = 0.5,
              k = 3)
$(make_docs_link("loci"))
"""
@pymodel mutable struct PyLOCI <: UnsupervisedDetector
    alpha::Real = 0.5::(0 < _ < 1)
    k::Real = 3::(_ > 0)
end

"""    PyMCD(store_precision = true,
             assume_centered = false,
             support_fraction = nothing,
             random_state = nothing)
$(make_docs_link("mcd"))
"""
@pymodel mutable struct PyMCD <: UnsupervisedDetector
    store_precision::Bool = true
    assume_centered::Bool = false
    support_fraction::Union{Nothing, Real} = nothing
    random_state::Union{Nothing, Integer} = nothing
end

"""    PyOCSVM(kernel = "rbf",
               degree = 3,
               gamma = "auto",
               coef0 = 0.0,
               tol = 0.001,
               nu = 0.5,
               shrinking = true,
               cache_size = 200,
               verbose = false,
               max_iter = -1)
$(make_docs_link("ocsvm"))
"""
@pymodel mutable struct PyOCSVM <: UnsupervisedDetector
    kernel::String = "rbf"::(_ in ("linear", "poly", "rbf", "sigmoid", "precomputed"))
    degree::Integer = 3::(_ > 1)
    gamma::Union{String, Real} = "auto"
    coef0::Real = 0.0
    tol::Real = 0.001
    nu::Real = 0.5::(0 < _ <= 1)
    shrinking::Bool = true
    cache_size::Integer = 200::(_ > 0)
    verbose::Bool = false
    max_iter::Integer = -1
end

"""    PyPCA(n_components = nothing,
             n_selected_components = nothing,
             copy = true,
             whiten = false,
             svd_solver = "auto",
             tol = 0.0
             iterated_power = "auto",
             standardization = true,
             weighted = true,
             random_state = nothing)
$(make_docs_link("pca"))
"""
@pymodel mutable struct PyPCA <: UnsupervisedDetector
    n_components::Union{Nothing, Real} = nothing
    n_selected_components::Union{Nothing, Integer} = nothing
    copy::Bool = true
    whiten::Bool = false
    svd_solver::String = "auto"::(_ in ("auto", "full", "arpack", "randomized"))
    tol::Real = 0.0
    iterated_power::Union{String, Integer} = "auto"
    standardization::Bool = true
    weighted::Bool = true
    random_state::Union{Nothing, Integer} = nothing
end

"""    PyROD(parallel_execution = false)
$(make_docs_link("rod"))
"""
@pymodel mutable struct PyROD <: UnsupervisedDetector
    parallel_execution::Bool = false
end

"""    PySOD(n_neighbors = 5,
             ref_set = 10,
             alpha = 0.8)
$(make_docs_link("sod"))
"""
@pymodel mutable struct PySOD <: UnsupervisedDetector
    n_neighbors::Integer = 20::(_ > 0)
    ref_set::Integer = 10::(_ > 0)
    alpha::Real = 0.8::(0 < _ < 1)
end

"""    PySOS(perplexity = 4.5,
             metric = "minkowski",
             eps = 1e-5)
$(make_docs_link("sos"))
"""
@pymodel mutable struct PySOS <: UnsupervisedDetector
    perplexity::Real = 4.5::(_ > 0)
    metric::String = "minkowski"::(_ in ("cityblock", "cosine", "euclidean", "l1", "l2", "manhatten", "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"))
    eps::Real = 1e-5::(_ > 0)
end
