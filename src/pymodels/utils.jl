using PyCall:PyObject, PyReverseDims, pyimport_conda

struct PyModel <: Model
    pyobject::PyObject
end

# lazily import a python model
pyod_import(name::String) = () -> begin
    module_name = lowercase(name)
    # names are not consistently uppercased
    model_name = name == "IForest" ? name : uppercase(name)
    getproperty(pyimport_conda("pyod.models.$(module_name)", "pyod", "conda-forge"), model_name)
end
pyod_import(name::Symbol) = pyod_import(String(name))

make_docs_link(name::String) =
    "<https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.$(lowercase(name))>"

"""
    pyod_fit(modelname, params)

Implements the `fit` method for an underlying python model.
"""
function pyod_fit(modelname, params)
    pymodelname = String(modelname)[3:end]
    quote
        function fit(model::$modelname, X::Data)::Fit
            Xt = PyReverseDims(X) # from column-major to row-major
            # load the underlying python model with key-word arguments
            detector = pyod_import($pymodelname)()($((Expr(:kw, p, :(model.$p)) for p in params)...))
            detector.fit(Xt)
            # the underlying python model is out model
            return Fit(PyModel(detector), detector.decision_scores_)
        end
    end
end

"""
    pyod_score(modelname)

Implements the `score` method for an underlying python model.
"""
function pyod_score(modelname)
    quote
        function score(_::$modelname, fitresult::Fit, X::Data)::Result
            Xt = PyReverseDims(X) # change from column-major to row-major
            scores_test = fitresult.model.pyobject.decision_function(Xt)
            return fitresult.scores, scores_test
        end
    end
end

"""
    py_constructor(expr)
Extracts the relevant information from the expr and build the expression
corresponding to the model constructor (see [`_model_constructor`](@ref)).
"""
function py_constructor(expr)
    # similar to @mlj_model
    expr, modelname, params, defaults, constraints = MMI._process_model_def(@__MODULE__, expr)

    # keyword constructor
    const_expr = MMI._model_constructor(modelname, params, defaults)

    # associate the constructor with the definition of the struct
    push!(expr.args[3].args, const_expr)

    # cleaner
    clean_expr = MMI._model_cleaner(modelname, defaults, constraints)

    # return
    return modelname, params, clean_expr, expr
end

macro pymodel(expr)
    modelname, params, clean_expr, expr = py_constructor(expr)
    @assert startswith(String(modelname), "Py") "A python model name has to start with `Py`, e.g. PyKNN"
    expr = quote
        Base.@__doc__($expr)
        $clean_expr
        $(pyod_fit(modelname, params))
        $(pyod_score(modelname))
        MMI.metadata_pkg($modelname,
            package_name="OutlierDetection.jl",
            package_uuid="262411bb-c475-4342-ba9e-03b8c0183ca6",
            package_url="https://github.com/davnn/OutlierDetection.jl",
            is_pure_julia=false,
            package_license="MIT",
            is_wrapper=false
        )
    end
    esc(expr)
end
