# Mirroring @mlj_model and adding the normalize, classify and threshold fields to the struct.
macro detector_model(ex)
    push!(ex.args[3].args, :(normalize::Function = $normalize))
    push!(ex.args[3].args, :(classify::Function = $classify))
    push!(ex.args[3].args, :(threshold::Real = 0.9))
    ex, modelname, params, defaults, constraints = MLJModelInterface._process_model_def(__module__, ex)
    const_ex = MLJModelInterface._model_constructor(modelname, params, defaults)
    push!(ex.args[3].args, const_ex)
    clean_ex = MLJModelInterface._model_cleaner(modelname, defaults, constraints)
    esc(
        quote
            Base.@__doc__($ex)
            $modelname
            $clean_ex
        end
    )
end
