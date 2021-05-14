using .MLJ

export deterministic, probabilistic

# deterministic model api
function deterministic(model, Xs::MLJ.Source, ys::MLJ.Source)
    model = scitype(model) <: MLJ.UnsupervisedScitype ?
            transform(machine(model, Xs), Xs) :
            predict(machine(model, Xs, ys), Xs)
    mach = machine(Deterministic(), Xs, ys, predict=model)
    return mach
end

# deterministic network api
function deterministic(model, Xs::MLJ.Source, ys::MLJ.Source, scores...)
    model = transform(machine(model), scores...)
    mach = machine(Deterministic(), Xs, ys, predict=model)
    return mach
end

# probabilistic model api
function probabilistic(model, Xs::MLJ.Source, ys::MLJ.Source)
    model = scitype(model) <: MLJ.UnsupervisedScitype ?
            transform(machine(model, Xs), Xs) :
            predict(machine(model, Xs, ys), Xs)
    mach = machine(Probabilistic(), Xs, ys, predict=model)
    return mach
end

# probabilistic network api
function probabilistic(model, Xs::MLJ.Source, ys::MLJ.Source, scores...)
    model = transform(machine(model), scores...)
    mach = machine(Probabilistic(), Xs, ys, predict=model)
    return mach
end
