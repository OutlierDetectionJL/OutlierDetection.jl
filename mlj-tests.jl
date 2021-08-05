# Helper because we currently cannot use @pipeline inside tests because detector is not available in global scope,
# https://discourse.julialang.org/t/defining-mlj-pipelines-within-a-function/60552/2
mutable struct Pipeline{D,E} <: MLJ.Unsupervised
    detector::D
    evaluator::E
end

function MLJ.fit(pipe::Pipeline, verbosity, X)
    Xs = source(X)
    ys = source(zeros(nrows(X))) # we dont really need labels here
    scores = isa(pipe.detector, SupervisedDetector) ? 
             transform(machine(pipe.detector, Xs, ys), Xs) :
             transform(machine(pipe.detector, Xs), Xs)
    result = transform(machine(pipe.evaluator), scores)
    mach = machine(Unsupervised(), Xs, predict=result) |> fit!
    return!(mach, pipe, verbosity)
end

# MLJ model with categorical output
pipe_cls = Pipeline(detector, Class())
detector_cat = deterministic(pipe_cls, X_dfs, ys)
fit!(detector_cat, rows=train)
test_class = predict(detector_cat, rows=test)

# MLJ network with categorical output
network = is_supervised ? transform(machine(detector, X_dfs, ys), X_dfs) :
                            transform(machine(detector, X_dfs), X_dfs)
detector_cat_net = deterministic(Class(), X_dfs, ys, network)
fit!(detector_cat_net, rows=train)
test_class_net = predict(detector_cat_net, rows=test)

# MLJ model with probabilistic output
pipe_scr = Pipeline(detector, Score())
detector_prob = probabilistic(pipe_scr, X_dfs, ys)
fit!(detector_prob, rows=train)
test_prob = predict(detector_prob, rows=test)

# MLJ network with probabilistic output
detector_prob_net = probabilistic(Score(), X_dfs, ys, network)
fit!(detector_prob_net, rows=train)
test_prob_net = predict(detector_prob, rows=test)
