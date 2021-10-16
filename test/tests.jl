# helper to create fake data from scores such that the detectors return the scores again
to_table(points) = reshape(repeat(points, length(points)), (length(points), length(points))) |> table
const base = [1., 2, 3]
const test_scores = [base, # train and test equal
                     [1.,2], # len(test) < len(train)
                     [-Inf, 0., 1, 2, 3, 4, Inf], # len(test) > len(train)
                     [1.,1,1],  # len(test) = len(train)
                     [2.,2,2],
                     [3.,3,3]]
const scores0, scores1, scores2, scores3, scores4, scores5 = map(t -> (base, t), test_scores)
const X_test0, X_test1, X_test2, X_test3, X_test4, X_test5 = map(to_table, test_scores)
const X = to_table(base)
const y = to_categorical(fill(missing, length(base)))

# # fake detectors
supervised = MinimalSupervisedDetector()
unsupervised = MinimalUnsupervisedDetector()

# # fake machines
u = machine(unsupervised, X) |> fit!
s = machine(supervised, X, y) |> fit!
raw_machines = [u, s]

uc = machine(CompositeDetector(unsupervised), X) |> fit!
sc = machine(CompositeDetector(supervised), X, y) |> fit!
up = machine(ProbabilisticDetector(unsupervised), X) |> fit!
sp = machine(ProbabilisticDetector(supervised), X, y) |> fit!
ud = machine(DeterministicDetector(unsupervised), X) |> fit!
sd = machine(DeterministicDetector(supervised), X, y) |> fit!
composite_machines = [uc, sc, up, sp, ud, sd]

upc = machine(ProbabilisticDetector(CompositeDetector(unsupervised)), X) |> fit!
spc = machine(ProbabilisticDetector(CompositeDetector(supervised)), X, y) |> fit!
udc = machine(DeterministicDetector(CompositeDetector(unsupervised)), X) |> fit!
sdc = machine(DeterministicDetector(CompositeDetector(supervised)), X, y) |> fit!
wrapped_machines = [upc, spc, udc, sdc]

usc = machine(CompositeDetector(u=unsupervised, s=supervised), X, y) |> fit!
usp = machine(ProbabilisticDetector(u=unsupervised, s=supervised), X, y) |> fit!
usd = machine(DeterministicDetector(u=unsupervised, s=supervised), X, y) |> fit!
combined_machines = [usc, usp, usd]

machines = [composite_machines..., wrapped_machines..., combined_machines...]
probabilistic_machines = [up, sp, upc, spc, usp]
deterministic_machines = [ud, sd, udc, sdc, usd]

@testset "normalization, combination and classification" begin
    minmax_test = score_tuple -> scale_minmax(score_tuple...)[2]
    unify_test = score_tuple -> scale_unify(score_tuple...)[2]
    classify = score_tuple -> classify_percentile(DEFAULT_THRESHOLD)(scale_minmax(score_tuple...)...)[2]

    @testset "scores" begin
        raw_proba(detector, data) = from_univariate_finite.(predict(detector, data))
        to_scores(data) = [[transform(m, data) for m in machines]..., # transform works for all machines
                           [raw_proba(m, data) for m in probabilistic_machines]...]
        test_scoring(scores, data, labels) = @test all((labels,) .== [minmax_test(scores), to_scores(data)...])

        # Tests expected results on simple score vectors
        test_scoring(scores0, X_test0, [0.0, 0.5, 1.0])
        test_scoring(scores1, X_test1, [0.0, 0.5])
        test_scoring(scores2, X_test2, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        test_scoring(scores3, X_test3, [0.0, 0.0, 0.0])
        test_scoring(scores4, X_test4, [0.5, 0.5, 0.5])
        test_scoring(scores5, X_test5, [1.0, 1.0, 1.0])

        # Min-Max normalization and unify should not change sort order
        @test sortperm(minmax_test(scores0)) == sortperm(unify_test(scores0))
        @test sortperm(minmax_test(scores1)) == sortperm(unify_test(scores1))
        @test sortperm(minmax_test(scores2)) == sortperm(unify_test(scores2))
        @test sortperm(minmax_test(scores3)) == sortperm(unify_test(scores3))
        @test sortperm(minmax_test(scores4)) == sortperm(unify_test(scores4))
        @test sortperm(minmax_test(scores5)) == sortperm(unify_test(scores5))
    end

    @testset "classification" begin
        raw_class(detector, data) = from_categorical.(predict(detector, data))
        to_classes(data) = [[predict(m, data) for m in deterministic_machines]...,
                            [raw_class(m, data) for m in deterministic_machines]...]
        test_classification(scores, data, labels) = @test all((labels,) .== [classify(scores), to_classes(data)...])

        # Test expected classification results with given threshold
        test_classification(scores0, X_test0, ["normal", "normal", "outlier"])
        test_classification(scores1, X_test1, ["normal", "normal"])
        test_classification(scores2, X_test2, ["normal", "normal", "normal", "normal", "outlier", "outlier", "outlier"])
        test_classification(scores3, X_test3, ["normal", "normal", "normal"])
        test_classification(scores4, X_test4, ["normal", "normal", "normal"])
        test_classification(scores5, X_test5, ["outlier", "outlier", "outlier"])
    end

    @testset "combination of scores" begin
        @test combine_mean(minmax_test.([scores3, scores4])...) == [0.25, 0.25, 0.25]
        @test combine_mean(minmax_test.([scores3, scores5])...) == [0.5, 0.5, 0.5]
        @test combine_mean(minmax_test.([scores4, scores5])...) == [0.75, 0.75, 0.75]

        @test combine_median(minmax_test.([scores3, scores4])...) == [0.25, 0.25, 0.25]
        @test combine_median(minmax_test.([scores3, scores5])...) == [0.5, 0.5, 0.5]
        @test combine_median(minmax_test.([scores4, scores5])...) == [0.75, 0.75, 0.75]

        @test combine_max(minmax_test.([scores3, scores4])...) == [0.5, 0.5, 0.5]
        @test combine_max(minmax_test.([scores3, scores5])...) == [1.0, 1.0, 1.0]
        @test combine_max(minmax_test.([scores4, scores5])...) == [1.0, 1.0, 1.0]
    end
end

@testset "wrappers result in expected types" begin
    # unsupervised only
    @test CompositeDetector(unsupervised) isa UnsupervisedDetectorComposite
    @test CompositeDetector(detector=unsupervised) isa UnsupervisedDetectorComposite
    @test CompositeDetector(uns1=unsupervised, uns2=unsupervised) isa UnsupervisedDetectorComposite

    @test ProbabilisticDetector(unsupervised) isa ProbabilisticUnsupervisedDetectorComposite
    @test ProbabilisticDetector(detector=unsupervised) isa ProbabilisticUnsupervisedDetectorComposite
    @test ProbabilisticDetector(uns1=unsupervised, uns2=unsupervised) isa ProbabilisticUnsupervisedDetectorComposite
    
    @test DeterministicDetector(unsupervised) isa DeterministicUnsupervisedDetectorComposite
    @test DeterministicDetector(detector=unsupervised) isa DeterministicUnsupervisedDetectorComposite
    @test DeterministicDetector(uns1=unsupervised, uns2=unsupervised) isa DeterministicUnsupervisedDetectorComposite

    # supervised only
    @test CompositeDetector(supervised) isa SupervisedDetectorComposite
    @test CompositeDetector(detector=supervised) isa SupervisedDetectorComposite
    @test CompositeDetector(sup1=supervised, sup2=supervised) isa SupervisedDetectorComposite

    @test ProbabilisticDetector(supervised) isa ProbabilisticSupervisedDetectorComposite
    @test ProbabilisticDetector(detector=supervised) isa ProbabilisticSupervisedDetectorComposite
    @test ProbabilisticDetector(sup1=supervised, sup2=supervised) isa ProbabilisticSupervisedDetectorComposite
    
    @test DeterministicDetector(supervised) isa DeterministicSupervisedDetectorComposite
    @test DeterministicDetector(detector=supervised) isa DeterministicSupervisedDetectorComposite
    @test DeterministicDetector(sup1=supervised, sup2=supervised) isa DeterministicSupervisedDetectorComposite

    # mixed supervised and unsupervised
    @test CompositeDetector(sup=supervised, uns=unsupervised) isa SupervisedDetectorComposite
    @test ProbabilisticDetector(sup=supervised, uns=unsupervised) isa ProbabilisticSupervisedDetectorComposite
    @test DeterministicDetector(sup=supervised, uns=unsupervised) isa DeterministicSupervisedDetectorComposite

    # wrapped composites
    @test CompositeDetector(CompositeDetector(unsupervised)) isa UnsupervisedDetectorComposite
    @test ProbabilisticDetector(CompositeDetector(unsupervised)) isa ProbabilisticUnsupervisedDetectorComposite
    @test DeterministicDetector(CompositeDetector(unsupervised)) isa DeterministicUnsupervisedDetectorComposite

    @test CompositeDetector(CompositeDetector(supervised)) isa SupervisedDetectorComposite
    @test ProbabilisticDetector(CompositeDetector(supervised)) isa ProbabilisticSupervisedDetectorComposite
    @test DeterministicDetector(CompositeDetector(supervised)) isa DeterministicSupervisedDetectorComposite
end

@testset "wrappers property access" begin
    normalization_strategy = scale_unify
    combination_strategy = combine_max
    classification_strategy = classify_percentile(0.5)

    for m in getproperty.(machines, :model)
        initial_normalization_strategy = m.normalize
        initial_combination_strategy = m.combine
        m.normalize = normalization_strategy
        m.combine =  combination_strategy
        @test m.normalize == normalization_strategy
        @test m.combine == combination_strategy
        # reset back to original value
        m.normalize = initial_normalization_strategy
        m.combine = initial_combination_strategy
    end

    for m in getproperty.(deterministic_machines, :model)
        initial_classification_strategy = m.classify
        m.classify = classification_strategy
        @test m.classify == classification_strategy
        # reset back to original value
        m.classify = initial_classification_strategy
    end

    # wrappers throw an error if a property does not exist
    for m in getproperty.(machines, :model)
        @test_throws ErrorException m.foo
        @test_throws ErrorException m.foo = "bar"
    end
end

@testset "erroneous wrapper calls" begin
    # wrappers do not work with models other than detectors
    static_model = MLJBase.WrappedFunction(identity)
    @test_throws MethodError CompositeDetector(static_model)
    @test_throws MethodError ProbabilisticDetector(static_model)
    @test_throws MethodError DeterministicDetector(static_model)

    # wrappers do not work with multiple unnamed detectors
    @test_throws ArgumentError CompositeDetector(unsupervised, supervised)
    @test_throws ArgumentError ProbabilisticDetector(unsupervised, supervised)
    @test_throws ArgumentError DeterministicDetector(unsupervised, supervised)

    # wrappers warn if both arguments and named arguments are provided
    @test_logs (:warn, r"Wrapping the single detector") CompositeDetector(unsupervised, s=supervised)
    @test_logs (:warn, r"Wrapping the single detector") ProbabilisticDetector(unsupervised, s=supervised)
    @test_logs (:warn, r"Wrapping the single detector") DeterministicDetector(unsupervised, s=supervised)
end

@testset "correct augmented_transform calls" begin
    test_a(m) = @test augmented_transform(m) == (m.report.scores, transform(m))
    test_b(m, X) = @test augmented_transform(m, X) == (m.report.scores, transform(m, X))
    test_c(m; rows=:) = @test augmented_transform(m; rows=rows) == (m.report.scores, transform(m; rows=rows))

    # make sure augmented_transform works as expected on all kinds of machines
    for m in [raw_machines..., machines...]
        test_a(m)
        test_b(m, X)
        test_c(m; rows=1:2)
    end
end

@testset "erroneous augmented_transform calls" begin
    u_not_fitted = machine(unsupervised, X)
    s_not_fitted = machine(supervised, X, y)
    c_not_fitted = machine(CompositeDetector(unsupervised), X)

    # not-yet-fitted machines
    @test_throws ErrorException augmented_transform(u_not_fitted)
    @test_throws ErrorException augmented_transform(s_not_fitted)
    @test_throws ErrorException augmented_transform(c_not_fitted)
end

@testset "transformers expected results" begin
    Xs = source(X)
    ys = source(y)

    # prepare learning network machines
    um = machine(MinimalUnsupervisedDetector(), Xs)
    sm = machine(MinimalSupervisedDetector(), Xs, ys)

    # prepare the transformers
    score_transformer = machine(ScoreTransformer())
    probabilistic_transformer = machine(ProbabilisticTransformer())
    deterministic_transformer = machine(DeterministicTransformer())

    # get the augmented scores
    u_scores = augmented_transform(um, Xs)
    s_scores = augmented_transform(sm, Xs)

    fit_transform(transformer, scores) = fit!(transform(transformer, scores))()
    fit_predict(transformer, scores) = fit!(predict(transformer, scores))()

    # test the transformed scores
    @test fit_transform(score_transformer, u_scores) isa OD.Scores
    @test fit_transform(score_transformer, s_scores) isa OD.Scores
    @test fit_transform(probabilistic_transformer, u_scores) isa OD.Scores
    @test fit_transform(probabilistic_transformer, s_scores) isa OD.Scores
    @test fit_transform(deterministic_transformer, u_scores) isa OD.Scores
    @test fit_transform(deterministic_transformer, s_scores) isa OD.Scores

    # test the predicted probabilities
    @test fit_predict(probabilistic_transformer, u_scores) isa UnivariateFiniteVector
    @test fit_predict(probabilistic_transformer, s_scores) isa UnivariateFiniteVector

    # test the predicted classes
    @test fit_predict(deterministic_transformer, u_scores) isa OD.Labels
    @test fit_predict(deterministic_transformer, s_scores) isa OD.Labels
end
