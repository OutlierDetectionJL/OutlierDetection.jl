# helper to create fake data from scores such that the detectors return the scores again
to_table(points) = reshape(repeat(points, length(points)), (length(points), length(points))) |> table
const base = [1.0, 2, 3]
const test_scores = [base, # train and test equal
    [1.0, 2], # len(test) < len(train)
    [-Inf, 0.0, 1, 2, 3, 4, Inf], # len(test) > len(train)
    [1.0, 1, 1],  # len(test) = len(train)
    [2.0, 2, 2],
    [3.0, 3, 3]]
const scores0, scores1, scores2, scores3, scores4, scores5 = map(t -> (base, t), test_scores)
const X_test0, X_test1, X_test2, X_test3, X_test4, X_test5 = map(to_table, test_scores)
const X = to_table(base)
const y = to_categorical(["normal", "normal", "outlier"])

# detectors
supervised = MinimalSupervisedDetector()
unsupervised = MinimalUnsupervisedDetector()
raw_supervised = MMISupervisedDetector()
raw_unsupervised = MMIUnsupervisedDetector()

# raw machines
unsupervised_machine = machine(unsupervised, X) |> fit!
raw_unsupervised_machine = machine(raw_unsupervised, X) |> fit!
supervised_machine = machine(supervised, X, y) |> fit!
raw_supervised_machine = machine(raw_supervised, X, y) |> fit!
raw_machines = [
    unsupervised_machine,
    raw_unsupervised_machine,
    supervised_machine,
    raw_supervised_machine]

transform(supervised_machine)
transform(unsupervised_machine)
transform(raw_supervised_machine)
transform(raw_unsupervised_machine)

# composite machines
unsupervised_composite = machine(CompositeDetector(unsupervised), X) |> fit!
raw_unsupervised_composite = machine(CompositeDetector(raw_unsupervised), X) |> fit!
supervised_composite = machine(CompositeDetector(supervised), X, y) |> fit!
raw_supervised_composite = machine(CompositeDetector(raw_supervised), X, y) |> fit!
unsupervised_probabilistic = machine(ProbabilisticDetector(unsupervised), X) |> fit!
raw_unsupervised_probabilistic = machine(ProbabilisticDetector(raw_unsupervised), X) |> fit!
supervised_probabilistic = machine(ProbabilisticDetector(supervised), X, y) |> fit!
raw_supervised_probabilistic = machine(ProbabilisticDetector(raw_supervised), X, y) |> fit!
unsupervised_deterministic = machine(DeterministicDetector(unsupervised), X) |> fit!
raw_unsupervised_deterministic = machine(DeterministicDetector(raw_unsupervised), X) |> fit!
supervised_deterministic = machine(DeterministicDetector(supervised), X, y) |> fit!
raw_supervised_deterministic = machine(DeterministicDetector(raw_supervised), X, y) |> fit!
composite_machines = [
    unsupervised_composite,
    raw_unsupervised_composite,
    supervised_composite,
    raw_supervised_composite,
    unsupervised_probabilistic,
    raw_unsupervised_probabilistic,
    supervised_probabilistic,
    raw_supervised_probabilistic,
    unsupervised_deterministic,
    raw_unsupervised_deterministic,
    supervised_deterministic,
    raw_supervised_deterministic]

# wrapped composite machines
unsupervised_probabilistic_composite =
    machine(ProbabilisticDetector(CompositeDetector(unsupervised)), X) |> fit!
raw_unsupervised_probabilistic_composite =
    machine(ProbabilisticDetector(CompositeDetector(raw_unsupervised)), X) |> fit!
supervised_probabilistic_composite =
    machine(ProbabilisticDetector(CompositeDetector(supervised)), X, y) |> fit!
raw_supervised_probabilistic_composite =
    machine(ProbabilisticDetector(CompositeDetector(raw_supervised)), X, y) |> fit!
unsupervised_deterministic_composite =
    machine(DeterministicDetector(CompositeDetector(unsupervised)), X) |> fit!
raw_unsupervised_deterministic_composite =
    machine(DeterministicDetector(CompositeDetector(raw_unsupervised)), X) |> fit!
supervised_deterministic_composite =
    machine(DeterministicDetector(CompositeDetector(supervised)), X, y) |> fit!
raw_supervised_deterministic_composite =
    machine(DeterministicDetector(CompositeDetector(raw_supervised)), X, y) |> fit!
wrapped_machines = [
    unsupervised_probabilistic_composite,
    raw_unsupervised_probabilistic_composite,
    supervised_probabilistic_composite,
    raw_supervised_probabilistic_composite,
    unsupervised_deterministic_composite,
    raw_unsupervised_deterministic_composite,
    supervised_deterministic_composite,
    raw_supervised_deterministic_composite]

# multiple detector composites
unsupervised_supervised_composite =
    machine(CompositeDetector(u=unsupervised, s=supervised), X, y) |> fit!
raw_unsupervised_supervised_composite =
    machine(CompositeDetector(u=raw_unsupervised, s=raw_supervised), X, y) |> fit!
raw_unsupervised_supervised_mixed_composite =
    machine(CompositeDetector(u=unsupervised, s=raw_supervised), X, y) |> fit!
unsupervised_supervised_probabilistic =
    machine(ProbabilisticDetector(u=unsupervised, s=supervised), X, y) |> fit!
raw_unsupervised_supervised_probabilistic =
    machine(ProbabilisticDetector(u=raw_unsupervised, s=raw_supervised), X, y) |> fit!
unsupervised_supervised_deterministic =
    machine(DeterministicDetector(u=unsupervised, s=supervised), X, y) |> fit!
raw_unsupervised_supervised_deterministic =
    machine(DeterministicDetector(u=raw_unsupervised, s=raw_supervised), X, y) |> fit!
combined_machines = [
    unsupervised_supervised_composite,
    raw_unsupervised_supervised_composite,
    raw_unsupervised_supervised_mixed_composite,
    unsupervised_supervised_probabilistic,
    raw_unsupervised_supervised_probabilistic,
    unsupervised_supervised_deterministic,
    raw_unsupervised_supervised_deterministic]

machines = [composite_machines..., wrapped_machines..., combined_machines...]
probabilistic_machines = [
    unsupervised_probabilistic,
    raw_unsupervised_probabilistic,
    supervised_probabilistic,
    raw_supervised_probabilistic,
    unsupervised_probabilistic_composite,
    raw_unsupervised_probabilistic_composite,
    supervised_probabilistic_composite,
    raw_supervised_probabilistic_composite,
    unsupervised_supervised_probabilistic,
    raw_unsupervised_supervised_probabilistic]
deterministic_machines = [
    unsupervised_deterministic,
    raw_unsupervised_deterministic,
    supervised_deterministic,
    raw_supervised_deterministic,
    unsupervised_deterministic_composite,
    raw_unsupervised_deterministic_composite,
    supervised_deterministic_composite,
    raw_supervised_deterministic_composite,
    unsupervised_supervised_deterministic,
    raw_unsupervised_supervised_deterministic]

@testset "normalization, combination and classification" begin
    minmax_test = scores -> last(scale_minmax(scores)) # extract test scores
    unify_test = scores -> last(scale_unify(scores)) # extract test scores
    classify = scores -> last(classify_quantile(DEFAULT_THRESHOLD)(scale_minmax(scores))) # extract test scores

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

    @testset "label utilities" begin
        @test normal_fraction(y) ≈ 2 / 3
        @test outlier_fraction(y) ≈ 1 / 3
        @test n_normal(y) == 2
        @test n_outlier(y) == 1
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
    classification_strategy = classify_quantile(0.5)

    for m in getproperty.(machines, :model)
        initial_normalization_strategy = m.normalize
        initial_combination_strategy = m.combine
        m.normalize = normalization_strategy
        m.combine = combination_strategy
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

@testset "transformers yield expected results" begin
    Xs = source(X)
    ys = source(y)

    # prepare learning network machines
    unsupervised_source_machine = machine(MinimalUnsupervisedDetector(), Xs)
    raw_unsupervised_source_machine = machine(MMIUnsupervisedDetector(), Xs)
    supervised_source_machine = machine(MinimalSupervisedDetector(), Xs, ys)
    raw_supervised_source_machine = machine(MMISupervisedDetector(), Xs, ys)

    # prepare the transformers
    score_transformer = machine(ScoreTransformer())
    probabilistic_transformer = machine(ProbabilisticTransformer())
    deterministic_transformer = machine(DeterministicTransformer())

    # get the augmented scores
    unsupservised_scores = augmented_transform(unsupervised_source_machine, Xs)
    raw_unsupervised_scores = augmented_transform(raw_unsupervised_source_machine, Xs)
    supervised_scores = augmented_transform(supervised_source_machine, Xs)
    raw_supervised_scores = augmented_transform(raw_supervised_source_machine, Xs)

    # helper functions
    fit_transform(transformer, scores) = fit!(transform(transformer, scores))()
    fit_predict(transformer, scores) = fit!(predict(transformer, scores))()

    # raw unsupervised transform
    @test fit_transform(score_transformer, unsupservised_scores) isa OD.Scores
    @test fit_transform(score_transformer, raw_unsupervised_scores) isa OD.Scores

    # raw supervised transform
    @test fit_transform(score_transformer, supervised_scores) isa OD.Scores
    @test fit_transform(score_transformer, raw_supervised_scores) isa OD.Scores

    # probabilistic unsupervised transform
    @test fit_transform(probabilistic_transformer, unsupservised_scores) isa OD.Scores
    @test fit_transform(probabilistic_transformer, raw_unsupervised_scores) isa OD.Scores

    # probabilistic supervised transform
    @test fit_transform(probabilistic_transformer, supervised_scores) isa OD.Scores
    @test fit_transform(probabilistic_transformer, raw_supervised_scores) isa OD.Scores

    # deterministic unsupervised transform
    @test fit_transform(deterministic_transformer, unsupservised_scores) isa OD.Scores
    @test fit_transform(deterministic_transformer, raw_unsupervised_scores) isa OD.Scores

    # deterministic supervised transform
    @test fit_transform(deterministic_transformer, supervised_scores) isa OD.Scores
    @test fit_transform(deterministic_transformer, raw_supervised_scores) isa OD.Scores

    # probabilistic unsupervised predict
    @test fit_predict(probabilistic_transformer, unsupservised_scores) isa UnivariateFiniteVector
    @test fit_predict(probabilistic_transformer, raw_unsupervised_scores) isa UnivariateFiniteVector

    # probabilistic supervised predict
    @test fit_predict(probabilistic_transformer, supervised_scores) isa UnivariateFiniteVector
    @test fit_predict(probabilistic_transformer, raw_supervised_scores) isa UnivariateFiniteVector

    # deterministic unsupervised predict
    @test fit_predict(deterministic_transformer, unsupservised_scores) isa OD.Labels
    @test fit_predict(deterministic_transformer, raw_unsupervised_scores) isa OD.Labels

    # deterministic supervised predict
    @test fit_predict(deterministic_transformer, supervised_scores) isa OD.Labels
    @test fit_predict(deterministic_transformer, raw_supervised_scores) isa OD.Labels
end

# create a simple detector learning network
Xs = source()
ys = source()

unsupervised_scores = augmented_transform(machine(unsupervised, Xs), Xs)
raw_unsupervised_scores = augmented_transform(machine(raw_unsupervised, Xs), Xs)

supervised_scores = augmented_transform(machine(supervised, Xs, ys), Xs)
raw_supervised_scores = augmented_transform(machine(raw_supervised, Xs, ys), Xs)

unsupervised_probabilistic_scores = predict(machine(ProbabilisticTransformer()), unsupervised_scores)
raw_unsupervised_probabilistic_scores = predict(machine(ProbabilisticTransformer()), raw_unsupervised_scores)

supervised_probabilistic_scores = predict(machine(ProbabilisticTransformer()), supervised_scores)
raw_supervised_probabilistic_scores = predict(machine(ProbabilisticTransformer()), raw_supervised_scores)

unsupervised_deterministic_scores = predict(machine(DeterministicTransformer()), unsupervised_scores)
raw_unsupervised_deterministic_scores = predict(machine(DeterministicTransformer()), raw_unsupervised_scores)

supervised_deterministic_scores = predict(machine(DeterministicTransformer()), supervised_scores)
raw_supervised_deterministic_scores = predict(machine(DeterministicTransformer()), raw_supervised_scores)

# TODO: The empty source for ys in unsupervised models is currently necessary to enable evaluation, but
# we could fix this in MLJBase
unsupervised_probabilistic_machine =
    machine(OD.ProbabilisticUnsupervisedDetector(), Xs, source(); predict=unsupervised_probabilistic_scores)
raw_unsupervised_probabilistic_machine =
    machine(OD.ProbabilisticUnsupervisedDetector(), Xs, source(); predict=raw_unsupervised_probabilistic_scores)

supervised_probabilistic_machine =
    machine(OD.ProbabilisticSupervisedDetector(), Xs, ys; predict=supervised_probabilistic_scores)
raw_supervised_probabilistic_machine =
    machine(OD.ProbabilisticSupervisedDetector(), Xs, ys; predict=raw_supervised_probabilistic_scores)

unsupervised_deterministic_machine =
    machine(OD.DeterministicUnsupervisedDetector(), Xs, source(); predict=unsupervised_deterministic_scores)
raw_unsupervised_deterministic_machine =
    machine(OD.DeterministicUnsupervisedDetector(), Xs, source(); predict=raw_unsupervised_deterministic_scores)

supervised_deterministic_machine =
    machine(OD.DeterministicSupervisedDetector(), Xs, ys; predict=supervised_deterministic_scores)
raw_supervised_deterministic_machine =
    machine(OD.DeterministicSupervisedDetector(), Xs, ys; predict=raw_supervised_deterministic_scores)

@testset "evaluation works as expected" begin
    train, test = [1, 2], [3]
    evaluate_detector(detector) =
        @test evaluate(detector, X, y; measure=misclassification_rate, resampling=[(train, test)]).measurement[1] == 0

    # evaluation of wrapped detectors
    evaluate_detector(ProbabilisticDetector(supervised))
    evaluate_detector(ProbabilisticDetector(raw_supervised))

    evaluate_detector(ProbabilisticDetector(unsupervised))
    evaluate_detector(ProbabilisticDetector(raw_unsupervised))

    evaluate_detector(DeterministicDetector(supervised))
    evaluate_detector(DeterministicDetector(unsupervised))

    evaluate_detector(DeterministicDetector(raw_supervised))
    evaluate_detector(DeterministicDetector(raw_unsupervised))

    # create the detectors from the network
    @from_network unsupervised_probabilistic_machine mutable struct CustomUnsupervisedProbabilisticDetector end
    @from_network raw_unsupervised_probabilistic_machine mutable struct RawCustomUnsupervisedProbabilisticDetector end

    @from_network supervised_probabilistic_machine mutable struct CustomSupervisedProbabilisticDetector end
    @from_network raw_supervised_probabilistic_machine mutable struct RawCustomSupervisedProbabilisticDetector end

    @from_network unsupervised_deterministic_machine mutable struct CustomUnsupervisedDeterministicDetector end
    @from_network raw_unsupervised_deterministic_machine mutable struct RawCustomUnsupervisedDeterministicDetector end

    @from_network supervised_deterministic_machine mutable struct CustomSupervisedDeterministicDetector end
    @from_network raw_supervised_deterministic_machine mutable struct RawCustomSupervisedDeterministicDetector end

    evaluate_detector(CustomUnsupervisedProbabilisticDetector())
    evaluate_detector(RawCustomUnsupervisedProbabilisticDetector())

    evaluate_detector(CustomSupervisedProbabilisticDetector())
    evaluate_detector(RawCustomSupervisedProbabilisticDetector())

    evaluate_detector(CustomUnsupervisedDeterministicDetector())
    evaluate_detector(RawCustomUnsupervisedDeterministicDetector())

    evaluate_detector(CustomSupervisedDeterministicDetector())
    evaluate_detector(RawCustomSupervisedDeterministicDetector())

    # TODO: Enable composites with custom detectors after https://github.com/JuliaAI/MLJBase.jl/pull/644 is merged
    # CompositeDetector(CustomUnsupervisedProbabilisticDetector)
end
