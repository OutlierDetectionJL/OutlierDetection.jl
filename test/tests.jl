using Combinatorics: permutations

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
const X, y = to_table(base), to_categorical(["normal", "normal", "outlier"])
const Xs, ys = source(X), source(y)

# learning network helpers
function surrogate_machine(base_detector, surrogate_detector, transformer, Xs, ys)
    source_machine = base_detector isa UnsupervisedDetector ?
                     machine(base_detector, Xs) :
                     machine(base_detector, Xs, ys)

    augmented_scores = augmented_transform(source_machine, Xs)

    # TODO: The source for ys in unsupervised models is currently necessary to enable evaluation, but
    # we could fix this in MLJBase
    machine(surrogate_detector, Xs, ys;
        predict=predict(transformer, augmented_scores),
        transform=last(augmented_scores),
        report=(scores=first(augmented_scores),))
end

function score_surrogate_machine(base_detector, Xs, ys)
    base_detector isa UnsupervisedDetector ?
    surrogate_machine(base_detector, OD.UnsupervisedDetector(), score_transformer, Xs, ys) :
    surrogate_machine(base_detector, OD.SupervisedDetector(), score_transformer, Xs, ys)
end

function probabilistic_surrogate_machine(base_detector, Xs, ys)
    base_detector isa UnsupervisedDetector ?
    surrogate_machine(base_detector, OD.ProbabilisticUnsupervisedDetector(), probabilistic_transformer, Xs, ys) :
    surrogate_machine(base_detector, OD.ProbabilisticSupervisedDetector(), probabilistic_transformer, Xs, ys)
end

function deterministic_surrogate_machine(base_detector, Xs, ys)
    base_detector isa UnsupervisedDetector ?
    surrogate_machine(base_detector, OD.DeterministicUnsupervisedDetector(), deterministic_transformer, Xs, ys) :
    surrogate_machine(base_detector, OD.DeterministicSupervisedDetector(), deterministic_transformer, Xs, ys)
end

# prepare the transformers
score_transformer = machine(ScoreTransformer())
probabilistic_transformer = machine(ProbabilisticTransformer())
deterministic_transformer = machine(DeterministicTransformer())

# detectors
unsupervised_detector = ODUnsupervisedDetector()
raw_unsupervised_detector = MMIUnsupervisedDetector()

supervised_detector = ODSupervisedDetector()
raw_supervised_detector = MMISupervisedDetector()

basic_unsupervised_detectors = [unsupervised_detector, raw_unsupervised_detector]
basic_supervised_detectors = [supervised_detector, raw_supervised_detector]

# raw machines
unsupervised_machines = [fit!(machine(detector, X)) for detector in basic_unsupervised_detectors]
supervised_machines = [fit!(machine(detector, X, y)) for detector in basic_supervised_detectors]

# surrogate machines
unsupervised_surrogate = score_surrogate_machine(unsupervised_detector, Xs, ys) |> fit!
raw_unsupervised_surrogate = score_surrogate_machine(raw_unsupervised_detector, Xs, ys) |> fit!
supervised_surrogate = score_surrogate_machine(supervised_detector, Xs, ys) |> fit!
raw_supervised_surrogate = score_surrogate_machine(raw_supervised_detector, Xs, ys) |> fit!

# probabilistic surrogate machines
unsupervised_probabilistic_surrogate = probabilistic_surrogate_machine(unsupervised_detector, Xs, ys) |> fit!
raw_unsupervised_probabilistic_surrogate = probabilistic_surrogate_machine(raw_unsupervised_detector, Xs, ys) |> fit!
supervised_probabilistic_surrogate = probabilistic_surrogate_machine(supervised_detector, Xs, ys) |> fit!
raw_supervised_probabilistic_surrogate = probabilistic_surrogate_machine(raw_supervised_detector, Xs, ys) |> fit!

# deterministic surrogate machines
unsupervised_deterministic_surrogate = deterministic_surrogate_machine(unsupervised_detector, Xs, ys) |> fit!
raw_unsupervised_deterministic_surrogate = deterministic_surrogate_machine(raw_unsupervised_detector, Xs, ys) |> fit!
supervised_deterministic_surrogate = deterministic_surrogate_machine(supervised_detector, Xs, ys) |> fit!
raw_supervised_deterministic_surrogate = deterministic_surrogate_machine(raw_supervised_detector, Xs, ys) |> fit!

score_surrogate_machines = [
    unsupervised_surrogate,
    raw_unsupervised_surrogate,
    supervised_surrogate,
    raw_supervised_surrogate]

probabilistic_surrogate_machines = [
    unsupervised_probabilistic_surrogate,
    raw_unsupervised_probabilistic_surrogate,
    supervised_probabilistic_surrogate,
    raw_supervised_probabilistic_surrogate]

deterministic_surrogate_machines = [
    unsupervised_deterministic_surrogate,
    raw_unsupervised_deterministic_surrogate,
    supervised_deterministic_surrogate,
    raw_supervised_deterministic_surrogate]

# surrogate detectors
@from_network unsupervised_surrogate mutable struct CustomUnsupervisedDetector end
@from_network raw_unsupervised_surrogate mutable struct RawCustomUnsupervisedDetector end
@from_network supervised_surrogate mutable struct CustomSupervisedDetector end
@from_network raw_supervised_surrogate mutable struct RawCustomSupervisedDetector end

surrogate_unsupervised = CustomUnsupervisedDetector()
raw_surrogate_unsupervised = RawCustomUnsupervisedDetector()
surrogate_supervised = CustomSupervisedDetector()
raw_surrogate_supervised = RawCustomSupervisedDetector()

# composite machines
unsupervised_detectors = [basic_unsupervised_detectors..., surrogate_unsupervised, raw_surrogate_unsupervised]
unsupervised_detectors = [unsupervised_detectors..., map(CompositeDetector, unsupervised_detectors)...]

supervised_detectors = [basic_supervised_detectors..., surrogate_supervised, raw_surrogate_supervised]
supervised_detectors = [supervised_detectors..., map(CompositeDetector, supervised_detectors)...]

# create composite detectors from raw detectors and already wrapped detectors
detectors = [unsupervised_detectors..., supervised_detectors...]

# all possible pairs of machines
detector_permutations = permutations(detectors, 2)

fit_composite(composite, permutations) = [fit!(machine(composite(d1=d1, d2=d2), X, y)) for (d1, d2) in permutations]

score_composites = fit_composite(CompositeDetector, detector_permutations)
probabilistic_composites = fit_composite(ProbabilisticDetector, detector_permutations)
deterministic_composites = fit_composite(DeterministicDetector, detector_permutations)

raw_machines = [unsupervised_machines..., supervised_machines...]
score_machines = [score_surrogate_machines..., score_composites...]
probabilistic_machines = [probabilistic_surrogate_machines..., probabilistic_composites...]
deterministic_machines = [deterministic_surrogate_machines..., deterministic_composites...]

surrogate_machines = [score_surrogate_machines..., probabilistic_surrogate_machines...,
    deterministic_surrogate_machines...]
composite_machines = [score_composites..., probabilistic_composites..., deterministic_composites...]
all_machines = [raw_machines..., score_machines..., probabilistic_machines..., deterministic_machines...]

@testset "normalization, combination and classification" begin
    minmax_test = scores -> last(scale_minmax(scores)) # extract test scores
    unify_test = scores -> last(scale_unify(scores)) # extract test scores
    classify = scores -> last(classify_quantile(DEFAULT_THRESHOLD)(scale_minmax(scores))) # extract test scores
    raw_proba(detector, data) = from_univariate_finite.(predict(detector, data))
    raw_class(detector, data) = from_categorical.(predict(detector, data))

    @testset "raw scores" begin
        test_scoring(scores, data) = begin
            for pred in [transform(m, data) for m in [raw_machines..., surrogate_machines...]]
                @test last(scores) == pred
            end
        end

        # Tests expected results on simple score vectors
        test_scoring(scores0, X_test0)
        test_scoring(scores1, X_test1)
        test_scoring(scores2, X_test2)
        test_scoring(scores3, X_test3)
        test_scoring(scores4, X_test4)
        test_scoring(scores5, X_test5)
    end

    @testset "normalized scores" begin
        test_scoring(scores, data, labels) = begin
            for pred in [minmax_test(scores),
                [transform(m, data) for m in composite_machines]...,
                [raw_proba(m, data) for m in probabilistic_machines]...]
                @test labels == pred
            end
        end

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
        test_classification(scores, data, labels) = begin
            for pred in [classify(scores),
                [predict(m, data) for m in deterministic_composites]...,
                [raw_class(m, data) for m in deterministic_composites]...]
                @test labels == pred
            end
        end

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
        @test normal_count(y) == 2
        @test outlier_count(y) == 1
    end
end

@testset "wrappers result in expected types" begin
    # unsupervised only
    composite_unsupervised_detectors = map(CompositeDetector, unsupervised_detectors)
    probabilistic_unsupervised_detectors = map(ProbabilisticDetector, unsupervised_detectors)
    deterministic_unsupervised_detectors = map(DeterministicDetector, unsupervised_detectors)

    for detector in composite_unsupervised_detectors
        @test detector isa UnsupervisedDetectorComposite
    end

    for detector in probabilistic_unsupervised_detectors
        @test detector isa ProbabilisticUnsupervisedDetectorComposite
    end

    for detector in deterministic_unsupervised_detectors
        @test detector isa DeterministicUnsupervisedDetectorComposite
    end

    # supervised only
    composite_supervised_detectors = map(CompositeDetector, supervised_detectors)
    probabilistic_supervised_detectors = map(ProbabilisticDetector, supervised_detectors)
    deterministic_supervised_detectors = map(DeterministicDetector, supervised_detectors)

    for detector in composite_supervised_detectors
        @test detector isa SupervisedDetectorComposite
    end

    for detector in probabilistic_supervised_detectors
        @test detector isa ProbabilisticSupervisedDetectorComposite
    end

    for detector in deterministic_supervised_detectors
        @test detector isa DeterministicSupervisedDetectorComposite
    end

    # mixed supervised/unsupervised
    for unsupervised_detector in [supervised_detectors..., composite_supervised_detectors...]
        for supervised_detector in [unsupervised_detectors..., composite_unsupervised_detectors...]
            @test (CompositeDetector(sup=supervised_detector, uns=unsupervised_detector) isa
                   SupervisedDetectorComposite)
            @test (ProbabilisticDetector(sup=supervised_detector, uns=unsupervised_detector) isa
                   ProbabilisticSupervisedDetectorComposite)
            @test (DeterministicDetector(sup=supervised_detector, uns=unsupervised_detector) isa
                   DeterministicSupervisedDetectorComposite)
        end
    end
end

@testset "wrappers property access" begin
    normalization_strategy = scale_unify
    combination_strategy = combine_max
    classification_strategy = classify_quantile(0.5)

    for m in getproperty.(composite_machines, :model)
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

    for m in getproperty.(deterministic_composites, :model)
        initial_classification_strategy = m.classify
        m.classify = classification_strategy
        @test m.classify == classification_strategy
        # reset back to original value
        m.classify = initial_classification_strategy
    end

    # wrappers throw an error if a property does not exist
    for m in getproperty.(all_machines, :model)
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
    @test_throws ArgumentError CompositeDetector(unsupervised_detector, supervised_detector)
    @test_throws ArgumentError ProbabilisticDetector(unsupervised_detector, supervised_detector)
    @test_throws ArgumentError DeterministicDetector(unsupervised_detector, supervised_detector)

    # wrappers warn if both arguments and named arguments are provided
    @test_logs (:warn, r"Wrapping") CompositeDetector(unsupervised_detector, s=supervised_detector)
    @test_logs (:warn, r"Wrapping") ProbabilisticDetector(unsupervised_detector, s=supervised_detector)
    @test_logs (:warn, r"Wrapping") DeterministicDetector(unsupervised_detector, s=supervised_detector)
end

@testset "correct augmented_transform calls" begin
    test_implicit(m) = @test augmented_transform(m) == (m.report.scores, transform(m))
    test_concrete(m, X) = @test augmented_transform(m, X) == (m.report.scores, transform(m, X))
    test_source(m, Xs) = @test augmented_transform(m, Xs)() == (m.report.scores, transform(m, Xs)())
    test_rows(m; rows=:) = @test augmented_transform(m; rows=rows) == (m.report.scores, transform(m; rows=rows))

    # make sure augmented_transform works as expected on all kinds of machines
    for m in all_machines
        test_implicit(m)
        test_concrete(m, X)
        test_source(m, Xs)
        test_rows(m; rows=1:2)
    end
end

@testset "erroneous augmented_transform calls" begin
    u_not_fitted = machine(unsupervised_detector, X)
    s_not_fitted = machine(supervised_detector, X, y)
    c_not_fitted = machine(CompositeDetector(unsupervised_detector), X)

    # not-yet-fitted machines
    @test_throws ErrorException augmented_transform(u_not_fitted)
    @test_throws ErrorException augmented_transform(s_not_fitted)
    @test_throws ErrorException augmented_transform(c_not_fitted)
end

@testset "evaluation works as expected" begin
    train, test = [1, 2], [3]
    evaluate_machine(detector) =
        @test evaluate!(detector; measure=misclassification_rate, resampling=[(train, test)]).measurement[1] == 0

    for mach in [probabilistic_machines..., deterministic_machines...]
        evaluate_machine(mach)
    end
end
