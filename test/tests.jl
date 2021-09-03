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
const test0, test1, test2, test3, test4, test5 = map(to_table, test_scores)
const train = to_table(base)
const train_labels = to_categorical(fill(missing, length(base)))

@testset "normalization, combination and classification" begin
    # fake detectors
    supervised = MinimalSupervised()
    unsupervised = MinimalUnsupervised()

    # fake machines
    up = machine(probabilistic(unsupervised), train) |> fit!
    sp = machine(probabilistic(supervised), train, train_labels) |> fit!
    ud = machine(deterministic(unsupervised), train) |> fit!
    sd = machine(deterministic(supervised), train, train_labels) |> fit!

    minmax_test = score_tuple -> scale_minmax(score_tuple...)[2]
    unify_test = score_tuple -> scale_unify(score_tuple...)[2]
    classify = score_tuple -> classify_percentile(DEFAULT_THRESHOLD)(scale_minmax(score_tuple...)...)[2]

    @testset "scores" begin
        raw_predict(detector, data) = OutlierDetection.raw_scores.(predict(detector, data))
        to_scores(data) = [transform(up, data), transform(sp, data), raw_predict(up, data), raw_predict(sp, data)]
        test_scoring(scores, data, labels) = @test all((labels,) .== [minmax_test(scores), to_scores(data)...])

        # Tests expected results on simple score vectors
        test_scoring(scores0, test0, [0.0, 0.5, 1.0])
        test_scoring(scores1, test1, [0.0, 0.5])
        test_scoring(scores2, test2, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        test_scoring(scores3, test3, [0.0, 0.0, 0.0])
        test_scoring(scores4, test4, [0.5, 0.5, 0.5])
        test_scoring(scores5, test5, [1.0, 1.0, 1.0])

        # Min-Max normalization and unify should not change sort order
        @test sortperm(minmax_test(scores0)) == sortperm(unify_test(scores0))
        @test sortperm(minmax_test(scores1)) == sortperm(unify_test(scores1))
        @test sortperm(minmax_test(scores2)) == sortperm(unify_test(scores2))
        @test sortperm(minmax_test(scores3)) == sortperm(unify_test(scores3))
        @test sortperm(minmax_test(scores4)) == sortperm(unify_test(scores4))
        @test sortperm(minmax_test(scores5)) == sortperm(unify_test(scores5))
    end

    @testset "classification" begin
        to_classes(data) = [transform(ud, data), transform(sd, data), predict(ud, data), predict(sd, data)]
        test_classification(scores, data, labels) = @test all((labels,) .== [classify(scores), to_classes(data)...])

        # Test expected classification results with given threshold
        test_classification(scores0, test0, ["normal", "normal", "outlier"])
        test_classification(scores1, test1, ["normal", "normal"])
        test_classification(scores2, test2, ["normal", "normal", "normal", "normal", "outlier", "outlier", "outlier"])
        test_classification(scores3, test3, ["normal", "normal", "normal"])
        test_classification(scores4, test4, ["normal", "normal", "normal"])
        test_classification(scores5, test5, ["outlier", "outlier", "outlier"])
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
