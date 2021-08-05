@testset "normalization, combination and classification" begin
    # Create some test data of scores
    base = [1., 2, 3]
    scores0 = (base, base) # train and test equal
    scores1 = (base, [1., 2]) # len(test) < len(train)
    scores2 = (base, [-Inf, 0., 1, 2, 3, 4, Inf]) # len(test) > len(train)
    scores3 = (base, [1.,1,1]) # len(test) = len(train)
    scores4 = (base, [2.,2,2])
    scores5 = (base, [3.,3,3])
    threshold = 2/3

    normalize_test = scores -> normalize(scores)[2]
    unify_test = scores -> unify(scores)[2]

    @testset "scores have appropriate values" begin
        # Tests expected results on simple score vectors
        @test normalize(base) == normalize_test(scores0) == [0.0, 0.5, 1.0]
        @test normalize_test(scores1) == [0.0, 0.5]
        @test normalize_test(scores2) == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        @test normalize_test(scores3) == [0.0, 0.0, 0.0]
        @test normalize_test(scores4) == [0.5, 0.5, 0.5]
        @test normalize_test(scores5) == [1.0, 1.0, 1.0]

        # Tests including combination of scores (mean)
        @test combine_mean(normalize_test.([scores3, scores4])...) == [0.25, 0.25, 0.25]
        @test combine_mean(normalize_test.([scores3, scores5])...) == [0.5, 0.5, 0.5]
        @test combine_mean(normalize_test.([scores4, scores5])...) == [0.75, 0.75, 0.75]

        @test combine_median(normalize_test.([scores3, scores4])...) == [0.25, 0.25, 0.25]
        @test combine_median(normalize_test.([scores3, scores5])...) == [0.5, 0.5, 0.5]
        @test combine_median(normalize_test.([scores4, scores5])...) == [0.75, 0.75, 0.75]

        @test combine_max(normalize_test.([scores3, scores4])...) == [0.5, 0.5, 0.5]
        @test combine_max(normalize_test.([scores3, scores5])...) == [1.0, 1.0, 1.0]
        @test combine_max(normalize_test.([scores4, scores5])...) == [1.0, 1.0, 1.0]

        # Min-Max normalization and unify should not change sort order
        @test sortperm(normalize(base)) == sortperm(unify(base))
        @test sortperm(normalize_test(scores1)) == sortperm(unify_test(scores1))
        @test sortperm(normalize_test(scores2)) == sortperm(unify_test(scores2))
        @test sortperm(normalize_test(scores3)) == sortperm(unify_test(scores3))
        @test sortperm(normalize_test(scores4)) == sortperm(unify_test(scores4))
        @test sortperm(normalize_test(scores5)) == sortperm(unify_test(scores5))
    end

    @testset "classify instances as expected" begin
        @test classify(threshold, normalize(base)) == [1, 1, -1]
        @test classify(threshold, normalize(scores1)) == [1, 1]
        @test classify(threshold, normalize(scores2)) == [1, 1, 1, 1, -1, -1, -1]
        @test classify(threshold, normalize(scores3)) == [1, 1, 1]
        @test classify(threshold, normalize(scores4)) == [1, 1, 1]
        @test classify(threshold, normalize(scores5)) == [-1, -1, -1]
    end
end
