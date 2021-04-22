@testset "Classifier" begin
    # Create some test data of scores
    base = [1., 2, 3]
    scores1 = (base, [1., 2]) # len(test) < len(train)
    scores2 = (base, [-Inf, 0., 1, 2, 3, 4, Inf]) # len(test) > len(train)
    scores3 = (base, [1.,1,1]) # len(test) = len(train)
    scores4 = (base, [2.,2,2])
    scores5 = (base, [3.,3,3])

    # A classifier with `noclassify` simply normalizes and combines scores
    _score = (scores...) -> detect(Binarize(classify=nothing), scores...)
    _classify = (scores...) -> detect(Binarize(outlier_fraction=0.3), scores...)
    _score_machine = (scores...) -> transform(machine(Binarize(classify=nothing)), scores...)
    _classify_machine = (scores...) -> transform(machine(Binarize(outlier_fraction=0.3)), scores...)

    @testset "scores have appropriate values" begin
        # Tests on simple score vectors
        @test _score(scores1) == _score_machine(scores1) == [0.0, 0.5]
        @test _score(scores2) == _score_machine(scores2) == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        @test _score(scores3) == _score_machine(scores3) == [0.0, 0.0, 0.0]
        @test _score(scores4) == _score_machine(scores4) == [0.5, 0.5, 0.5]
        @test _score(scores5) == _score_machine(scores5) == [1.0, 1.0, 1.0]

        # Tests including combination of scores
        @test _score(scores3, scores4) == _score(scores4, scores3) == [0.25, 0.25, 0.25] ==
              _score_machine(scores3, scores4) == _score_machine(scores4, scores3)
        @test _score(scores3, scores5) == _score(scores5, scores3) == [0.5, 0.5, 0.5] ==
              _score_machine(scores3, scores5) == _score_machine(scores5, scores3)
        @test _score(scores4, scores5) == _score(scores5, scores4) == [0.75, 0.75, 0.75] ==
              _score_machine(scores4, scores5) == _score_machine(scores5, scores4)
    end

    @testset "classifies instances as expected" begin
        @test _classify(scores1) == _classify_machine(scores1) == [1, 1]
        @test _classify(scores2) == _classify_machine(scores2) == [1, 1, 1, 1, -1, -1, -1]
        @test _classify(scores3) == _classify_machine(scores3) == [1, 1, 1]
        @test _classify(scores4) == _classify_machine(scores4) == [1, 1, 1]
        @test _classify(scores5) == _classify_machine(scores5) == [-1, -1, -1]

        @test _classify(scores3, scores4) == _classify(scores4, scores3) == [1, 1, 1] ==
              _classify_machine(scores3, scores4) == _classify_machine(scores4, scores3)
        @test _classify(scores3, scores5) == _classify(scores5, scores3) == [1, 1, 1] ==
              _classify_machine(scores3, scores5) == _classify_machine(scores5, scores3)
        @test _classify(scores4, scores5) == _classify(scores5, scores4) == [-1, -1, -1] ==
              _classify_machine(scores4, scores5) == _classify_machine(scores5, scores4)
    end
end
