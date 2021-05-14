@testset "Evaluator" begin
    # Create some test data of scores
    base = [1., 2, 3]
    scores0 = (base, base) # train and test equal
    scores1 = (base, [1., 2]) # len(test) < len(train)
    scores2 = (base, [-Inf, 0., 1, 2, 3, 4, Inf]) # len(test) > len(train)
    scores3 = (base, [1.,1,1]) # len(test) = len(train)
    scores4 = (base, [2.,2,2])
    scores5 = (base, [3.,3,3])

    _cls = Class(outlier_fraction=0.3)
    _nor = Score(normalize=normalize)
    _uni = Score(normalize=unify)

    _norm = (scores...) -> detect(_nor, scores...)
    _unify = (scores...) -> detect(_uni, scores...)
    _classify = (scores...) -> detect(_cls, scores...)
    _norm_machine = (scores...) -> pdf.(transform(machine(_nor), scores...), -1)
    _unify_machine = (scores...) -> pdf.(transform(machine(_uni), scores...), -1)
    _classify_machine = (scores...) -> transform(machine(_cls), scores...)

    @testset "scores have appropriate values" begin
        # Tests on simple score vectors
        @test normalize(base) == _norm(scores0) == _norm_machine(scores0) == [0.0, 0.5, 1.0]
        @test _norm(scores1) == _norm_machine(scores1) == [0.0, 0.5]
        @test _norm(scores2) == _norm_machine(scores2) == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        @test _norm(scores3) == _norm_machine(scores3) == [0.0, 0.0, 0.0]
        @test _norm(scores4) == _norm_machine(scores4) == [0.5, 0.5, 0.5]
        @test _norm(scores5) == _norm_machine(scores5) == [1.0, 1.0, 1.0]

        @test unify(base) == _unify(scores0) == _unify_machine(scores0)
        @test _unify(scores1) == _unify_machine(scores1)
        @test _unify(scores2) == _unify_machine(scores2)
        @test _unify(scores3) == _unify_machine(scores3)
        @test _unify(scores4) == _unify_machine(scores4)
        @test _unify(scores5) == _unify_machine(scores5)

        # Tests including combination of scores (mean)
        @test _norm(scores3, scores4) == _norm(scores4, scores3) == [0.25, 0.25, 0.25] ==
              _norm_machine(scores3, scores4) == _norm_machine(scores4, scores3)
        @test _norm(scores3, scores5) == _norm(scores5, scores3) == [0.5, 0.5, 0.5] ==
              _norm_machine(scores3, scores5) == _norm_machine(scores5, scores3)
        @test _norm(scores4, scores5) == _norm(scores5, scores4) == [0.75, 0.75, 0.75] ==
              _norm_machine(scores4, scores5) == _norm_machine(scores5, scores4)
    end

    @testset "classifies instances as expected" begin
        @test classify(0.3, base) == _classify(scores0) == _classify_machine(scores0) == [1, 1, -1]
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
