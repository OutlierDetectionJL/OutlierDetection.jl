function test_detector(detector)
    @testset "$detector" begin
        # create data
        dim = trainDim
        ntrain = 1000
        ntest = 10
        data_train = rand(dim, ntrain)
        label_train = rand((-1,0,1), ntrain)
        data_test = rand(dim, ntest)

        # learn model
        model = fit(detector, data_train, label_train)
        scores = model.scores

        # calculate test scores
        scores_train, scores_test = transform(detector, model, data_test)

        @testset "fit and transform yield equal scores" begin
            @test scores_train ≈ scores
        end

        @testset "scores have appropriate dimensions" begin
            @test length(scores) == ntrain
            @test length(scores_train) == ntrain
            @test length(scores_test) == ntest
        end

        @testset "scores have appropriate values" begin
            @test all(-Inf .< scores .< Inf)
            @test all(-Inf .< scores_train .< Inf)
            @test all(-Inf .< scores_test .< Inf)
        end
    end
end
