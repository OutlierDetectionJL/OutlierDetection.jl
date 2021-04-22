using Random:MersenneTwister
using MLJBase

function test_detector(detector)
    @testset "$detector" begin
        # create data
        is_supervised = isa(detector, SupervisedDetector)
        
        # specify test parameters
        rng = MersenneTwister(0)
        dim = trainDim
        fraction_train = 0.7
        n_samples = 1000
        n_train = Int(n_samples * fraction_train)
        n_test = n_samples - n_train

        # test all different input formats
        y = rand(rng, (-1,0,1), n_samples)
        X_raw = rand(rng, dim, n_samples)
        X_mat = collect(X_raw')
        X_df = table(X_mat)
        train, test  = partition(eachindex(y), fraction_train, rng=rng);

        # Raw detector with matrix transpose input
        model = is_supervised ? OutlierDetection.fit(detector, X_raw[:, train], y[train]) :
                                OutlierDetection.fit(detector, X_raw[:, train])
        train_raw, test_raw = OutlierDetection.transform(detector, model, X_raw[:, test])

        # MLJ with matrix input
        detector_mat = is_supervised ? machine(detector, X_mat, y) : machine(detector, X_mat)
        fit!(detector_mat, rows=train)
        train_mat, test_mat = MLJBase.predict(detector_mat, rows=test)

        # MLJ with matrix table
        detector_df = is_supervised ? machine(detector, X_df, y) : machine(detector, X_df)
        fit!(detector_df, rows=train)
        train_df, test_df = MLJBase.predict(detector_df, rows=test)

        @testset "scores have appropriate dimensions" begin
            @test length(train_raw) == length(train_mat) == length(train_df) == n_train
            @test length(test_raw) == length(test_mat) == length(test_df) == n_test
        end

        @testset "scores have appropriate values" begin
            @test all(-Inf .< train_raw .< Inf)
            @test all(-Inf .< train_mat .< Inf)
            @test all(-Inf .< train_df .< Inf)

            @test all(-Inf .< test_raw .< Inf)
            @test all(-Inf .< test_mat .< Inf)
            @test all(-Inf .< test_df .< Inf)
        end
    end
end
