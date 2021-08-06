function test_detector(detector)
    @testset "$detector" begin
        is_supervised = isa(detector, SupervisedDetector)

        # raw detector with matrix input
        raw = is_supervised ? fit(detector, X_raw[:, train], y[train]) :
                              fit(detector, X_raw[:, train])
        raw_train = raw.scores
        raw_test = score(detector, raw, X_raw[:, test])
        raw_train_norm, raw_test_norm = detector.normalize(raw_train, raw_test)
        raw_train_labels = detector.classify(detector.threshold, raw_train_norm)
        raw_test_labels = detector.classify(detector.threshold, raw_train_norm, raw_test_norm)
 
        # raw detector with table input
        raw_df = is_supervised ? fit(detector, table(X_mat[train, :]), y[train]) :
                                 fit(detector, table(X_mat[train, :]))
        raw_df_train = raw_df.scores
        raw_df_test = score(detector, raw_df, table(X_mat[test, :]))
        raw_df_train_norm, raw_df_test_norm = detector.normalize(raw_df_train, raw_df_test)
        raw_df_train_labels = detector.classify(detector.threshold, raw_df_train_norm)
        raw_df_test_labels = detector.classify(detector.threshold, raw_df_train_norm, raw_df_test_norm)

        # MLJ with matrix input
        mlj_mat = is_supervised ? machine(detector, X_mat, y) : machine(detector, X_mat)
        fit!(mlj_mat, rows=train)
        mlj_mat_train = report(mlj_mat).scores
        mlj_mat_test = transform(mlj_mat, rows=test)
        mlj_mat_train_norm = predict(mlj_mat, rows=train)
        mlj_mat_test_norm = predict(mlj_mat, rows=test)
        mlj_mat_train_labels = predict_mode(mlj_mat, rows=train)
        mlj_mat_test_labels = predict_mode(mlj_mat, rows=test)

        # MLJ with table input
        mlj_df = is_supervised ? machine(detector, X_df, y) : machine(detector, X_df)
        fit!(mlj_df, rows=train)
        mlj_df_train = report(mlj_df).scores
        mlj_df_test = transform(mlj_df, rows=test)
        mlj_df_train_norm = predict(mlj_df, rows=train)
        mlj_df_test_norm = predict(mlj_df, rows=test)
        mlj_df_train_labels = predict_mode(mlj_df, rows=train)
        mlj_df_test_labels = predict_mode(mlj_df, rows=test)

        @testset "outputs have appropriate dimensions" begin
            @test length(raw_train) == length(raw_df_train) == length(mlj_df_train) == length(mlj_mat_train) == n_train
            @test length(raw_test) == length(raw_df_test) == length(mlj_df_test) == length(mlj_mat_test) == n_test
        end

        @testset "raw scores have appropriate values" begin
            @test all(-Inf .< raw_train .< Inf)
            @test all(-Inf .< raw_df_train .< Inf)
            @test all(-Inf .< mlj_df_train .< Inf)
            @test all(-Inf .< mlj_mat_train .< Inf)

            @test all(-Inf .< raw_test .< Inf)
            @test all(-Inf .< raw_df_test .< Inf)
            @test all(-Inf .< mlj_df_test .< Inf)
            @test all(-Inf .< mlj_mat_test .< Inf)
        end

        @testset "normalized scores have appropriate values" begin
            to_scores(x) = pdf(x, OutlierDetection.CLASS_OUTLIER)
            @test all(0 .<= raw_train_norm .<= 1)
            @test all(0 .<= raw_df_train_norm .<= 1)
            @test all(0 .<= to_scores.(mlj_df_train_norm) .<= 1)
            @test all(0 .<= to_scores.(mlj_mat_train_norm) .<= 1)

            @test all(0 .<= raw_test_norm .<= 1)
            @test all(0 .<= raw_df_test_norm .<= 1)
            @test all(0 .<= to_scores.(mlj_df_test_norm) .<= 1)
            @test all(0 .<= to_scores.(mlj_mat_test_norm) .<= 1)
        end

        @testset "labels have appropriate values" begin
            in_labels(x) = x in (OutlierDetection.CLASS_NORMAL, OutlierDetection.CLASS_OUTLIER)
            @test all(in_labels.(raw_train_labels))
            @test all(in_labels.(raw_df_train_labels))
            @test all(in_labels.(mlj_df_train_labels))
            @test all(in_labels.(mlj_mat_train_labels))

            @test all(in_labels.(raw_test_labels))
            @test all(in_labels.(raw_df_test_labels))
            @test all(in_labels.(mlj_df_test_labels))
            @test all(in_labels.(mlj_mat_test_labels))
        end
    end
end
