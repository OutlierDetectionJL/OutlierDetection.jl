using Flux: train!, params
using IterTools:ncycle
using Statistics:mean

"""
    DeepSAD(encoder = Chain()
            decoder = Chain()
            batchsize = 32
            epochs = 1
            shuffle = true
            partial = false
            opt = ADAM()
            loss = mse
            eta = 1
            eps = 1e-6
            callback = _ -> () -> ())

Deep Semi-Supervised Anomaly detection technique based on the distance to a hypersphere center as described in [1].

Parameters
----------
$_ae_params

$_ae_loss

    eta::Real
Weighting parameter for the labeled data; i.e. higher values of eta assign higher weight to labeled data in the svdd
loss function. For a sensitivity analysis of this parameter, see [1].

    eps::Real
Because the inverse distance used in the svdd loss can lead to division by zero, the parameters `eps` is added for
numerical stability.

    callback::Function
*Experimental parameter that might change*. A function to be called after the model parameters have been updated that
can call Flux's callback helpers, see <https://fluxml.ai/Flux.jl/stable/utilities/#Callback-Helpers-1>.

**Notice:** The parameters `batchsize`, `epochs`, `shuffle`, `partial`, `opt` and `callback` can also be tuples of size
2, specifying the corresponding values for (1) pretraining and (2) training; otherwise the same values are used for
pretraining and training.

Examples
--------
$(_transform_supervised("DeepSAD"))

References
----------
[1] Ruff, Lukas; Vandermeulen, Robert A.; Görnitz, Nico; Binder, Alexander; Müller, Emmanuel; Müller, Klaus-Robert;
Kloft, Marius (2019): Deep Semi-Supervised Anomaly Detection.
"""
mutable struct DeepSAD <: SupervisedDetector
    encoder::Chain
    decoder::Chain
    batchsize::Union{Integer,Tuple{Integer,Integer}}
    epochs::Union{Integer,Tuple{Integer,Integer}}
    shuffle::Union{Bool,Tuple{Bool, Bool}}
    partial::Union{Bool,Tuple{Bool, Bool}}
    opt::Any
    loss::Function
    eta::Number
    eps::Number
    callback
    function DeepSAD(;encoder::Chain = Chain(), decoder::Chain = Chain(), batchsize=32, epochs=1, shuffle=false,
        partial=true, opt=ADAM(), loss=mse, eta=1, eps=1e-6, callback=_ -> () -> ())

        # unify all possible tuples to tuples
        tuplify = t -> isa(t, Tuple) ? t : (t, t)
        batchsize, epochs, shuffle, partial, opt, callback =
            map(tuplify, (batchsize, epochs, shuffle, partial, opt, callback))

        new(encoder, decoder, batchsize, epochs, shuffle, partial, opt, loss, eta, eps, callback)
    end
end

struct DeepSADModel <: DetectorModel
    chain::Chain
    center::AbstractArray
end

function fit(detector::DeepSAD, X::Data, y::Labels)::Tuple{DeepSADModel, Scores}
    makeLoader = i -> DataLoader((X, y), batchsize=detector.batchsize[i], shuffle=detector.shuffle[i],
        partial=detector.partial[i])
    loaderPretrain = makeLoader(1)
    loaderTrain = makeLoader(2)

    # Create the autoencoder
    model = Chain(detector.encoder, detector.decoder)

    # pretraining (train the autoencoder based on a reconstruction loss)
    train!((x, _) -> detector.loss(model(x), x), params(model), ncycle(loaderPretrain, detector.epochs[1]),
                                      detector.opt[1]; cb=detector.callback[1](model))

    # Only use normal data and unlabeled data to calculate hypersphere center
    dims = ndims(X)
    nColons = X -> ntuple(_ -> :, dims - 1)
    prediction = detector.encoder(X[nColons(X)..., findall((y .== 0) .| (y .== 1))])
    center = dropdims(mean(prediction, dims=dims), dims=dims)

    # training based on the calculated hypersphere center
    train!((x, y) -> svddLoss(detector.encoder(x), center, y, detector.eta, detector.eps, dims),
           params(detector.encoder), ncycle(loaderTrain, detector.epochs[2]), detector.opt[2];
           cb=detector.callback[2](detector.encoder))

    scores = svddScore(detector.encoder(X), center, dims)
    DeepSADModel(model, center), scores
end

function transform(detector::DeepSAD, model::DeepSADModel, X::Data)::Scores
    svddScore(detector.encoder(X), model.center, ndims(X))
end

function svddLoss(latent, center, y, eta, eps, dims)
    # The svdd loss function is based on the distance to the hypersphere center. The inverse distance is used if an
    # example is an outlier and labeled samples are weighted using the hyperparameter eta.
    dist = dropdims(sum((latent .- center) .^ 2, dims=1:dims-1), dims=1)
    inverse_dist = eta .* ((dist .+ eps) .^ y)
    mean(ifelse.(y .== 0, dist, inverse_dist))
end

function svddScore(latent, center, dims)
    # Element-wise mean squared distance to the hypersphere center.
    dropdims(mean((latent .- center) .^ 2, dims=1:dims-1), dims=1)
end
