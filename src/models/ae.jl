using Flux: Chain, train!, params
using Flux.Losses:mse
using Flux.Data:DataLoader
using Flux.Optimise:ADAM
using IterTools:ncycle
using Statistics:mean

"""
    AE(encoder= Chain(),
       decoder = Chain(),
       batchsize= 32,
       epochs = 1,
       shuffle = false,
       partial = true,
       opt = ADAM(),
       loss = mse)

Calculate the anomaly score of an instance based on the reconstruction loss of an autoencoder, see [1] for an
explanation of auto encoders.

Parameters
----------
$_ae_params

$_ae_loss

Examples
--------
$(_score_unsupervised("AE"))

References
----------
[1] Aggarwal, Charu C. (2017): Outlier Analysis.
"""
MMI.@mlj_model struct AE <: UnsupervisedDetector
    encoder::Chain = Chain()
    decoder::Chain = Chain()
    batchsize::Integer = 32
    epochs::Integer = 1
    shuffle::Bool = false
    partial::Bool = true
    opt::Any = ADAM()
    loss::Function = mse
end

struct AEModel <: Model
    chain::Chain
end

function fit(detector::AE, X::Data)::Fit
    loader = DataLoader(X, batchsize=detector.batchsize, shuffle=detector.shuffle, partial=detector.partial)

    # Create the autoencoder
    model = Chain(detector.encoder, detector.decoder)

    # train the neural network model
    train!(x -> detector.loss(model(x), x), params(model), ncycle(loader, detector.epochs), detector.opt)

    scores = detector.loss(model(X), X, agg=instance_mean)
    Fit(AEModel(model), scores)
end

@score function score(detector::AE, model::Fit, X::Data)::Result
    detector.loss(model.chain(X), X, agg=instance_mean)
end
