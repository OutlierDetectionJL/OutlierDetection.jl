using Flux: train!, params
using IterTools:ncycle
using Statistics:mean
using LinearAlgebra:norm

"""
    ESAD(encoder = Chain()
         decoder = Chain()
         batchsize = 32
         epochs = 1
         shuffle = false
         partial = true
         opt = ADAM()
         λ1 = 1
         λ2 = 1
         noise = identity)

End-to-End semi-supervised anomaly detection algorithm similar to DeepSAD, but without the pretraining phase. The
algorithm was published by Huang et al., see [1].

Parameters
----------
$_ae_params

    λ1::Real
Weighting parameter of the norm loss, which minimizes the empirical variance and thus minimizes entropy.

    λ2::Real
Weighting parameter of the assistent loss function to define the consistency between the two encoders.

    noise::Function (AbstractArray{T} -> AbstractArray{T})
A function to be applied to a batch of input data to add noise, see [1] for an explanation.

Examples
--------
$(_score_supervised("ESAD"))

References
----------
[1] Huang, Chaoqin; Ye, Fei; Zhang, Ya; Wang, Yan-Feng; Tian, Qi (2020): ESAD: End-to-end Deep Semi-supervised Anomaly
Detection.
"""
MMI.@mlj_model mutable struct ESAD <: SupervisedDetector
    encoder::Chain = Chain()
    decoder::Chain = Chain()
    batchsize::Integer = 32::(_ > 0)
    epochs::Integer = 1::(_ > 0)
    shuffle::Bool = false
    partial::Bool = true
    opt::Any = ADAM()
    λ1::Number = 1
    λ2::Number = 1
    noise::Function = identity
end

struct ESADModel <: Model
    chain::Chain
end

function fit(detector::ESAD, X::Data, y::Labels)::Fit
    loader = DataLoader((X, y), batchsize=detector.batchsize, shuffle=detector.shuffle, partial=detector.partial)

    # Create the autoencoder // TODO: deepcopy the encoder/decoder?
    model = Chain(detector.encoder, detector.decoder, detector.encoder)

    # Precalculate dimensions
    dims = ndims(X)

    # Calculate loss as described in the paper
    train!((x, y) -> _esadloss(model[1:2](x), x, model(x), model[1](x), y, detector.λ1, detector.λ2, detector.noise,
    dims), params(model), ncycle(loader, detector.epochs), detector.opt)

    # Score as described in the paper
    scores = _esadscore(model[1:2](X), X, model(X), dims)
    Fit(ESADModel(model), scores)
end

@score function score(_::ESAD, model::Fit, X::Data)::Result
    _esadscore(model.chain[1:2](X), X, model.chain(X), ndims(X))
end

function _esadloss(x̂, x, ẑ, z, y, λ1, λ2, noise, dims)
    # The esad loss function is based on the distance to the hypersphere center. The inverse distance is used if an
    # example is an outlier and labeled samples are weighted using the hyperparameter eta
    rec_outlier = dropdims(sum((x̂ .- noise(x)).^2, dims=1:dims - 1), dims=1)
    rec_normal = dropdims(sum((x̂ .- x).^2, dims=1:dims - 1), dims=1)
    origin_dist = map(norm, eachslice(ẑ; dims=ndims(ẑ)))

    # reconstruction error (optimize mutual information of latent)
    l_rec_semi = ifelse.(y .== -1, rec_outlier, rec_normal) |> mean

    # minimize distance to origin (thus minimize entropy for normal and maximize entropy for outliers)
    l_norm_semi = ifelse.(y .== -1, origin_dist .^ -1, origin_dist) |> mean

    # similarity constraint of latent embeddings
    l_ass_semi = mse(ẑ, z)

    # combine the (weighted) loss terms (scalars)
    l_rec_semi + λ1 * l_norm_semi + λ2 * l_ass_semi
end

function _esadscore(x̂, x, ẑ, dims)
    # Combines reconstruction loss and distance to the origin
    origin_dist = map(norm, eachslice(ẑ; dims=ndims(ẑ)))
    rec_loss = dropdims(mean((x̂ .- x).^2, dims=1:dims - 1), dims=1)
    rec_loss .+ origin_dist
end
