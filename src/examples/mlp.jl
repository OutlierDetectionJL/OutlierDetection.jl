using Flux: Dense, BatchNorm, Chain, relu, sigmoid

"""
    DenseBNReLU(in, out; bias)

A linear layer followed by batch normalization.
"""
function DenseBNReLU(in:: Integer, out:: Integer; bias = false)
    dense = Dense(in, out; bias=bias)
    batchnorm = BatchNorm(out, relu)
    Chain(dense, batchnorm)
end

"""
    MLPEncoder(in, latent, hidden; bias)

A MLP encoder with variable number of hidden layers.
"""
function MLPEncoder(in:: Integer, latent:: Integer, hidden:: AbstractVector{T}; bias:: Bool = false) where {T <: Integer}
    hidden = append!([in], hidden)
    layers = [DenseBNReLU(hidden[i], hidden[i + 1], bias=bias) for i=1:length(hidden) - 1]
    Chain(layers..., Dense(hidden[end], latent; bias=bias))
end

"""
    MLPDecoder(in, latent, hidden; bias)

A MLP decoder with variable number of hidden layers.
"""
function MLPDecoder(in:: Integer, latent:: Integer, hidden:: AbstractVector{T}; bias:: Bool = false) where {T <: Integer} 
    hidden = append!([latent], hidden)
    layers = [DenseBNReLU(hidden[i], hidden[i + 1], bias=bias) for i=1:length(hidden) - 1]
    Chain(layers..., Dense(hidden[end], in, sigmoid; bias=bias))
end

"""
    MLPAutoEncoder(in, latent, hidden; bias)

A MLP auto-encoder with variable number of hidden layers.
"""
function MLPAutoEncoder(in:: Integer, latent:: Integer, hidden:: AbstractVector{T}; bias:: Bool = false) where {T <: Integer} 
    encoder = MLPEncoder(in, latent, hidden; bias)
    decoder = MLPDecoder(in, latent, reverse(hidden); bias)
    encoder, decoder
end
