using Statistics: mean, median

function instance_mean(scores)
    dropdims(mean(scores, dims=1:ndims(scores) - 1), dims=1) 
end

const _ae_params = """    encoder::Chain
Transforms the input data into a latent state with a fixed shape.

    decoder::Chain
Transforms the latent state back into the shape of the input data.

    batchsize::Integer
The number of samples to work through before updating the internal model parameters.

    epochs::Integer
The number of passes of the entire training dataset the machine learning algorithm has completed. 

    shuffle::Bool
If `shuffle=true`, shuffles the observations each time iterations are re-started, else no shuffling is performed.

    partial::Bool
If `partial=false`, drops the last mini-batch if it is smaller than the batchsize.

    opt::Any
Any Flux-compatibale optimizer, typically a `struct`  that holds all the optimiser parameters along with a definition of
`apply!` that defines how to apply the update rule associated with the optimizer."""

const _ae_loss = """    loss::Function
The loss function used to calculate the reconstruction error, see <https://fluxml.ai/Flux.jl/stable/models/losses/>
for examples."""
