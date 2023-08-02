module Optimization

using ..Layers: AbstractContainer, AbstractParamLayer, extract_layers

# export all optimizers and their step! methods
export SGD, MSGD, Nesterov, Adam, step!
# export all loss functions
export mse_loss, mae_loss, bce_loss

#=
Internal functions, Internals
=#

# computes the mean of an arbitrary sized input array
function mean(x)
    return sum(x) / length(x)
end

include("optimization/gv_optimizers.jl")
include("optimization/gv_loss_functions.jl")

end # end of module "Optimization"