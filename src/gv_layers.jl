module Layers
using ..Functional
using CUDA

# export all layers and nearly all functions
export Conv, ConvTranspose, BatchNorm, MaxPool, AdaptiveMaxPool, AvgPool, AdaptiveAvgPool, Fc, Reshape, Softmax, Identity, SequentialContainer, GraphContainer
export forward, backward, zero_gradients, trainmode!, testmode!, summarize_model, clean_model_from_backward_information!, module_to_eltype_device!
# export Conv, forward, backward, GraphContainer, zero_gradients, module_to_eltype_device!

# abstract layer types 
abstract type AbstractLayer end
abstract type AbstractParamLayer <: AbstractLayer end
abstract type AbstractNonParamLayer <: AbstractLayer end
abstract type AbstractContainer <: AbstractLayer end

# dependencies for auto diff
using ExprTools
using SpecialFunctions
using LinearAlgebra
using Statistics
using SparseArrays
# ChainRules provides a collection of common derivative rules
using ChainRules
using ChainRules: rrule, unthunk, NoTangent
# hepls to overload all the functions for which rules exists
using ChainRulesOverloadGeneration
# resolve conflicts while this code exists in both.
const on_new_rule = ChainRulesOverloadGeneration.on_new_rule
const refresh_rules = ChainRulesOverloadGeneration.refresh_rules
# custom rules for the layers
import Base: +, -, *, /, ^

# auto diff
include("layers/gv_auto_diff.jl")
include("layers/gv_auto_diff_rules.jl")
# activation and weight init
include("layers/gv_init_tools.jl")
# layers
include("layers/gv_convolution.jl")
include("layers/gv_pooling.jl")
include("layers/gv_fully_connected.jl")
include("layers/gv_reshape_flatten.jl")
include("layers/gv_softmax.jl")
include("layers/gv_batch_normalization.jl")
include("layers/gv_identity.jl")
# containers
include("layers/gv_containers.jl")
# utils and pretty-printing
include("layers/gv_utilities.jl")
include("layers/gv_show.jl")

#=
Documentation for functions with many but very similar methods
=#

"""
    forward(layer, input::AbstractArray{T, N}) where {T, N}

The forward function for computing the output of a module. For every layer/container, an individual method exists.
However, all these methods work exactly the same. They all take the layer/container as the first argument and the input data
as the second argument. The output is returned. 

# Examples
```julia-repl
# define some layers and containers
julia> layer = Conv(3, 6, (5, 5))
julia> container = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# create some random input data
julia> layer_input = rand(50, 50, 3, 32)
julia> container_input = rand(1000, 32)
# compute the output of the modules
julia> layer_output = forward(layer, layer_input)
julia> container_output = forward(container, container_input)
```
"""
function forward end

"""
    zero_gradients(layer_or_container)

Resets the gradients of a layer or a container (any kind of module with trainable parameters). 

There only exists methods for layers with parameters, however, if a container without layers with trainable parameters is given, NO error will be thrown.
So if the given container contains layers with trainable parameters/weights, regardless of whether they are nested somewhere in a submodule or not, 
the gradients of all these layers at once will be reset.
"""
function zero_gradients end

"""
    trainmode!(batch_norm_layer_or_container)

Switches the mode of the given batch normalization layer or container to training mode. See [Normalization](@ref)

If the given container contains batch normalization layers (regardless of whether they are nested somewhere in a submodule or not), 
the mode of all these layers at once will be switched to training mode.
"""
function trainmode! end

"""
    testmode!(batch_norm_layer_or_container)

Switches the mode of the given batch normalization layer or container to test mode. See [Normalization](@ref)

If the given container contains batch normalization layers (regardless of whether they are nested somewhere in a submodule or not), 
the mode of all these layers at once will be switched to test mode.
"""
function testmode! end

end # end of module "Layers"