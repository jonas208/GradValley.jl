#=
Reshape layer (Reshape, mostly used as a flatten layer)
=#

@doc raw"""
    Reshape(out_shape; activation_function::Union{Nothing, AbstractString}=nothing)

A reshape layer (probably mostly used as a flatten layer). Reshape the input signal (effects all dimensions except the batch dimension).

# Arguments
- `out_shape`: the target output size (the output has the same data as the input and must have the same number of elements)
- `activation_function::Union{Nothing, AbstractString}=nothing`: the element-wise activation function which will be applied to the output

# Shapes
- Input: ``(*, N)``, where * means any number of dimensions
- Output: ``(out\_shape..., N)``

# Definition
This layer uses the standard [reshape function](https://docs.julialang.org/en/v1/base/arrays/#Base.reshape) inbuilt in Julia.

# Examples
```julia-repl
# flatten the input of size 28*28*1 to a vector of length 784 (each plus batch dimension of course)
julia> m = Reshape((784, ))
# computing the output of the layer (with random inputs)
julia> input = rand(Float32, 28, 28, 1, 32)
julia> output = forward(m, input)
julia> size(output) # specified size plus batch dimension
(784, 32)
```
"""
mutable struct Reshape <: AbstractNonParamLayer # maybe use parametric types for specifing number of dimension of arrays for concrete types
    # characteristics of the layer
    out_shape::Tuple
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Nothing, Function} # derivative of activation function, can be nothing
    # custom constructor
    function Reshape(out_shape; activation_function::Union{Nothing, AbstractString}=nothing)
        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # create new instance/object
        new(out_shape,
            new_activation_function, 
            df
        )
    end
end

function forward(reshape_layer::Reshape, input::AbstractArray{T, N}) where {T, N}
    output_no_activation = Functional.reshape_forward(input, reshape_layer.out_shape)
    if !isnothing(reshape_layer.activation_function)
        output = reshape_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    return output
end

function (reshape_layer::Reshape)(input::AbstractArray{T, N}) where {T, N}
    return forward(reshape_layer, input)
end

function forward(layer::Reshape, input::TrackedArray{T, N}) where {T, N}
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), reshape_layer::Reshape, input::AbstractArray{T, N}) where {T, N}
    # doing the forward pass of the layer 

    output_no_activation = Functional.reshape_forward(input, reshape_layer.out_shape)
    if !isnothing(reshape_layer.activation_function)
        output = reshape_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    function forward_pullback(output_gradient::AbstractArray{T, N}) where {T, N}
        # doing the backpropagation of the reshape_layer

        if !isnothing(reshape_layer.df)
            output_gradient = output_gradient .* reshape_layer.df(output_no_activation)
        end

        input_gradient = Functional.reshape_backward(output_gradient, input)

        # defining gradients
        forward_gradient = NoTangent()
        reshape_layer_gradient = NoTangent()

        return forward_gradient, reshape_layer_gradient, input_gradient
    end

    return output, forward_pullback
end