#=
Softmax layer for nd-inputs
=#

@doc raw"""
    Softmax(; dims=1)

A softmax activation function layer (probably mostly used at the "end" of a classifier model). Apply the softmax function to an n-dimensional input array.
The softmax will be computed along the given dimensions (`dims`), so every slice along these dimensions will sum to 1.

!!! note
    Note that this is the only activation function in form of a layer. All other activation functions can be used with the `activation_function::AbstractString`
    keyword argument nearly every layer provides. All the activation functions which can be used that way are simple element-wise activation functions.
    Softmax is currently the only non-element-wise activation function. Besides, it is very important to be able to select a specific dimension along the 
    softmax should be computed. That would also not work well with the use of simple keyword argument taking only a string which is the name of the function.

# Arguments
- `dims=1`: the dimensions along the softmax will be computed (so every slice along these dimensions will sum to 1)

# Shapes
- Input: ``(*)``, where ``*`` means any number of dimensions
- Output: ``(*)`` (same shape as input)

# Definition
The softmax function converts a vector of real numbers into a probability distribution.
The softmax function is defined as:
```math
\begin{align*}
softmax(x_i) = \frac{e^{x_i}}{\sum_{j}e^{x_j}} = \frac{exp(x_i)}{\sum_{j}exp(x_j)}
\end{align*}
```
Where *X* is the input array (slice). Note that the ``x_j`` values are taken from each slice individually along the specified dimension.
So each slice along the specified dimension will sum to 1. All values in the output are between 0 and 1.

# Examples
```julia-repl
# the softmax will be computed along the first dimension
julia> m = Softmax(dims=1)
# computing the output of the layer 
# (with random input data which could represent a batch of unnormalized output values from a classifier)
julia> input = rand(10, 32)
julia> output = forward(m, input)
# summing up the values in the output along the first dimension result in a batch of 32 ones
julia> sum(output, dims=1)
1x32 Matrix{Float64}:
1.0 1.0 ... 1.0
```
"""
mutable struct Softmax <: AbstractNonParamLayer
    # characteristics of the layer
    dims # the softmax will be calculated along these dimensions
    # custom constructor
    function Softmax(; dims=1)
        # create new instance/object
        new(dims)
    end
end

function forward(softmax_layer::Softmax, input::AbstractArray{T, N}) where {T, N}
    output = Functional.softmax_forward(input, dims=softmax_layer.dims)

    return output
end

function (softmax_layer::Softmax)(input::AbstractArray{T, N}) where {T, N}
    return forward(softmax_layer, input)
end

function forward(layer::Softmax, input::AbstractArray{T, N}) where {T, N}
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), softmax_layer::Softmax, input::AbstractArray{T, N}) where {T, N}
    # doing the forward pass of the layer

    output = Functional.softmax_forward(input, dims=softmax_layer.dims)

    function forward_pullback(output_gradient::AbstractArray{T, N}) where {T, N}
        # doing the backpropagation of the softmax_layer

        input_gradient = Functional.softmax_backward(output_gradient, output, dims=softmax_layer.dims)

        # defining gradients
        forward_gradient = NoTangent()
        softmax_layer_gradient = NoTangent()

        return forward_gradient, softmax_layer_gradient, input_gradient
    end

    return output, forward_pullback
end