@doc raw"""
    Identity(; activation_function::Union{Nothing, AbstractString}=nothing)

An identity layer (can be used as an activation function layer). If no activation function is used, this layer does not change the signal in any way.
However, if an activation function is used, the activation function will be applied to the input element-wise. 

!!! tip
    This layer is helpful to apply an element-wise activation independent of a "normal" computational layer.

# Arguments
- `activation_function::Union{Nothing, AbstractString}=nothing`: the element-wise activation function which will be applied to the inputs

# Shapes
- Input: ``(*)``, where ``*`` means any number of dimensions
- Output: ``(*)`` (same shape as input)

# Definition
A placeholder identity operator, except the optional activation function, the input signal is not changed in any way.
If an activation function is used, the activation function will be applied to the input element-wise. 

# Examples
```julia-repl
# an independent relu activation
julia> m = Identity(activation_function="relu")
# computing the output of the layer (with random inputs)
julia> input = rand(Float32, 10, 32)
julia> output = forward(m, input)
```
"""
mutable struct Identity <: AbstractNonParamLayer
    # characteristics of the layer
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Nothing, Function} # derivative of activation function
    # custom constructor
    function Identity(; activation_function::Union{Nothing, AbstractString}=nothing)
        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # create new instance/object
        new(new_activation_function, df)
    end
end

function forward(identity_layer::Identity, input::AbstractArray{T, N}) where {T, N}
    output_no_activation = input
    if !isnothing(identity_layer.activation_function)
        output = identity_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    return output
end

function (identity_layer::Identity)(input::AbstractArray{T, N}) where {T, N}
    return forward(identity_layer, input)
end

function forward(layer::Identity, input::TrackedArray{T, N}) where {T, N}
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), identity_layer::Identity, input::AbstractArray{T, N}) where {T, N}
    # doing the forward pass of the layer

    output_no_activation = input
    if !isnothing(identity_layer.activation_function)
        output = identity_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    function forward_pullback(output_gradient::AbstractArray{T, N}) where {T, N}
        # doing the backpropagation of the identity_layer

        if !isnothing(identity_layer.df)
            output_gradient = output_gradient .* identity_layer.df(output_no_activation)
        end

        input_gradient = output_gradient

        # defining gradients
        forward_gradient = NoTangent()
        identity_layer_gradient = NoTangent()

        return forward_gradient, identity_layer_gradient, input_gradient
    end

    return output, forward_pullback
end