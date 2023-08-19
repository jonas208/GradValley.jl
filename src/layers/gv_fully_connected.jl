#=
Fully connected layer (Fc)
=#

@doc raw"""
    Fc(in_features::Int, out_features::Int; activation_function::Union{Nothing, AbstractString}=nothing, init_mode::AbstractString="default_uniform", use_bias::Bool=true)

A fully connected layer (sometimes also known as dense or linear). Apply a linear transformation (matrix multiplication) to the input signal with additional batch dimension.

# Arguments
- `in_features::Int`: the size of each input sample (*"number of input neurons"*)
- `out_features::Int`: the size of each output sample (*"number of output neurons"*)
- `activation_function::Union{Nothing, AbstractString}=nothing`: the element-wise activation function which will be applied to the output
- `init_mode::AbstractString="default_uniform"`: the initialization mode of the weights
    (can be `"default_uniform"`, `"default"`, `"kaiming_uniform"`, `"kaiming"`, `"xavier_uniform"` or `"xavier"`)
`use_bias::Bool=true`: if true, adds a learnable bias to the output

# Shapes
- Input: ``(in\_features, N)``
- Weight: ``(out\_features, in\_features)``
- Bias: ``(out\_features, )``
- Output: ``(out\_features, N)``

# Useful Fields/Variables
- `weight::AbstractArray{<: Real, 2}`: the learnable weights of the layer
- `bias::AbstractVector{<: Real}`: the learnable bias of the layer (used when `use_bias=true`)
- `weight_gradient::AbstractArray{<: Real, 2}`: the current gradients of the weights
- `bias_gradient::AbstractVector{<: Real}`: the current gradients of the bias

# Definition
The forward pass of a fully connected layer is given by the matrix multiplication between the weight matrix and the input vector 
(disregarding batch dimension and activation function):
- ``O = WI + B``
This operation can also be described by:
- ``o_{j} = \big(\sum_{k=1}^{in\_features} w_{j,k} \cdot i_{k}\big) + b_{j}``
*O* is the output vector, *I* the input vector, *W* the weight matrix and *B* the bias vector.
Visually interpreted, it means that each input neuron *i* is weighted with the corresponding weight *w* connecting the input neuron 
to the output neuron *o* where all the incoming signals are summed up.

# Examples
```julia-repl
# a fully connected layer with 784 input features and 120 output features
julia> m = Fc(784, 120)
# computing the output of the layer (with random inputs)
julia> input = rand(Float32, 784, 32)
julia> output = forward(m, input)
```
"""
mutable struct Fc <: AbstractParamLayer
    # characteristics of the layer
    in_features::Int
    out_features::Int
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Nothing, Function} # derivative of activation function, can be nothing
    use_bias::Bool
    # data
    weight::AbstractArray{<: Real, 2}
    bias::AbstractVector{<: Real}
    weight_gradient::AbstractArray{<: Real, 2}
    bias_gradient::AbstractVector{<: Real}
    # custom constructor
    function Fc(in_features::Int, out_features::Int; activation_function::Union{Nothing, AbstractString}=nothing, init_mode::AbstractString="default_uniform", use_bias::Bool=true)
        # setting up the activation function
        new_activation_function, df, gain = general_activation_function_init(activation_function)

        # initialize weight and bias
        weight_shape = (out_features, in_features)
        bias_shape = (out_features, )
        weight, bias = general_weight_and_bias_init(weight_shape, bias_shape, init_mode, gain)
        # default dtype is Float32 on the cpu
        weight = convert(Array{Float32, 2}, weight)
        bias = convert(Vector{Float32}, bias)
        # initialize gradient of weight and bias
        weight_gradient = zeros(Float32, weight_shape)
        bias_gradient = zeros(Float32, bias_shape)

        # create new instance/object
        new(in_features, 
            out_features, 
            new_activation_function, 
            df,
            use_bias, 
            weight,
            bias,
            weight_gradient,
            bias_gradient
        )
    end
end

function forward(fc_layer::Fc, input::AbstractArray{T, 2}) where T
    if fc_layer.use_bias
        bias = fc_layer.bias
    else
        if typeof(input) <: CuArray
            bias = CUDA.zeros(eltype(fc_layer.bias), size(fc_layer.bias))
        else
            bias = zeros(eltype(fc_layer.bias), size(fc_layer.bias))
        end
    end
    
    output_no_activation = Functional.fc_forward(input, fc_layer.weight, fc_layer.bias)
    if !isnothing(fc_layer.activation_function)
        output = fc_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    return output
end

function (fc_layer::Fc)(input::AbstractArray{T, 2}) where T
    return forward(fc_layer, input)
end

function forward(layer::Fc, input::TrackedArray{T, 2}) where T
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), fc_layer::Fc, input::AbstractArray{T, 2}) where T
    # doing the forward pass of the layer

    if fc_layer.use_bias
        bias = fc_layer.bias
    else
        if typeof(input) <: CuArray
            bias = CUDA.zeros(eltype(fc_layer.bias), size(fc_layer.bias))
        else
            bias = zeros(eltype(fc_layer.bias), size(fc_layer.bias))
        end
    end
    
    output_no_activation = Functional.fc_forward(input, fc_layer.weight, fc_layer.bias)
    if !isnothing(fc_layer.activation_function)
        output = fc_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    function forward_pullback(output_gradient::AbstractArray{T, 2}) where T
        # doing the backpropagation of the fc_layer

        if !isnothing(fc_layer.df)
            output_gradient = output_gradient .* fc_layer.df(output_no_activation)
        end

        input_gradient, weight_gradient, bias_gradient = Functional.fc_backward(output_gradient, input, fc_layer.weight)

        # update layer's gradients
        fc_layer.weight_gradient = fc_layer.weight_gradient + weight_gradient
        if fc_layer.use_bias
            fc_layer.bias_gradient = fc_layer.bias_gradient + bias_gradient
        end

        # defining gradients
        forward_gradient = NoTangent()
        fc_layer_gradient = NoTangent()

        return forward_gradient, fc_layer_gradient, input_gradient
    end

    return output, forward_pullback
end

# zero gradients function for a fully connected layer (Fc)
# resets the gradients of the given layer
function zero_gradients(fc_layer::Fc)
    fc_layer.weight_gradient .= zero(eltype(fc_layer.weight_gradient))
    fc_layer.bias_gradient .= zero(eltype(fc_layer.bias_gradient))
end