#=
Batch normalization layer for 4d-inputs (BatchNorm)
=#

@doc raw"""
    BatchNorm(num_features::Int; epsilon::Real=1e-05, momentum::Real=0.1, affine::Bool=true, track_running_stats::Bool=true, activation_function::Union{Nothing, AbstractString}=nothing)

A batch normalization layer. Apply a batch normalization over a 4D input signal (a mini-batch of 2D inputs with additional channel dimension).

This layer has two modes: training mode and test mode. If `track_running_stats::Bool=true`, this layer behaves differently in the two modes.
During training, this layer always uses the currently calculated batch statistics. If `track_running_stats::Bool=true`, the running mean and variance are tracked
during training and will be used while testing. If `track_running_stats::Bool=false`, even in test mode, the currently calculated batch statistics are used.
The mode can be switched with [`trainmode!`](@ref) or [`testmode!`](@ref) respectively. The training mode is active by default.

# Arguments
- `num_features::Int`: the number of channels
- `epsilon::Real=1e-05`: a value added to the denominator for numerical stability
- `momentum::Real=0.1`: the value used for the running mean and running variance computation
- `affine::Bool=true`: if true, this layer uses learnable affine parameters/weights (``\gamma`` and ``\beta``)
- `track_running_stats::Bool=true`: if true, this layer tracks the running mean and variance during training and will use them for testing/evaluation, if false, such statistics are not tracked and, even in test mode, the batch statistics are always recalculated for each new input
- `activation_function::Union{Nothing, AbstractString}=nothing`: the element-wise activation function which will be applied to the output

# Shapes
- Input: ``(W, H, C, N)``
- ``\gamma`` Weight, ``\beta`` Bias: ``(C, )``
- Running Mean/Variance: ``(C, )``
- Output: ``(W, H, C, N)`` (same shape as input)

# Useful Fields/Variables
## Weights (used if `affine::Bool=true`)
- `weight::AbstractVector{<: Real}`: ``\gamma``, a learnabele parameter for each channel, initialized with ones
- `bias::AbstractVector{<: Real}`: ``\beta``, a learnabele parameter for each channel, initialized with zeros
## Gradients of weights (used if `affine::Bool=true`)
- `weight_gradient::AbstractVector{<: Real}`: the gradient of ``\gamma``
- `bias_gradient::AbstractVector{<: Real}`: the gradient of ``\beta``
## Running statistics (used if `rack_running_stats::Bool=true`)
- `running_mean::AbstractVector{<: Real}`: the continuously updated batch statistics of the mean
- `running_variance::AbstractVector{<: Real}`: the continuously updated batch statistics of the variance

# Definition
A batch normalization operation can be described as:
For input values over a mini-batch: ``\mathcal{B} = \{x_1, x_2, ..., x_n\}``
```math
\begin{align*}
y_i = \frac{x_i - \overline{\mathcal{B}}}{\sqrt{Var(\mathcal{B}) + \epsilon}} \cdot \gamma + \beta
\end{align*}
```
Where ``y_i`` is an output value and ``x_i`` an input value. ``\overline{\mathcal{B}}`` is the mean of the input values in ``\mathcal{B}`` and ``Var(\mathcal{B})`` 
is the variance of the input values in ``\mathcal{B}``.
Note that this definition is fairly general and not specified to 4D inputs.
In this case, the input values of ``\mathcal{B}`` are taken for each channel individually. 
So the mean and variance are calculated per channel over the mini-batch.

The update rule for the running statistics (running mean/variance) is:
```math
\begin{align*}
\hat{x}_{new} = (1 - momentum) \cdot \hat{x} + momentum \cdot x
\end{align*}
```
Where ``\hat{x}`` is the estimated statistic and ``x`` is the new observed value.
So ``\hat{x}_{new}`` is the new, updated estimated statistic.

# Examples
```julia-repl
# a batch normalization layer (3 channels) with learnabel parameters and continuously updated batch statistics for evaluation
julia> m = BatchNorm(3)
# the mode can be switched with trainmode! or testmode!
julia> trainmode!(m)
julia> testmode!(m)
# compute the output of the layer (with random inputs)
julia> input = rand(50, 50, 3, 32)
julia> output = forward(m, input)
```
"""
mutable struct BatchNorm <: AbstractParamLayer
    # characteristics of the layer
    num_features::Int
    epsilon::Real
    momentum::Real
    affine::Bool
    track_running_stats::Bool
    test_mode::Bool
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Nothing, Function} # derivative of activation function, can be nothing
    # learnabel parameters and their gradients
    weight::AbstractVector{<: Real}
    bias::AbstractVector{<: Real}
    weight_gradient::AbstractVector{<: Real}
    bias_gradient::AbstractVector{<: Real}
    # running statistics
    running_mean::AbstractVector{<: Real}
    running_variance::AbstractVector{<: Real}
    # custom constructor
    function BatchNorm(num_features::Int; epsilon::Real=1e-05, momentum::Real=0.1, affine::Bool=true, track_running_stats::Bool=true, activation_function::Union{Nothing, AbstractString}=nothing)
        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # default mode is training mode 
        test_mode = false

        # initialize weights (default dtype is Float32 on the cpu)
        weight = ones(Float32, num_features)
        bias = zeros(Float32, num_features)
        # initialize gradients of weights
        weight_gradient = zeros(Float32, num_features)
        bias_gradient = zeros(Float32, num_features)
        # initialize running statistics
        running_mean = zeros(Float32, num_features)
        running_variance = ones(Float32, num_features)

        # create new instance/object
        new(num_features,
            epsilon,
            momentum,
            affine,
            track_running_stats,
            test_mode,
            new_activation_function,
            df,
            weight,
            bias,
            weight_gradient,
            bias_gradient,
            running_mean,
            running_variance
        )
    end
end

function forward(batchnorm_layer::BatchNorm, input::AbstractArray{T, 4}) where T
    output_no_activation, running_mean, running_variance = Functional.batch_norm2d_forward(input, 
                                                           batchnorm_layer.weight, 
                                                           batchnorm_layer.bias, 
                                                           batchnorm_layer.track_running_stats, 
                                                           batchnorm_layer.running_mean, 
                                                           batchnorm_layer.running_variance, 
                                                           batchnorm_layer.test_mode, 
                                                           momentum=T(batchnorm_layer.momentum), 
                                                           epsilon=T(batchnorm_layer.epsilon))
    if !isnothing(batchnorm_layer.activation_function)
        output = batchnorm_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    # update running statistics
    batchnorm_layer.running_mean = running_mean
    batchnorm_layer.running_variance = running_variance

    return output
end

function (batchnorm_layer::BatchNorm)(input::AbstractArray{T, 4}) where T
    return forward(batchnorm_layer, input)
end

function forward(layer::BatchNorm, input::TrackedArray{T, 4}) where T
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), batchnorm_layer::BatchNorm, input::AbstractArray{T, 4}) where T
    # doing the forward pass of the layer

    output_no_activation, running_mean, running_variance = Functional.batch_norm2d_forward(input, 
                                                           batchnorm_layer.weight, 
                                                           batchnorm_layer.bias, 
                                                           batchnorm_layer.track_running_stats, 
                                                           batchnorm_layer.running_mean, 
                                                           batchnorm_layer.running_variance, 
                                                           batchnorm_layer.test_mode, 
                                                           momentum=T(batchnorm_layer.momentum), 
                                                           epsilon=T(batchnorm_layer.epsilon))
    if !isnothing(batchnorm_layer.activation_function)
        output = batchnorm_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    # update running statistics
    batchnorm_layer.running_mean = running_mean
    batchnorm_layer.running_variance = running_variance

    function forward_pullback(output_gradient::AbstractArray{T, 4}) where T
        # doing the backpropagation of the batchnorm_layer

        if !isnothing(batchnorm_layer.df)
            output_gradient = output_gradient .* batchnorm_layer.df(output_no_activation)
        end

        # which inputs are correct? old running statistics or new running statistics? just matter in test_mode? But in test_mode statistics are not updated?
        input_gradient, weight_gradient, bias_gradient = Functional.batch_norm2d_backward(output_gradient, 
                                                                                          output_no_activation, 
                                                                                          input, batchnorm_layer.weight, 
                                                                                          batchnorm_layer.bias, 
                                                                                          batchnorm_layer.track_running_stats, 
                                                                                          batchnorm_layer.running_mean, 
                                                                                          batchnorm_layer.running_variance, 
                                                                                          batchnorm_layer.test_mode,
                                                                                          epsilon=T(batchnorm_layer.epsilon))

        # update layer's gradients
        batchnorm_layer.weight_gradient = batchnorm_layer.weight_gradient + weight_gradient
        batchnorm_layer.bias_gradient = batchnorm_layer.bias_gradient + bias_gradient

        # defining gradients
        forward_gradient = NoTangent()
        batchnorm_layer_gradient = NoTangent()

        return forward_gradient, batchnorm_layer_gradient, input_gradient
    end

    return output, forward_pullback
end

# zero gradients function for a batch normalization layer (BatchNorm)
# resets the gradients of the given layer
function zero_gradients(batchnorm_layer::BatchNorm)
    batchnorm_layer.weight_gradient .= zero(eltype(batchnorm_layer.weight_gradient))
    batchnorm_layer.bias_gradient .= zero(eltype(batchnorm_layer.bias_gradient))
end

# changes the mode of the given batchnorm_layer to trainmode
function trainmode!(batchnorm_layer::BatchNorm)
    batchnorm_layer.test_mode = false
end

# changes the mode of the given batchnorm_layer to testmode
function testmode!(batchnorm_layer::BatchNorm)
    batchnorm_layer.test_mode = true
end