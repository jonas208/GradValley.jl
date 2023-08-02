#= 
Convolution-Layers (Conv & ConvTranspose)
=#

@doc raw"""
    Conv(in_channels::Int, out_channels::Int, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1, activation_function::Union{Nothing, AbstractString}=nothing, init_mode::AbstractString="default_uniform", use_bias::Bool=true)

A convolution layer. Apply a 2D convolution over an input signal with additional batch and channel dimensions.

# Arguments
- `in_channels::Int`: the number of channels in the input image
- `out_channels::Int`: the number of channels produced by the convolution
- `kernel_size::NTuple{2, Int}`: the size of the convolving kernel
- `stride::NTuple{2, Int}=(1, 1)`: the stride of the convolution
- `padding::NTuple{2, Int}=(0, 0)`: the zero padding added to all four sides of the input
- `dilation::NTuple{2, Int}=(1, 1)`: the spacing between kernel elements
- `groups::Int=1`: the number of blocked connections from input channels to output channels (in-channels and out-channels must both be divisible by groups)
- `activation_function::Union{Nothing, AbstractString}=nothing`: the element-wise activation function which will be applied to the output after the convolution 
- `init_mode::AbstractString="default_uniform"`: the initialization mode of the weights
    (can be `"default_uniform"`, `"default"`, `"kaiming_uniform"`, `"kaiming"`, `"xavier_uniform"` or `"xavier"`)
- `use_bias::Bool=true`: if true, adds a learnable bias to the output

# Shapes
- Input: ``(W_{in}, H_{in}, C_{in}, N)``
- Weight: ``(W_{w}, H_{w}, \frac{C_{in}}{groups}, C_{out})``
- Bias: ``(C_{out}, )``
- Output: ``(W_{out}, H_{out}, C_{out}, N)``
    - ``H_{out} = {\frac{H_{in} + 2 \cdot padding[1] - dilation[1] \cdot (H_w - 1) - 1}{stride[1]}} + 1``
    - ``W_{out} = {\frac{W_{in} + 2 \cdot padding[2] - dilation[2] \cdot (W_w - 1) - 1}{stride[2]}} + 1``

# Useful Fields/Variables
- `weight::AbstractArray{<: Real, 4}`: the learnable weight of the layer
- `bias::AbstractVector{<: Real}`: the learnable bias of the layer (used when `use_bias=true`)
- `weight_gradient::AbstractArray{<: Real, 4}`: the current gradient of the weight/kernel
- `bias_gradient::AbstractVector{<: Real}`: the current gradient of the bias

# Definition
For one group, a multichannel 2D convolution (disregarding batch dimension and activation function) can be described as:
- ``o_{c_{out}, y_{out}, x_{out}} = \big(\sum_{c_{in=1}}^{C_{in}}\sum_{y_w=1}^{H_{w}}\sum_{x_w=1}^{W_{w}} i_{c_{in}, y_{in}, x_{in}} \cdot w_{c_{out}, c_{in}, y_w, x_w}\big) + b_{c_{out}}``, where
    - ``y_{in} = y_{out} + (stride[1] - 1) \cdot (y_{out} - 1) + (y_w - 1) \cdot dilation[1]``
    - ``x_{in} = x_{out} + (stride[2] - 1) \cdot (x_{out} - 1) + (x_w - 1) \cdot dilation[2]``
*O* is the output array, *I* the input array, *W* the weight array and *B* the bias array.

# Examples
```julia-repl
# square kernels and fully default values of keyword arguments
julia> m = Conv(3, 6, (5, 5))
# non-square kernels and unequal stride and with padding as well as specified weight initialization mode
# (init_mode="kaiming" stands for kaiming weight initialization with normally distributed values)
julia> m = Conv(3, 6, (3, 5), stride=(2, 1), padding=(2, 1))
# non-square kernels and unequal stride and with padding, dilation and 3 groups
# (because groups=in_channels and out_channles is divisible by groups, it is even a depthwise convolution)
julia> m = Conv(3, 6, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), groups=3)
# computing the output of the layer (with random inputs)
julia> input = rand(50, 50, 3, 32)
julia> output = forward(m, input)
```
"""
mutable struct Conv <: AbstractParamLayer
    in_channels::Int
    out_channels::Int
    kernel_size::NTuple{2, Int}
    stride::NTuple{2, Int}
    padding::NTuple{2, Int}
    dilation::NTuple{2, Int}
    groups::Int
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Nothing, Function} # derivative of activation function, can be nothing
    use_bias::Bool
    # data 
    weight::AbstractArray{<: Real, 4}
    bias::AbstractVector{<: Real}
    weight_gradient::AbstractArray{<: Real, 4}
    bias_gradient::AbstractVector{<: Real}
    # custom constructor
    function Conv(in_channels::Int, out_channels::Int, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1, activation_function::Union{Nothing, AbstractString}=nothing, init_mode::AbstractString="default_uniform", use_bias::Bool=true)
        # setting up the activation function
        new_activation_function, df, gain = general_activation_function_init(activation_function)

        # checks if number of groups is a valid value
        if !(in_channels % groups == 0 && out_channels % groups == 0)
            error("GradValley: Conv: in_channels and out_channels must both be divisible by groups")
        end

        # initialize weight/kernel and bias 
        weight_shape = (kernel_size[2], kernel_size[1], convert(Int, in_channels / groups), out_channels)
        bias_shape = (out_channels, )
        weight, bias = general_weight_and_bias_init(weight_shape, bias_shape, init_mode, gain)
        # default dtype is Float32 on the cpu
        weight = convert(Array{Float32, 4}, weight)
        bias = convert(Vector{Float32}, bias)
        # initialize gradient of weight/kernel and bias
        weight_gradient = zeros(Float32, weight_shape)
        bias_gradient = zeros(Float32, bias_shape)

        # create new instance/object
        new(in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding,
            dilation, 
            groups,
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

function forward(conv_layer::Conv, input::AbstractArray{T, 4}) where T
    if conv_layer.use_bias
        bias = conv_layer.bias
    else
        if typeof(input) <: CuArray
            bias = CUDA.zeros(eltype(conv_layer.bias), size(conv_layer.bias))
        else
            bias = zeros(eltype(conv_layer.bias), size(conv_layer.bias))
        end
    end
    
    output_no_activation = Functional.convolution2d(input, conv_layer.weight, bias, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
    if !isnothing(conv_layer.activation_function)
        output = conv_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    return output
end

function (conv_layer::Conv)(input::AbstractArray{T, 4}) where T
    return forward(conv_layer, input)
end

function forward(layer::Conv, input::TrackedArray{T, 4}) where T
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), conv_layer::Conv, input::AbstractArray{T, 4}) where T
    # doing the forward pass of the layer

    if conv_layer.use_bias
        bias = conv_layer.bias
    else
        if typeof(input) <: CuArray
            bias = CUDA.zeros(eltype(conv_layer.bias), size(conv_layer.bias))
        else
            bias = zeros(eltype(conv_layer.bias), size(conv_layer.bias))
        end
    end
    
    output_no_activation = Functional.convolution2d(input, conv_layer.weight, bias, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
    if !isnothing(conv_layer.activation_function)
        output = conv_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    function forward_pullback(output_gradient::AbstractArray{T, 4}) where T
        # doing the backpropagation of the conv_layer

        # println(sum(output_gradient))
        if !isnothing(conv_layer.df)
            output_gradient = output_gradient .* conv_layer.df(output_no_activation)
        end
        # println(sum(output_gradient))

        input_gradient = Functional.convolution2d_data_backward(output_gradient, input, conv_layer.weight, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
        weight_gradient = Functional.convolution2d_filter_backward(output_gradient, input, conv_layer.weight, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
        if conv_layer.use_bias
            bias_gradient = Functional.convolution2d_bias_backward(output_gradient)
        end

        # update layer's gradients
        conv_layer.weight_gradient = conv_layer.weight_gradient + weight_gradient
        if conv_layer.use_bias
            conv_layer.bias_gradient = conv_layer.bias_gradient + bias_gradient
        end

        # defining gradients
        forward_gradient = NoTangent()
        conv_layer_gradient = NoTangent()

        return forward_gradient, conv_layer_gradient, input_gradient
    end

    return output, forward_pullback
end

# zero gradients function for a convolution layer (Conv)
# resets the gradients of the given layer
function zero_gradients(conv_layer::Conv)
    conv_layer.weight_gradient .= zero(eltype(conv_layer.weight_gradient))
    conv_layer.bias_gradient .= zero(eltype(conv_layer.bias_gradient))
end

@doc raw"""
    ConvTranspose(in_channels::Int, out_channels::Int, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), output_padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1, activation_function::Union{Nothing, AbstractString}=nothing, init_mode::AbstractString="default_uniform", use_bias::Bool=true)

A transpose convolution layer (also known as fractionally-strided convolution or deconvolution). Apply a 2D transposed convolution over an input signal with additional batch and channel dimensions.

# Arguments
- `in_channels::Int`: the number of channels in the input image
- `out_channels::Int`: the number of channels produced by the convolution
- `kernel_size::NTuple{2, Int}`: the size of the convolving kernel
- `stride::NTuple{2, Int}=(1, 1)`: the stride of the convolution
- `padding::NTuple{2, Int}=(0, 0)`: because transposed convolution can be seen as a partly (not true) inverse of convolution, padding means is this case to cut off the desired number of pixels on each side (instead of adding pixels)
- `output_padding::NTuple{2, Int}=(0, 0)`: additional size added to one side of each dimension in the output shape (note that output_padding is only used to calculate the output shape, but does not actually add zero-padding to the output)
- `dilation::NTuple{2, Int}=(1, 1)`: the spacing between kernel elements
- `groups::Int=1`: the number of blocked connections from input channels to output channels (in-channels and out-channels must both be divisible by groups)
- `activation_function::Union{Nothing, AbstractString}=nothing`: the element-wise activation function which will be applied to the output after the convolution 
- `init_mode::AbstractString="default_uniform"`: the initialization mode of the weights
    (can be `"default_uniform"`, `"default"`, `"kaiming_uniform"`, `"kaiming"`, `"xavier_uniform"` or `"xavier"`)
- `use_bias::Bool=true`: if true, adds a learnable bias to the output

# Shapes
- Input: ``( W_{in}, H_{in}, C_{in}, N)``
- Weight: ``(W_{w}, H_{w}, \frac{C_{out}}{groups}, C_{in})``
- Bias: ``(C_{out}, )``
- Output: ``(W_{out}, H_{out}, C_{out}, N)``, where
    - ``H_{out} = (H_{in} - 1) \cdot stride[1] - 2 \cdot padding[1] + dilation[1] \cdot (H_w - 1) + output\_padding[1] + 1``
    - ``W_{out} = (W_{in} - 1) \cdot stride[2] - 2 \cdot padding[2] + dilation[2] \cdot (W_w - 1) + output\_padding[2] + 1``

# Useful Fields/Variables
- `weight::AbstractArray{<: Real, 4}`: the learnable weight of the layer
- `bias::AbstractVector{<: Real}`: the learnable bias of the layer (used when `use_bias=true`)
- `weight_gradient::AbstractArray{<: Real, 4}`: the current gradient of the weight/kernel
- `bias_gradient::Vector{<: Real}`: the current gradient of the bias

# Definition
A transposed convolution can be seen as the gradient of a normal convolution with respect to its input. 
The forward pass of a transposed convolution is the backward pass of a normal convolution, so the forward pass
of a normal convolution becomes the backward pass of a transposed convolution (with respect to its input). 
For more detailed information, you can look at the [source code of (transposed) convolution](https://github.com/jonas208/GradValley.jl/blob/main/src/functional/gv_convolution.jl).
A nice looking visualization of (transposed) convolution can be found [here](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md).

# Examples
```julia-repl
# square kernels and fully default values of keyword arguments
julia> m = ConvTranspose(6, 3, (5, 5))
# upsampling an output from normal convolution like in GANS, Unet, etc.
julia> input = forward(Conv(3, 6, (5, 5)), rand(50, 50, 3, 32))
julia> output = forward(m, input)
# the size of the output of the transposed convolution is equal to the size of the original input of the normal convolution
julia> size(output)
(50, 50, 3, 32)
```
"""
mutable struct ConvTranspose <: AbstractParamLayer
    in_channels::Int
    out_channels::Int
    kernel_size::NTuple{2, Int}
    stride::NTuple{2, Int}
    padding::NTuple{2, Int}
    output_padding::NTuple{2, Int}
    dilation::NTuple{2, Int}
    groups::Int
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Nothing, Function} # derivative of activation function, can be nothing
    use_bias::Bool
    # data 
    weight::AbstractArray{<: Real, 4}
    bias::AbstractVector{<: Real}
    weight_gradient::AbstractArray{<: Real, 4}
    bias_gradient::AbstractVector{<: Real}
    # custom constructor
    function ConvTranspose(in_channels::Int, out_channels::Int, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), output_padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1, activation_function::Union{Nothing, AbstractString}=nothing, init_mode::AbstractString="default_uniform", use_bias::Bool=true)
        # setting up the activation function
        new_activation_function, df, gain = general_activation_function_init(activation_function)

        # checks if number of groups is a valid value
        if !(in_channels % groups == 0 && out_channels % groups == 0)
            error("GradValley: ConvTranspose: in_channels and out_channels must both be divisible by groups")
        end

        # splitting up the hyperparameters per dimension
        y_stride, x_stride = stride
        y_out_padding, x_out_padding = output_padding
        y_dilation, x_dilation = dilation

        # check if output padding has valid values
        if !(y_out_padding < y_stride || y_out_padding < y_dilation) || !(x_out_padding < x_stride || x_out_padding < x_dilation)
            error("GradValley: ConvTranspose: output_padding must be smaller than either stride or dilation, but got invalid values: y_output_padding: $y_out_padding x_output_padding: $x_out_padding y_stride: $y_stride x_stride: $x_stride y_dilation: $y_dilation x_dilation: $x_dilation")
        end

        # initialize weight/kernel and bias 
        weight_shape = (kernel_size[2], kernel_size[1], convert(Int, out_channels / groups), in_channels)
        bias_shape = (out_channels, )
        weight, bias = general_weight_and_bias_init(weight_shape, bias_shape, init_mode, gain) # weight_init may calculates the wrong fan_in and fan_out here because it assumes that it is a normal convolution, to do: add a kw-arg for fan_mode
        # default dtype is Float32 on the cpu
        weight = convert(Array{Float32, 4}, weight)
        bias = convert(Vector{Float32}, bias)
        # initialize gradient of weight/kernel and bias
        weight_gradient = zeros(Float32, weight_shape)
        bias_gradient = zeros(Float32, bias_shape)

        # create new instance/object
        new(in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding,
            output_padding,
            dilation, 
            groups,
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

function forward(conv_layer::ConvTranspose, input::AbstractArray{T, 4}) where T
    if conv_layer.use_bias
        bias = conv_layer.bias
    else
        if typeof(input) <: CuArray
            bias = CUDA.zeros(eltype(conv_layer.bias), size(conv_layer.bias))
        else
            bias = zeros(eltype(conv_layer.bias), size(conv_layer.bias))
        end
    end
    
    output_no_activation = Functional.deconvolution2d(input, conv_layer.weight, bias, stride=conv_layer.stride, padding=conv_layer.padding, output_padding=conv_layer.output_padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
    if !isnothing(conv_layer.activation_function)
        output = conv_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    return output
end

function (conv_layer::ConvTranspose)(input::AbstractArray{T, 4}) where T
    return forward(conv_layer, input)
end

function forward(layer::ConvTranspose, input::TrackedArray{T, 4}) where T
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), conv_layer::ConvTranspose, input::AbstractArray{T, 4}) where T
    # doing the forward pass of the layer

    if conv_layer.use_bias
        bias = conv_layer.bias
    else
        if typeof(input) <: CuArray
            bias = CUDA.zeros(eltype(conv_layer.bias), size(conv_layer.bias))
        else
            bias = zeros(eltype(conv_layer.bias), size(conv_layer.bias))
        end
    end
    
    output_no_activation = Functional.deconvolution2d(input, conv_layer.weight, bias, stride=conv_layer.stride, padding=conv_layer.padding, output_padding=conv_layer.output_padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
    if !isnothing(conv_layer.activation_function)
        output = conv_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    function forward_pullback(output_gradient::AbstractArray{T, 4}) where T
        # doing the backpropagation of the conv_layer
        
        if !isnothing(conv_layer.df)
            output_gradient = output_gradient .* conv_layer.df(output_no_activation)
        end

        input_gradient = Functional.deconvolution2d_data_backward(output_gradient, input, conv_layer.weight, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
        weight_gradient = Functional.deconvolution2d_filter_backward(output_gradient, input, conv_layer.weight, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
        if conv_layer.use_bias
            bias_gradient = Functional.deconvolution2d_bias_backward(output_gradient)
        end

        # update layer's gradients
        conv_layer.weight_gradient = conv_layer.weight_gradient + weight_gradient
        if conv_layer.use_bias
            conv_layer.bias_gradient = conv_layer.bias_gradient + bias_gradient
        end

        # defining gradients
        forward_gradient = NoTangent()
        conv_layer_gradient = NoTangent()

        return forward_gradient, conv_layer_gradient, input_gradient
    end

    return output, forward_pullback
end

# zero gradients function for a deconvolution layer (ConvTranspose)
# resets the gradients of the given layer
function zero_gradients(conv_layer::ConvTranspose)
    conv_layer.weight_gradient .= zero(eltype(conv_layer.weight_gradient))
    conv_layer.bias_gradient .= zero(eltype(conv_layer.bias_gradient))
end