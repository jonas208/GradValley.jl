module Layers
using ..Functional
# make Functional accessible via gv_functional
gv_functional = Functional

# export all layers and nearly all functions
export Conv, DepthwiseConv, Fc, BatchNorm2d, MaxPool, AdaptiveMaxPool, AvgPool, AdaptiveAvgPool, Reshape, SequentialContainer, Softmax
export forward, backward, zero_gradients, trainmode!, testmode!, summarize_model

#=
Internal functions, Internals
=#

function general_activation_function_init(activation_function::Union{Nothing, String})
    if isnothing(activation_function)
        new_activation_function = nothing
        df = 1
        gain = 1
    elseif activation_function == "relu"
        new_activation_function = gv_functional.relu
        df = gv_functional.d_relu
        gain = sqrt(2)
    elseif activation_function == "sigmoid"
        new_activation_function = gv_functional.sigmoid
        df = gv_functional.d_sigmoid
        gain = 1
    elseif activation_function == "softmax"
        new_activation_function = gv_functional.softmax
        df = gv_functional.d_softmax
        gain = 1
    elseif activation_function == "tanh"
        new_activation_function = gv_functional.gv_tanh
        df = gv_functional.d_tanh
        gain = 5 / 3
    else
        error("""GradValley: general_activation_function_init: activation_function must be one of the following:\n
            "relu",
            "sigmoid",
            "tanh",
            use the stand alone Softmax layer for softmax activation
        """)
    end

    return new_activation_function, df, gain
end

function general_weight_and_bias_init(weight_shape::NTuple{N1, Int} where N1, bias_shape::NTuple{N2, Int} where N2, init_mode::String, gain::Real)
    if init_mode == "default"
        weight = gv_functional.default_init(weight_shape, gain)
        bias = gv_functional.bias_init(bias_shape, weight_shape, gain)
    elseif init_mode == "default_uniform"
        weight = gv_functional.default_uniform_init(weight_shape, gain)
        bias = gv_functional.bias_uniform_init(bias_shape, weight_shape, gain)
    elseif init_mode == "kaiming"
        weight = gv_functional.kaiming_init(weight_shape, gain)
        bias = gv_functional.bias_init(bias_shape, weight_shape, gain)
    elseif init_mode == "xavier"
        weight = gv_functional.xavier_init(weight_shape, gain)
        bias = gv_functional.bias_init(bias_shape, weight_shape, gain)
    elseif init_mode == "kaiming_uniform"
        weight = gv_functional.kaiming_uniform_init(weight_shape, gain)
        bias = gv_functional.bias_uniform_init(bias_shape, weight_shape, gain)
    elseif init_mode == "xavier_uniform"
        weight = gv_functional.xavier_uniform_init(weight_shape, gain)
        bias = gv_functional.bias_uniform_init(bias_shape, weight_shape, gain)
    else
        error("""GradValley: general_weight_and_bias_init: init_mode must be one of the following:\n
            "default",
            "default_uniform",
            "kaiming",
            "xavier",
            "kaiming_uniform",
            "xavier_uniform"
        """)
    end

    return weight, bias
end

#= 
Convolution-Layers (Conv & DepthwiseConv)
=#

@doc raw"""
    Conv(in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=(1, 1), padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), groups::Int=1, activation_function::Union{Nothing, String}=nothing, init_mode::String="default_uniform", use_bias::Bool=true)

A convolution layer. Apply a 2D convolution over an input signal with additional batch and channel dimensions.
This layer currently (!) only accepts Float64 array inputs. 

# Arguments
- `in_channels::Int`: the number of channels in the input image
- `out_channels::Int`: the number of channels produced by the convolution
- `kernel_size::Tuple{Int, Int}`: the size of the convolving kernel
- `stride::Tuple{Int, Int}=(1, 1)`: the stride of the convolution
- `padding::Tuple{Int, Int}=(0, 0)`: the zero padding added to all four sides of the input
- `dilation::Tuple{Int, Int}=(1, 1)`: the spacing between kernel elements
- `groups::Int=1`: the number of blocked connections from input channels to output channels (in-channels and out-channels must both be divisible by groups)
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the output after the convolution 
- `init_mode::String="default_uniform"`: the initialization mode of the weights
    (can be `"default_uniform"`, `"default"`, `"kaiming_uniform"`, `"kaiming"`, `"xavier_uniform"` or `"xavier"`)
`use_bias::Bool=true`: if true, adds a learnable bias to the output

# Shapes
- Input: ``(N, C_{in}, H_{in}, W_{in})``
- Weight: ``(C_{out}, \frac{C_{in}}{groups}, H_{w}, W_{w})``
- Bias: ``(C_{out}, )``
- Output: ``(N, C_{out}, H_{out}, W_{out})``, where 
    - ``H_{out} = {\frac{H_{in} + 2 \cdot padding[1] - dilation[1] \cdot (H_w - 1)}{stride[1]}} + 1``
    - ``W_{out} = {\frac{W_{in} + 2 \cdot padding[2] - dilation[2] \cdot (W_w - 1)}{stride[2]}} + 1``

# Useful Fields/Variables
- `kernels::Array{Float64, 4}`: the learnable weights of the layer
- `bias::Vector{Float64}`: the learnable bias of the layer (used when `use_bias=true`)
- `gradients::Array{Float64, 4}`: the current gradients of the weights/kernels
- `bias_gradients::Vector{Float64}`: the current gradients of the bias

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
julia> input = rand(32, 3, 50, 50)
julia> output = forward(m, rand)
```
"""
mutable struct Conv
    in_channels::Int
    out_channels::Int
    kernel_size::Tuple{Int, Int}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    dilation::Tuple{Int, Int}
    groups::Int
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Function, Int} # derivative of activation function
    use_bias::Bool
    # data
    inputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    inputs_padded::Union{Nothing, Array{Float64, 4}} # saved for performence optimization
    kernels::Array{Float64, 4} # weights
    bias::Vector{Float64} # bias
    outputs_no_activation::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    losses::Union{Nothing, Array{Float64, 4}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64, 4}} # losses for the previous layer, can be nothing
    gradients::Array{Float64, 4} # gradients of the kernels/weights
    bias_gradients::Vector{Float64}
    # custom constructor
    function Conv(in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=(1, 1), padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), groups::Int=1, activation_function::Union{Nothing, String}=nothing, init_mode::String="default_uniform", use_bias::Bool=true)
        # setting up the activation function
        new_activation_function, df, gain = general_activation_function_init(activation_function)

        # checks if number of groups is a valid value
        if !(in_channels % groups == 0 && out_channels % groups == 0)
            error("GradValley: Conv: in_channels and out_channels must both be divisible by groups")
        end

        # initialize kernels/weights
        kernels_shape = (out_channels, convert(Int, in_channels / groups), kernel_size[1], kernel_size[2])
        bias_shape = (out_channels, )
        kernels, bias = general_weight_and_bias_init(kernels_shape, bias_shape, init_mode, gain)
        # initialize gradients of kernels/weights and bias
        gradients = zeros(kernels_shape)
        bias_gradients = zeros(bias_shape)

        # placeholders
        inputs = nothing
        inputs_padded = nothing
        outputs_no_activation = nothing
        outputs = nothing
        losses = nothing
        previous_losses = nothing

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
            inputs,
            inputs_padded,
            kernels,
            bias,
            outputs_no_activation,
            outputs,
            losses,
            previous_losses,
            gradients,
            bias_gradients
        )
    end
end

@doc raw"""
    DepthwiseConv(in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=(1, 1), padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), activation_function::Union{Nothing, String}=nothing, init_mode::String="default_uniform", use_bias::Bool=true)

A depthwise convolution layer. Apply a 2D depthwise convolution over an input signal with additional batch and channel dimensions.
This layer currently (!) only accepts Float64 array inputs. 

# Arguments
- `in_channels::Int`: the number of channels in the input image
- `out_channels::Int`: the number of channels produced by the convolution
- `kernel_size::Tuple{Int, Int}`: the size of the convolving kernel
- `stride::Tuple{Int, Int}=(1, 1)`: the stride of the convolution
- `padding::Tuple{Int, Int}=(0, 0)`: the zero padding added to all four sides of the input
- `dilation::Tuple{Int, Int}=(1, 1)`: the spacing between kernel elements
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the output after the convolution 
- `init_mode::String="default_uniform"`: the initialization mode of the weights
    (can be `"default_uniform"`, `"default"`, `"kaiming_uniform"`, `"kaiming"`, `"xavier_uniform"` or `"xavier"`)
`use_bias::Bool=true`: if true, adds a learnable bias to the output

# Shapes
- Input: ``(N, C_{in}, H_{in}, W_{in})``
- Weight: ``(C_{out}, \frac{C_{in}}{groups}, H_{w}, W_{w})``, where ``groups = in\_channels``
- Bias: ``(C_{out}, )``
- Output: ``(N, C_{out}, H_{out}, W_{out})``, where 
    - ``H_{out} = {\frac{H_{in} + 2 \cdot padding[1] - dilation[1] \cdot (H_w - 1)}{stride[1]}} + 1``
    - ``W_{out} = {\frac{W_{in} + 2 \cdot padding[2] - dilation[2] \cdot (W_w - 1)}{stride[2]}} + 1``

# Useful Fields/Variables
- `kernels::Array{Float64, 4}`: the learnable weights of the layer
- `bias::Vector{Float64}`: the learnable bias of the layer (used when `use_bias=true`)
- `gradients::Array{Float64, 4}`: the current gradients of the weights/kernels
- `bias_gradients::Vector{Float64}`: the current gradients of the bias

# Definition
A convolution is called depthwise if ``groups=in\_channels`` and ``out\_channels=k \cdot in\_channels``, where ``k`` is a positive integer.
The second condition ensures that the of number out-channels is divisible by the number of groups/in-channels.
In the background, the standard convolution operation is also used for this layer. 
It is just an interface making clear that this layer can only perform a depthwise convolution.

# Examples
```julia-repl
# square kernels and fully default values of keyword arguments
julia> m = DepthwiseConv(3, 6, (5, 5))
# non-square kernels and unequal stride and with padding as well as specified weight initialization mode
# (init_mode="kaiming" stands for kaiming weight initialization with normally distributed values)
julia> m = DepthwiseConv(3, 6, (3, 5), stride=(2, 1), padding=(2, 1))
# non-square kernels and unequal stride and with padding, dilation and 3 groups
# (because groups=in_channels and out_channles is divisible by groups, it is even a depthwise convolution)
julia> m = DepthwiseConv(3, 6, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), groups=3)
# computing the output of the layer (with random inputs)
julia> input = rand(32, 3, 50, 50)
julia> output = forward(m, rand)
```
"""
mutable struct DepthwiseConv
    in_channels::Int
    out_channels::Int
    kernel_size::Tuple{Int, Int}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    dilation::Tuple{Int, Int}
    groups::Int
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Function, Int} # derivative of activation function
    use_bias::Bool
    # data
    inputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    inputs_padded::Union{Nothing, Array{Float64, 4}} # saved for performence optimization
    kernels::Array{Float64, 4} # weights
    bias::Vector{Float64} # bias
    outputs_no_activation::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    losses::Union{Nothing, Array{Float64, 4}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64, 4}} # losses for the previous layer, can be nothing
    gradients::Array{Float64, 4} # gradients of the kernels/weights
    bias_gradients::Vector{Float64}
    # custom constructor
    function DepthwiseConv(in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=(1, 1), padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), activation_function::Union{Nothing, String}=nothing, init_mode::String="default_uniform", use_bias::Bool=true) # init_mode::String="kaiming", dilation::Tuple{Int, Int}
        # setting up the number of groups and check the validity of out_channels
        groups = in_channels # important for a true depthwise convolution
        if !(out_channels % groups == 0)
            error("GradValley: DepthwiseConv: out_channels must be divisible by groups, so because by definition groups = in_channels,
                out_channels must be divisible by in_channels")
        end
        # setting up the activation function
        new_activation_function, df, gain = general_activation_function_init(activation_function)

        #= not necessary because groups = in_channels and the number of out_channels has already been checked
        # checks if number of groups is a valid value
        if !(in_channels % groups == 0 && out_channels % groups == 0)
            error("GradValley: DepthwiseConv: in_channels and out_channels must both be divisible by groups")
        end
        =#

        # initialize kernels/weights
        kernels_shape = (out_channels, convert(Int, in_channels / groups), kernel_size[1], kernel_size[2])
        bias_shape = (out_channels, )
        kernels, bias = general_weight_and_bias_init(kernels_shape, bias_shape, init_mode, gain)
        # initialize gradients of kernels/weights and bias
        gradients = zeros(kernels_shape)
        bias_gradients = zeros(bias_shape)

        # placeholders
        inputs = nothing
        inputs_padded = nothing
        outputs_no_activation = nothing
        outputs = nothing
        losses = nothing
        previous_losses = nothing

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
            inputs,
            inputs_padded,
            kernels,
            bias,
            outputs_no_activation,
            outputs,
            losses,
            previous_losses,
            gradients,
            bias_gradients
        )
    end
end

# the corresponding functions for DepthwiseConv would be identical to those of Conv, so these are used for both (using Union{Conv, Depthwise} types)

# forward function for a convolution layer (Conv & DepthwiseConv)
# Shape of inputs: (batch_size, in_channels, height, width)
function forward(conv_layer::Union{Conv, DepthwiseConv}, inputs::Array{Float64, 4}) # conv_layer::Conv
    # inputs = copy(inputs)
    conv_layer.inputs = inputs
    outputs_no_activation, inputs_padded = gv_functional.multichannel_conv(inputs, conv_layer.kernels, conv_layer.bias, conv_layer.use_bias, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
    if !(isnothing(conv_layer.activation_function))
        outputs = conv_layer.activation_function(outputs_no_activation)
    else
        outputs = outputs_no_activation
    end

    # saving the results of forward computation in the layer struct (mutable)
    conv_layer.outputs_no_activation = outputs_no_activation
    conv_layer.outputs = outputs
    conv_layer.inputs_padded = inputs_padded

    return outputs
end

# save current losses function for a convolution layer (Conv & DepthwiseConv)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(conv_layer::Union{Conv, DepthwiseConv}, next_layer) # conv_layer::Conv
    conv_layer.losses = next_layer.previous_losses
end

# compute previous losses function for a convolution layer (Conv & DepthwiseConv)
# computes the losses for the previous layer
function compute_previous_losses(conv_layer::Union{Conv, DepthwiseConv}) # conv_layer::Conv
    conv_layer.previous_losses = gv_functional.multichannel_conv_losses(conv_layer)
end

# compute gradients function for a convolution layer (Conv & DepthwiseConv)
# computes the gradients of the kernels/weights of the given layer -> the gradients are not reset but added to the existing gradients
# conv_layer.losses must have a valid content (not nothing), usally losses were given by calling save_current_losses() before
function compute_gradients(conv_layer::Union{Conv, DepthwiseConv}) # conv_layer::Conv
    conv_layer.gradients += gv_functional.multichannel_conv_gradients(conv_layer)
    # conv_layer.gradients = gv_functional.multichannel_conv_gradients(conv_layer)
    if conv_layer.use_bias
        conv_layer.bias_gradients = gv_functional.multichannel_conv_bias_gradients(conv_layer)
    end
end

# zero gradients function for a convolution layer (Conv & DepthwiseConv)
# resets the gradients of the given layer
function zero_gradients(conv_layer::Union{Conv, DepthwiseConv}) # conv_layer::Conv
    # println("reset grads")
    # println(conv_layer.out_channels)
    conv_layer.gradients = zeros(size(conv_layer.kernels))
    conv_layer.bias_gradients = zeros(size(conv_layer.bias))
end

# backward function for a convolution layer (Conv & DepthwiseConv)
# calls save_current_losses(), compute_previous_losses(), compute_gradients() -> so like a shortcut for calling these functions seperatly
function backward(conv_layer::Union{Conv, DepthwiseConv}, next_layer) # conv_layer::Conv
    save_current_losses(conv_layer, next_layer)
    compute_previous_losses(conv_layer)
    compute_gradients(conv_layer)
    # println("conv losses")
    # @time compute_previous_losses(conv_layer)
    # println("conv grads")
    # @time compute_gradients(conv_layer)
end

#= 
Pooling Layers (MaxPool & AvgPool)
=#

@doc raw"""
    MaxPool(kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=kernel_size, padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), activation_function::Union{Nothing, String}=nothing)

A maximum pooling layer. Apply a 2D maximum pooling over an input signal with additional batch and channel dimensions.
This layer currently (!) only accepts Float64 array inputs. 

# Arguments
- `kernel_size::Tuple{Int, Int}`: the size of the window to take the maximum over
- `stride::Tuple{Int, Int}=kernel_size`: the stride of the window
- `padding::Tuple{Int, Int}=(0, 0)`: the zero padding added to all four sides of the input
- `dilation::Tuple{Int, Int}=(1, 1)`: the spacing between the window elements
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the output after the pooling

# Shapes
- Input: ``(N, C, H_{in}, W_{in})``
- Output: ``(N, C, H_{out}, W_{out})``, where 
    - ``H_{out} = {\frac{H_{in} + 2 \cdot padding[1] - dilation[1] \cdot (kernel\_size[1] - 1)}{stride[1]}} + 1``
    - ``W_{out} = {\frac{W_{in} + 2 \cdot padding[2] - dilation[2] \cdot (kernel\_size[2] - 1)}{stride[2]}} + 1``

# Definition
A multichannel 2D maximum pooling (disregarding batch dimension and activation function) can be described as:
```math
\begin{align*}
o_{c, y_{out}, x_{out}} = \max
_{y_w = 1, ..., kernel\_size[1] \ x_w = 1, ..., kernel\_size[2]}
i_{c, y_{in}, x_{in}}
\end{align*}
```
Where
- ``y_{in} = y_{out} + (stride[1] - 1) \cdot (y_{out} - 1) + (y_w - 1) \cdot dilation[1]``
- ``x_{in} = x_{out} + (stride[2] - 1) \cdot (x_{out} - 1) + (x_w - 1) \cdot dilation[2]``
*O* is the output array and *I* the input array.

# Examples
```julia-repl
# pooling of square window of size=(3, 3) and automatically selected stride
julia> m = MaxPool((3, 3))
# pooling of non-square window with custom stride and padding
julia> m = MaxPool((3, 2), stride=(2, 1), padding=(1, 1))
# computing the output of the layer (with random inputs)
julia> input = rand(32, 3, 50, 50)
julia> output = forward(m, rand)
```
"""
mutable struct MaxPool
    # characteristics of the layer
    kernel_size::Tuple{Int, Int}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    dilation::Tuple{Int, Int}
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Function, Int} # derivative of activation function
    # data
    inputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    inputs_paddded::Union{Nothing, Array{Float64, 4}} # not really necessary for pooling
    outputs_no_activation::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    losses::Union{Nothing, Array{Float64, 4}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64, 4}} # losses for the previous layer, can be nothing
    positions::Union{Nothing, Tuple{Array{Int, 4}, Array{Int, 4}}} # only necessary for backpropagation, stores the position of the max value of inputs (for every output), can be nothing
    # custom constructor
    function MaxPool(kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=kernel_size, padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), activation_function::Union{Nothing, String}=nothing)
        kernel_height = kernel_size[1]
        kernel_width = kernel_size[2]
        padding_height = padding[1]
        padding_width = padding[2]
        if !(padding_height <= kernel_height ÷ 2 && padding_width <= kernel_width ÷ 2) # wie genau runden?
            error("GradValley: MaxPool: padding should be smaller than or equal to half of kernel size, but got padding_height = $padding_height, padding_width = $padding_width,
                kernel_height = $kernel_height, kernel_width = $kernel_width")
        end

        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # placeholders
        inputs = nothing
        inputs_padded = nothing
        outputs_no_activation = nothing
        outputs = nothing
        losses = nothing
        previous_losses = nothing
        positions = nothing

        # create new instance/object
        new(kernel_size, 
            stride, 
            padding,
            dilation,
            new_activation_function, 
            df, 
            inputs,
            inputs_padded,
            outputs_no_activation,
            outputs,
            losses,
            previous_losses,
            positions
        )
    end
end

# forward function for a max pooling layer (MaxPool)
# Shape of inputs: (batch_size, in_channels, height, width)
function forward(pool_layer::MaxPool, inputs::Array{Float64, 4})
    # inputs = copy(inputs)
    pool_layer.inputs = inputs
    # outputs_no_activation, positions, inputs_padded = gv_functional.multichannel_pool(inputs, pool_layer.kernel_size, "max", stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
    outputs_no_activation, positions, inputs_padded = gv_functional.multichannel_maxpool(inputs, pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
    if !(isnothing(pool_layer.activation_function))
        outputs = pool_layer.activation_function(outputs_no_activation)
    else
        outputs = outputs_no_activation
    end

    # saving the results of forward computation in the layer struct (mutable)
    pool_layer.outputs_no_activation = outputs_no_activation
    pool_layer.outputs = outputs
    pool_layer.positions = positions
    pool_layer.inputs_paddded = inputs_padded

    return outputs
end

# save current losses function for a max pooling layer (MaxPool)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(pool_layer::MaxPool, next_layer)
    pool_layer.losses = next_layer.previous_losses
end

# compute previous losses function for a max pooling layer (MaxPool)
# computes the losses for the previous layer
function compute_previous_losses(pool_layer::MaxPool)
    pool_layer.previous_losses = gv_functional.multichannel_maxpool_backward(pool_layer)
end

# backward function for a max pooling layer (MaxPool)
# calls save_current_losses(), compute_previous_losses() -> so like a shortcut for calling these functions seperatly
function backward(pool_layer::MaxPool, next_layer)
    save_current_losses(pool_layer, next_layer)
    compute_previous_losses(pool_layer)
end

@doc raw"""
    AvgPool(kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=kernel_size, padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), activation_function::Union{Nothing, String}=nothing)

An average pooling layer. Apply a 2D average pooling over an input signal with additional batch and channel dimensions.
This layer currently (!) only accepts Float64 array inputs. 

# Arguments
- `kernel_size::Tuple{Int, Int}`: the size of the window to take the average over
- `stride::Tuple{Int, Int}=kernel_size`: the stride of the window
- `padding::Tuple{Int, Int}=(0, 0)`: the zero padding added to all four sides of the input
- `dilation::Tuple{Int, Int}=(1, 1)`: the spacing between the window elements
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the output after the pooling

# Shapes
- Input: ``(N, C, H_{in}, W_{in})``
- Output: ``(N, C, H_{out}, W_{out})``, where 
    - ``H_{out} = {\frac{H_{in} + 2 \cdot padding[1] - dilation[1] \cdot (kernel\_size[1] - 1)}{stride[1]}} + 1``
    - ``W_{out} = {\frac{W_{in} + 2 \cdot padding[2] - dilation[2] \cdot (kernel\_size[2] - 1)}{stride[2]}} + 1``

# Definition
A multichannel 2D average pooling (disregarding batch dimension and activation function) can be described as:
- ``o_{c, y_{out}, x_{out}} = \frac{1}{kernel\_size[1] \cdot kernel\_size[2]} \sum_{i=1}^{kernel\_size[1]}\sum_{j=1}^{kernel\_size[2]} i_{c, y_{in}, x_{in}}``, where
    - ``y_{in} = y_{out} + (stride[1] - 1) \cdot (y_{out} - 1) + (y_w - 1) \cdot dilation[1]``
    - ``x_{in} = x_{out} + (stride[2] - 1) \cdot (x_{out} - 1) + (x_w - 1) \cdot dilation[2]``
*O* is the output array and *I* the input array.

# Examples
```julia-repl
# pooling of square window of size=(3, 3) and automatically selected stride
julia> m = AvgPool((3, 3))
# pooling of non-square window with custom stride and padding
julia> m = AvgPool((3, 2), stride=(2, 1), padding=(1, 1))
# computing the output of the layer (with random inputs)
julia> input = rand(32, 3, 50, 50)
julia> output = forward(m, rand)
```
"""
mutable struct AvgPool
    # characteristics of the layer
    kernel_size::Tuple{Int, Int}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    dilation::Tuple{Int, Int}
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Int, Function} # derivative of activation function
    # data
    inputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    inputs_paddded::Union{Nothing, Array{Float64, 4}} # not really necessary for pooling
    outputs_no_activation::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    losses::Union{Nothing, Array{Float64, 4}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64, 4}} # losses for the previous layer, can be nothing
    # custom constructor
    function AvgPool(kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=kernel_size, padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), activation_function::Union{Nothing, String}=nothing)
        kernel_height = kernel_size[1]
        kernel_width = kernel_size[2]
        padding_height = padding[1]
        padding_width = padding[2]
        if !(padding_height <= kernel_height ÷ 2 && padding_width <= kernel_width ÷ 2) # wie genau runden?
            error("GradValley: AvgPool: padding should be smaller than or equal to half of kernel size, but got padding_height = $padding_height, padding_width = $padding_width,
                kernel_height = $kernel_height, kernel_width = $kernel_width")
        end

        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # placeholders
        inputs = nothing
        inputs_padded = nothing
        outputs_no_activation = nothing
        outputs = nothing
        losses = nothing
        previous_losses = nothing

        # create new instance/object
        new(kernel_size, 
            stride, 
            padding,
            dilation, 
            new_activation_function, 
            df, 
            inputs,
            inputs_padded,
            outputs_no_activation,
            outputs,
            losses,
            previous_losses,
        )
    end
end

# forward function for a avg pooling layer (AvgPool)
# Shape of inputs: (batch_size, in_channels, height, width)
function forward(pool_layer::AvgPool, inputs::Array{Float64, 4})
    # inputs = copy(inputs)
    pool_layer.inputs = inputs
    # outputs_no_activation, _, inputs_padded = gv_functional.multichannel_pool(inputs, pool_layer.kernel_size, "avg", stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
    outputs_no_activation, inputs_padded = gv_functional.multichannel_avgpool(inputs, pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
    if !(isnothing(pool_layer.activation_function))
        outputs = pool_layer.activation_function(outputs_no_activation)
    else
        outputs = outputs_no_activation
    end

    # saving the results of forward computation in the layer struct (mutable)
    pool_layer.outputs_no_activation = outputs_no_activation
    pool_layer.outputs = outputs
    pool_layer.inputs_paddded = inputs_padded

    return outputs
end

# save current losses function for a avg pooling layer (AvgPool)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(pool_layer::AvgPool, next_layer)
    pool_layer.losses = next_layer.previous_losses
end

# compute previous losses function for a avg pooling layer (AvgPool)
# computes the losses for the previous layer
function compute_previous_losses(pool_layer::AvgPool)
    pool_layer.previous_losses = gv_functional.multichannel_avgpool_backward(pool_layer)
end

# backward function for a avg pooling layer (AvgPool)
# calls save_current_losses(), compute_previous_losses() -> so like a shortcut for calling these functions seperatly
function backward(pool_layer::AvgPool, next_layer)
    save_current_losses(pool_layer, next_layer)
    compute_previous_losses(pool_layer)
end

#= 
Adaptive Pooling Layers (AdaptiveMaxPool & AdaptiveAvgPool)
=#

@doc raw"""
    AdaptiveMaxPool(output_size::Tuple{Int, Int}; activation_function::Union{Nothing, String}=nothing)

An adaptive maximum pooling layer. Apply a 2D adaptive maximum pooling over an input signal with additional batch and channel dimensions.
For any input size, the size of the output is always equal to the specified ``output\_size``.
This layer currently (!) only accepts Float64 array inputs. 

# Arguments
- `output_size::Tuple{Int, Int}`: the target output size of the image (can even be larger than the input size) of the form ``(H_{out}, W_{out})``
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the output after the pooling

# Shapes
- Input: ``(N, C, H_{in}, W_{in})``
- Output: ``(N, C, H_{out}, W_{out})``, where ``(H_{out}, W_{out}) = output\_size``

# Definition
In some cases, the kernel-size and stride could be calculated in a way that the output would have the target size 
(using a standard maximum pooling with the calculated kernel-size and stride, padding and dilation would not 
be used in this case). However, this approach would only work if the input size is an integer multiple of the output size (See this question at stack overflow
for further information: [stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work](https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work)).
A more generic approach is to calculate the indices of the input with an additional algorithm only for adaptive pooling. 
With this approach, it is even possible that the output is larger than the input what is really unusual for pooling simply because that is the opposite
of what pooling actually should do, namely reducing the size. The `function get_in_indices(in_len, out_len)` in 
[`gv_functional.jl`](https://github.com/jonas208/GradValley.jl/blob/main/src/gv_functional.jl)
(line 95 - 113) implements such an algorithm (similar to the one at the stack overflow question), so you could check there on how exactly it is defined.
Thus, the mathematical definition would be identical to the one at [`MaxPool`](@ref) with the difference that the indices ``y_{in}`` and ``x_{in}`` 
have already been calculated beforehand.

# Examples
```julia-repl
# target output size of 5x5
julia> m = AdaptiveMaxPool((5, 5))
# computing the output of the layer (with random inputs)
julia> input = rand(32, 3, 50, 50)
julia> output = forward(m, rand)
```
"""
mutable struct AdaptiveMaxPool
    # characteristics of the layer
    output_size::Tuple{Int, Int}
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Function, Int} # derivative of activation function
    # data
    inputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs_no_activation::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    losses::Union{Nothing, Array{Float64, 4}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64, 4}} # losses for the previous layer, can be nothing
    positions::Union{Nothing, Tuple{Array{Int, 4}, Array{Int, 4}}} # only necessary for backpropagation, stores the position of the max value of inputs (for every output), can be nothing
    # custom constructor
    function AdaptiveMaxPool(output_size::Tuple{Int, Int}; activation_function::Union{Nothing, String}=nothing)
        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # placeholders
        inputs = nothing
        outputs_no_activation = nothing
        outputs = nothing
        losses = nothing
        previous_losses = nothing
        positions = nothing

        # create new instance/object
        new(output_size,
            new_activation_function, 
            df, 
            inputs,
            outputs_no_activation,
            outputs,
            losses,
            previous_losses,
            positions
        )
    end
end

# forward function for a adaptive max pooling layer (AdaptiveMaxPool)
# Shape of inputs: (batch_size, in_channels, height, width)
function forward(pool_layer::AdaptiveMaxPool, inputs::Array{Float64, 4})
    # println("Input Size: ", size(inputs))
    # println("Output Size: ", pool_layer.output_size)
    # inputs = copy(inputs)
    pool_layer.inputs = inputs
    outputs_no_activation, positions = gv_functional.multichannel_adaptive_maxpool(inputs, pool_layer.output_size)
    if !(isnothing(pool_layer.activation_function))
        outputs = pool_layer.activation_function(outputs_no_activation)
    else
        outputs = outputs_no_activation
    end

    # saving the results of forward computation in the layer struct (mutable)
    pool_layer.outputs_no_activation = outputs_no_activation
    pool_layer.outputs = outputs
    pool_layer.positions = positions

    return outputs
end

# save current losses function for a adaptive max pooling layer (AdaptiveMaxPool)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(pool_layer::AdaptiveMaxPool, next_layer)
    pool_layer.losses = next_layer.previous_losses
end

# compute previous losses function for a adaptive max pooling layer (AdaptiveMaxPool)
# computes the losses for the previous layer
function compute_previous_losses(pool_layer::AdaptiveMaxPool)
    pool_layer.previous_losses = gv_functional.multichannel_adaptive_maxpool_backward(pool_layer)
end

# backward function for a adaptive adaptive max pooling layer (AdaptiveMaxPool)
# calls save_current_losses(), compute_previous_losses() -> so like a shortcut for calling these functions seperatly
function backward(pool_layer::AdaptiveMaxPool, next_layer)
    save_current_losses(pool_layer, next_layer)
    compute_previous_losses(pool_layer)
end

@doc raw"""
    AdaptiveAvgPool(output_size::Tuple{Int, Int}; activation_function::Union{Nothing, String}=nothing)

An adaptive average pooling layer. Apply a 2D adaptive average pooling over an input signal with additional batch and channel dimensions.
For any input size, the size of the output is always equal to the specified ``output\_size``.
This layer currently (!) only accepts Float64 array inputs. 

# Arguments
- `output_size::Tuple{Int, Int}`: the target output size of the image (can even be larger than the input size) of the form ``(H_{out}, W_{out})``
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the output after the pooling

# Shapes
- Input: ``(N, C, H_{in}, W_{in})``
- Output: ``(N, C, H_{out}, W_{out})``, where ``(H_{out}, W_{out}) = output\_size``

# Definition
In some cases, the kernel-size and stride could be calculated in a way that the output would have the target size 
(using a standard average pooling with the calculated kernel-size and stride, padding and dilation would not 
be used in this case). However, this approach would only work if the input size is an integer multiple of the output size (See this question at stack overflow
for further information: [stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work](https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work)).
A more generic approach is to calculate the indices of the input with an additional algorithm only for adaptive pooling. 
With this approach, it is even possible that the output is larger than the input what is really unusual for pooling simply because that is the opposite
of what pooling actually should do, namely reducing the size. The `function get_in_indices(in_len, out_len)` in 
[`gv_functional.jl`](https://github.com/jonas208/GradValley.jl/blob/main/src/gv_functional.jl)
(line 95 - 113) implements such an algorithm (similar to the one at the stack overflow question), so you could check there on how exactly it is defined.
Thus, the mathematical definition would be identical to the one at [`AvgPool`](@ref) with the difference that the indices ``y_{in}`` and ``x_{in}`` 
have already been calculated beforehand.

# Examples
```julia-repl
# target output size of 5x5
julia> m = AdaptiveAvgPool((5, 5))
# computing the output of the layer (with random inputs)
julia> input = rand(32, 3, 50, 50)
julia> output = forward(m, rand)
```
"""
mutable struct AdaptiveAvgPool
    # characteristics of the layer
    output_size::Tuple{Int, Int}
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Function, Int} # derivative of activation function
    # data
    inputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs_no_activation::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    losses::Union{Nothing, Array{Float64, 4}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64, 4}} # losses for the previous layer, can be nothing
    # custom constructor
    function AdaptiveAvgPool(output_size::Tuple{Int, Int}; activation_function::Union{Nothing, String}=nothing)
        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # placeholders
        inputs = nothing
        outputs_no_activation = nothing
        outputs = nothing
        losses = nothing
        previous_losses = nothing

        # create new instance/object
        new(output_size,
            new_activation_function, 
            df, 
            inputs,
            outputs_no_activation,
            outputs,
            losses,
            previous_losses,
        )
    end
end

# forward function for a adaptive avg pooling layer (AdaptiveAvgPool)
# Shape of inputs: (batch_size, in_channels, height, width)
function forward(pool_layer::AdaptiveAvgPool, inputs::Array{Float64, 4})
    # println("Input Size: ", size(inputs))
    # println("Output Size: ", pool_layer.output_size)
    # inputs = copy(inputs)
    pool_layer.inputs = inputs
    outputs_no_activation = gv_functional.multichannel_adaptive_avgpool(inputs, pool_layer.output_size)
    if !(isnothing(pool_layer.activation_function))
        outputs = pool_layer.activation_function(outputs_no_activation)
    else
        outputs = outputs_no_activation
    end

    # saving the results of forward computation in the layer struct (mutable)
    pool_layer.outputs_no_activation = outputs_no_activation
    pool_layer.outputs = outputs

    return outputs
end

# save current losses function for a adaptive avg pooling layer (AdaptiveAvgPool)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(pool_layer::AdaptiveAvgPool, next_layer)
    pool_layer.losses = next_layer.previous_losses
end

# compute previous losses function for a avg pooling layer (AdaptiveAvgPool)
# computes the losses for the previous layer
function compute_previous_losses(pool_layer::AdaptiveAvgPool)
    pool_layer.previous_losses = gv_functional.multichannel_adaptive_avgpool_backward(pool_layer)
end

# backward function for a adaptive avg pooling layer (AdaptiveAvgPool)
# calls save_current_losses(), compute_previous_losses() -> so like a shortcut for calling these functions seperatly
function backward(pool_layer::AdaptiveAvgPool, next_layer)
    save_current_losses(pool_layer, next_layer)
    compute_previous_losses(pool_layer)
end

#=
Fully connected layer (Fc)
=#

mutable struct Fc
    # characteristics of the layer
    in_features::Int
    out_features::Int
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Int, Function} # derivative of activation function
    use_bias::Bool
    # data
    inputs::Union{Nothing, Array{Float64, 2}} # can be nothing
    weights::Array{Float64, 2} # weights
    bias::Vector{Float64} # bias
    outputs_no_activation::Union{Nothing, Array{Float64, 2}} # can be nothing
    outputs::Union{Nothing, Array{Float64, 2}} # can be nothing
    losses::Union{Nothing, Array{Float64, 2}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64, 2}} # losses for the previous layer, can be nothing
    gradients::Array{Float64, 2} # gradients of the kernels/weights
    bias_gradients::Vector{Float64}
    # custom constructor
    function Fc(in_features::Int, out_features::Int; activation_function::Union{Nothing, String}=nothing, init_mode::String="default_uniform", use_bias::Bool=true) # init_mode::String="kaiming"
        # setting up the activation function
        new_activation_function, df, gain = general_activation_function_init(activation_function)

        # initialize weights
        weights_shape = (out_features, in_features)
        bias_shape = (out_features, )
        weights, bias = general_weight_and_bias_init(weights_shape, bias_shape, init_mode, gain)
        # initialize gradients of weights
        gradients = zeros(weights_shape)
        bias_gradients = zeros(out_features)

        # placeholders
        inputs = nothing
        outputs_no_activation = nothing
        outputs = nothing
        losses = nothing
        previous_losses = nothing

        # create new instance/object
        new(in_features, 
            out_features, 
            new_activation_function, 
            df,
            use_bias, 
            inputs,
            weights,
            bias,
            outputs_no_activation,
            outputs,
            losses,
            previous_losses,
            gradients,
            bias_gradients
        )
    end
end

# forward function for a fully connected layer (Fc)
# Shape of inputs: (batch_size, in_features)
function forward(fc_layer::Fc, inputs::Array{Float64, 2})
    # inputs = copy(inputs)
    fc_layer.inputs = inputs
    outputs_no_activation = gv_functional.fc_forward(inputs, fc_layer.weights, fc_layer.bias, fc_layer.use_bias)
    if !(isnothing(fc_layer.activation_function))
        outputs = fc_layer.activation_function(outputs_no_activation)
    else
        outputs = outputs_no_activation
    end

    # saving the results of forward computation in the layer struct (mutable)
    fc_layer.outputs_no_activation = outputs_no_activation
    fc_layer.outputs = outputs

    return outputs
end

# save current losses function for a fully connected layer (Fc)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(fc_layer::Fc, next_layer)
    fc_layer.losses = next_layer.previous_losses
end

# compute previous losses function for a fully connected layer (Fc)
# computes the losses for the previous layer
function compute_previous_losses(fc_layer::Fc)
    fc_layer.previous_losses = gv_functional.fc_losses(fc_layer)
end

# compute gradients function for a fully connected layer (Fc)
# computes the gradients of the weights of the given layer -> the gradients are not reset but added to the existing gradients
# conv_layer.losses must have a valid content (not nothing), usally losses were given by calling save_current_losses() before
function compute_gradients(fc_layer::Fc)
    gradients, bias_gradients = gv_functional.fc_gradients(fc_layer)
    fc_layer.gradients += gradients
    if fc_layer.use_bias
        fc_layer.bias_gradients += bias_gradients
    end
end

# zero gradients function for a fully connected layer (Fc)
# resets the gradients of the given layer
function zero_gradients(fc_layer::Fc)
    fc_layer.gradients = zeros(size(fc_layer.weights))
    fc_layer.bias_gradients = zeros(size(fc_layer.bias))
end

# backward function for a fully connected layer (Fc)
# calls save_current_losses(), compute_previous_losses(), compute_gradients() -> so like a shortcut for calling these functions seperatly
function backward(fc_layer::Fc, next_layer)
    save_current_losses(fc_layer, next_layer)
    # compute_previous_losses(fc_layer)
    # compute_gradients(fc_layer)
    gradients, bias_gradients, previous_losses = gv_functional.fc_backward(fc_layer)
    fc_layer.previous_losses = previous_losses
    fc_layer.gradients += gradients
    if fc_layer.use_bias
        fc_layer.bias_gradients += bias_gradients
    end
end

#=
Reshape layer (Reshape, mostly used as a flatten layer)
=#

mutable struct Reshape # maybe use parametric types for specifing number of dimension of arrays for concrete types
    # characteristics of the layer
    out_shape::Tuple
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Function, Int} # derivative of activation function
    # data
    inputs::Union{Nothing, Array{Float64}} # can be nothing
    outputs_no_activation::Union{Nothing, Array{Float64}} # can be nothing
    outputs::Union{Nothing, Array{Float64}} # can be nothing
    losses::Union{Nothing, Array{Float64}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64}} # losses for the previous layer, can be nothing
    # custom constructor
    function Reshape(out_shape; activation_function::Union{Nothing, String}=nothing)
        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # placeholders
        inputs = nothing
        outputs_no_activation = nothing
        outputs = nothing
        losses = nothing
        previous_losses = nothing

        # create new instance/object
        new(out_shape,
            new_activation_function, 
            df, 
            inputs,
            outputs_no_activation,
            outputs,
            losses,
            previous_losses,
        )
    end
end

# forward function for a reshape layer (Reshape)
# Shape of inputs: (batch_size, in_channels, height, width)
function forward(reshape_layer::Reshape, inputs::Array{Float64}) # inputs::Array{Float64, N}) where N
    # inputs = copy(inputs)
    reshape_layer.inputs = inputs
    outputs_no_activation = gv_functional.reshape_forward(inputs, reshape_layer.out_shape)
    if !(isnothing(reshape_layer.activation_function))
        outputs = reshape_layer.activation_function(outputs_no_activation)
    else
        outputs = outputs_no_activation
    end

    # saving the results of forward computation in the layer struct (mutable)
    reshape_layer.outputs_no_activation = outputs_no_activation
    reshape_layer.outputs = outputs

    return outputs
end

# save current losses function for a reshape layer (Reshape)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(reshape_layer::Reshape, next_layer)
    reshape_layer.losses = next_layer.previous_losses
end

# compute previous losses function for a reshape layer (Reshape)
# computes the losses for the previous layer
function compute_previous_losses(reshape_layer::Reshape)
    reshape_layer.previous_losses = gv_functional.reshape_backward(reshape_layer)
end

# backward function for a reshape layer (Reshape)
# calls save_current_losses(), compute_previous_losses() -> so like a shortcut for calling these functions seperatly
function backward(reshape_layer::Reshape, next_layer)
    save_current_losses(reshape_layer, next_layer)
    compute_previous_losses(reshape_layer)
end

#=
Batch normalization layer for 4d-inputs (BatchNorm2d)
=#

mutable struct BatchNorm2d
    # characteristics of the layer
    num_features::Int
    epsilon::Float64
    momentum::Float64
    affine::Bool
    track_running_stats::Bool
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Function, Int} # derivative of activation function
    # data
    inputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs_no_activation::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    losses::Union{Nothing, Array{Float64, 4}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64, 4}} # losses for the previous layer, can be nothing
    outputs_no_weights_applied::Union{Nothing, Array{Float64, 4}} # no weights and of course no activation applied, used as a cache for backpropagation
    # learnabel parameters
    weight_gamma::Vector{Float64} # ::Array{Float64, 1}
    weight_beta::Vector{Float64} # ::Array{Float64, 1}
    gradient_gamma::Vector{Float64} # ::Array{Float64, 1}
    gradient_beta::Vector{Float64} # ::Array{Float64, 1}
    # running statistics
    running_mean::Vector{Float64} # ::Array{Float64, 1}
    running_variance::Vector{Float64} # ::Array{Float64, 1}
    test_mode::Bool
    # custom constructor
    function BatchNorm2d(num_features::Int; epsilon::Float64=1e-05, momentum::Float64=0.1, affine::Bool=true, track_running_stats::Bool=true, activation_function::Union{Nothing, String}=nothing)
        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # initialize weights
        weight_gamma = ones(num_features)
        weight_beta = zeros(num_features) # size(weight_gamma)
        
        # initialize gradients of weights
        gradient_gamma = zeros(size(weight_gamma))
        gradient_beta = zeros(size(weight_beta)) # size(weight_gamma)

        # initialize running statistics
        running_mean = zeros(num_features)
        running_variance = ones(num_features)
        test_mode = false

        # placeholders
        inputs = nothing
        outputs_no_activation = nothing
        outputs = nothing
        losses = nothing
        previous_losses = nothing
        outputs_no_weights_applied = nothing

        # create new instance/object
        new(num_features,
            epsilon,
            momentum,
            affine,
            track_running_stats,
            new_activation_function,
            df,
            inputs,
            outputs_no_activation,
            outputs,
            losses,
            previous_losses,
            outputs_no_weights_applied,
            weight_gamma,
            weight_beta,
            gradient_gamma,
            gradient_beta,
            running_mean,
            running_variance,
            test_mode
        )
    end
end

# forward function for a batch normalization layer (BatchNorm2d)
# Shape of inputs: (batch_size, in_channels, height, width)
function forward(batchnorm_layer::BatchNorm2d, inputs::Array{Float64, 4}) # inputs::Array
    # inputs = copy(inputs)
    batchnorm_layer.inputs = inputs
    outputs_no_activation, running_mean, running_variance, outputs_no_weights = gv_functional.batchNorm2d_forward(inputs,
                                                                batchnorm_layer.weight_gamma, 
                                                                batchnorm_layer.weight_beta,  
                                                                batchnorm_layer.track_running_stats, 
                                                                batchnorm_layer.momentum, 
                                                                batchnorm_layer.running_mean, 
                                                                batchnorm_layer.running_variance, 
                                                                batchnorm_layer.test_mode, 
                                                                epsilon=batchnorm_layer.epsilon)
    if !(isnothing(batchnorm_layer.activation_function))
        outputs = batchnorm_layer.activation_function(outputs_no_activation)
    else
        outputs = outputs_no_activation
    end

    # saving the results of forward computation in the layer struct (mutable)
    batchnorm_layer.outputs_no_activation = outputs_no_activation
    batchnorm_layer.outputs = outputs
    batchnorm_layer.running_mean = running_mean
    batchnorm_layer.running_variance = running_variance
    batchnorm_layer.outputs_no_weights_applied = outputs_no_weights

    return outputs
end

# save current losses function for a batch normalization layer (BatchNorm2d)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(batchnorm_layer::BatchNorm2d, next_layer)
    batchnorm_layer.losses = next_layer.previous_losses
end

# compute previous losses function for a batch normalization layer (BatchNorm2d)
# computes the losses for the previous layer
function compute_previous_losses(batchnorm_layer::BatchNorm2d)
    batchnorm_layer.previous_losses = gv_functional.batchNorm2d_losses(batchnorm_layer)
end

# compute gradients function for a batch normalization layer (BatchNorm2d)
# computes the gradients of the weights of the given layer -> the gradients are not reset but added to the existing gradients
# conv_layer.losses must have a valid content (not nothing), usally losses were given by calling save_current_losses() before
function compute_gradients(batchnorm_layer::BatchNorm2d)
    gradient_gamma, gradient_beta = gv_functional.batchNorm2d_gradients(batchnorm_layer)
    batchnorm_layer.gradient_gamma += gradient_gamma
    batchnorm_layer.gradient_beta += gradient_beta
end

# zero gradients function for a batch normalization layer (BatchNorm2d)
# resets the gradients of the given layer
function zero_gradients(batchnorm_layer::BatchNorm2d)
    batchnorm_layer.gradient_gamma = zeros(size(batchnorm_layer.weight_gamma))
    batchnorm_layer.gradient_beta = zeros(size(batchnorm_layer.weight_beta))
end

# backward function for a batch normalization layer (BatchNorm2d)
# calls save_current_losses(), compute_previous_losses(), compute_gradients() -> so like a shortcut for calling these functions seperatly
function backward(batchnorm_layer::BatchNorm2d, next_layer)
    save_current_losses(batchnorm_layer, next_layer)
    compute_previous_losses(batchnorm_layer)
    compute_gradients(batchnorm_layer)
end

# changes the mode of the given batchnorm_layer to trainmode
function trainmode!(batchnorm_layer::BatchNorm2d)
    batchnorm_layer.test_mode = false
end

# changes the mode of the given batchnorm_layer to testmode
function testmode!(batchnorm_layer::BatchNorm2d)
    batchnorm_layer.test_mode = true
end

#=
Softmax layer for nd-inputs
=#

mutable struct Softmax
    # characteristics of the layer
    dim::Integer # the softmax will be calculated along that dimension
    outputs::Union{Nothing, Array{Float64}} # can be nothing
    losses::Union{Nothing, Array{Float64}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64}} # losses for the previous layer, can be nothing
    # custom constructor
    function Softmax(; dim::Integer=1)
        # placeholders
        outputs = nothing
        losses = nothing
        previous_losses = nothing

        # create new instance/object
        new(dim, outputs, losses, previous_losses)
    end
end

# forward function for a softmax layer (Softmax)
# Shape of inputs: (*) where * means an arbitrary number of dimensions
function forward(softmax_layer::Softmax, inputs::Array{Float64}) # inputs::Array, inputs::Array{Float64, N}) where N
    # println(typeof(inputs))
    outputs = gv_functional.softmax(inputs, dim=softmax_layer.dim)

    # saving the results of forward computation in the layer struct (mutable)
    softmax_layer.outputs = outputs

    return outputs
end

# save current losses function for a softmax layer (Softmax)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(softmax_layer::Softmax, next_layer)
    softmax_layer.losses = next_layer.previous_losses
end

# compute previous losses function for a softmax layer (Softmax)
# computes the losses for the previous layer
function compute_previous_losses(softmax_layer::Softmax)
    softmax_layer.previous_losses = gv_functional.softmax_backward(softmax_layer)
end

# backward function for a softmax layer (Softmax)
# calls save_current_losses(), compute_previous_losses() -> so like a shortcut for calling these functions seperatly
function backward(softmax_layer::Softmax, next_layer)
    save_current_losses(softmax_layer, next_layer)
    compute_previous_losses(softmax_layer)
end

# struct SequentialContainer
mutable struct SequentialContainer # pay attention when mutable because when last_stack will be changed during training, the backpropagation process will may not work correctly
    layer_stack::Vector{<: Any} # Union{Conv, MaxPool, AvgPool, Reshape, Fc, BatchNorm2d}
    num_layers::Int
    # previous_losses::Array # required for backpropagation when other sequential containers are contained in an outer sequential container
    previous_losses::Array{Float64} # required for backpropagation when other sequential containers are contained in an outer sequential container
    # custom constructor
    function SequentialContainer(layer_stack::Vector{<: Any}) # Union{Conv, MaxPool, AvgPool, Reshape, Fc, BatchNorm2d}, Union{Conv, MaxPool, AvgPool, Reshape, Fc, BatchNorm2d, SequentialContainer}
        num_layers = length(layer_stack)
        if num_layers < 2
            error("GradValley: SequentialContainer: the number of layers in layer_stack muste be at least 2")
        end
        # create new instance/object
        new(layer_stack, num_layers)
    end
end

# making the SequentialContainer iterable
Base.iterate(SC::SequentialContainer, state=1) = state > SC.num_layers ? nothing : (SC.layer_stack[state], state+1)
# making the length/size (=num_batches) of the SequentialContainer available
Base.length(SC::SequentialContainer) = SC.num_layers
Base.size(SC::SequentialContainer) = SC.num_layers # to do: might be better if it returned a nice looking overview of the model's architecture and number of parameters
# making the SequentialContainer indexable
function Base.getindex(SC::SequentialContainer, index::Integer)
    1 <= index <= SC.num_layers || throw(BoundsError(SC, index))
    return SC.layer_stack[index]
end
Base.firstindex(SC::SequentialContainer) = 1
Base.lastindex(SC::SequentialContainer) = SC.num_layers
function Base.getindex(SC::SequentialContainer, index_range::UnitRange{<: Integer})
    1 <= index_range[1] <= SC.num_layers || throw(BoundsError(SC, index))
    1 <= index_range[end] <= SC.num_layers || throw(BoundsError(SC, index))
    layer_stack = SC.layer_stack[index_range]
    sc = SequentialContainer(layer_stack)
    return sc
end
# Base.show(SC::SequentialContainer) = "my model"

function forward(sc::SequentialContainer, inputs::Array{Float64}) # sc stands for SequentialContainer, inputs::Array{Float64, N}) where N
    layer_stack = sc.layer_stack
    outputs = inputs
    for layer in layer_stack
        outputs = forward(layer, outputs)
        # println(typeof(layer))
        # outputs = @time forward(layer, outputs)
    end

    return outputs
end

function backward(sc::SequentialContainer, derivative_loss::Array{Float64}) # sc stands for SequentialContainer
    # println("called")
    layer_stack = sc.layer_stack
    # last_layer = sc.layer_stack[sc.num_layers]
    last_layer = sc.layer_stack[end]
    if typeof(last_layer) != SequentialContainer
        last_layer.losses = derivative_loss
        # if !(typeof(last_layer) == AvgPool || typeof(last_layer) == MaxPool || typeof(last_layer) == AdaptiveAvgPool || typeof(last_layer) == AdaptiveMaxPool || typeof(last_layer) == Reshape || typeof(last_layer) == Softmax)
        if typeof(last_layer) == Conv || typeof(last_layer) == Fc || typeof(last_layer) == BatchNorm2d            
            compute_gradients(last_layer)
        end
        # println(typeof(last_layer))
        # @time compute_previous_losses(last_layer)
        compute_previous_losses(last_layer)
    else
        backward(last_layer, derivative_loss)
        # println(typeof(last_layer))
        # @time backward(last_layer, derivative_loss)
    end

    for layer_index in sc.num_layers-1:-1:1
        current_layer = layer_stack[layer_index]
        next_layer = layer_stack[layer_index + 1]
        if typeof(current_layer) != SequentialContainer
            backward(current_layer, next_layer)
            # println(typeof(current_layer))
            # @time backward(current_layer, next_layer)
        else
            backward(current_layer, next_layer.previous_losses)
        end
    end

    first_layer = sc.layer_stack[1]
    sc.previous_losses = first_layer.previous_losses
end

# resets the the gradients of all Conv/DepthwiseConv/Fc/BatchNorm2d layers in the given SequentialContainer
function zero_gradients(sc::SequentialContainer)
    # println("called zero_gradients")
    for layer in sc.layer_stack
        if typeof(layer) == Conv || typeof(layer) == DepthwiseConv || typeof(layer) == Fc || typeof(layer) == BatchNorm2d || typeof(layer) == SequentialContainer
            zero_gradients(layer)
        end
    end
end

# if the given SequentialContainer contains BatchNorm2d layers, there mode will be set to trainmode
function trainmode!(sc::SequentialContainer)
    for layer in sc.layer_stack
        if typeof(layer) == BatchNorm2d || typeof(layer) == SequentialContainer
            # layer.test_mode = false
            trainmode!(layer)
        end
    end
end

# if the given SequentialContainer contains BatchNorm2d layers, there mode will be set to testmode
function testmode!(sc::SequentialContainer)
    for layer in sc.layer_stack
        if typeof(layer) == BatchNorm2d || typeof(layer) == SequentialContainer
            # layer.test_mode = true
            testmode!(layer)
        end
    end
end

# extracts all layers in a stack of (nested) Sequential Containers and layers recursively, returns a vector only containing all the pure layers
function extract_layers(sc::SequentialContainer, layer_stack)
    # println("called extract_layers")
    for layer in sc.layer_stack
        if typeof(layer) == SequentialContainer
            extract_layers(layer, layer_stack)
        else
            push!(layer_stack, layer)
        end
    end

    return layer_stack
end

# returns a string containing useful information about the given layer
function get_layer_summary(layer)
    summary = "$(typeof(layer)): "
    num_params = 0
    if typeof(layer) == Conv
    # if typeof(layer) == Conv || typeof(layer) == DepthwiseConv
        # summary *= "in_channels=$(layer.in_channels), out_channels=$(layer.out_channels), kernel_size=$(layer.kernel_size), stride=$(layer.stride), padding=$(layer.stride), dilation=$(layer.dilation), groups=$(layer.groups), activation_function=$(layer.activation_function), init_mode=$(layer.init_mode), use_bias=$(layer.use_bias)"
        summary *= "in_channels=$(layer.in_channels), out_channels=$(layer.out_channels), kernel_size=$(layer.kernel_size), stride=$(layer.stride), padding=$(layer.stride), dilation=$(layer.dilation), groups=$(layer.groups), activation_function=$(layer.activation_function), use_bias=$(layer.use_bias)"
        if layer.use_bias
            num_params += length(layer.kernels) + length(layer.bias)
        else
            num_params += length(layer.kernels)
        end
    # groups are not shown with this version
    elseif typeof(layer) == DepthwiseConv
            # summary *= "in_channels=$(layer.in_channels), out_channels=$(layer.out_channels), kernel_size=$(layer.kernel_size), stride=$(layer.stride), padding=$(layer.stride), dilation=$(layer.dilation), activation_function=$(layer.activation_function), init_mode=$(layer.init_mode), use_bias=$(layer.use_bias)"
            summary *= "in_channels=$(layer.in_channels), out_channels=$(layer.out_channels), kernel_size=$(layer.kernel_size), stride=$(layer.stride), padding=$(layer.stride), dilation=$(layer.dilation), activation_function=$(layer.activation_function), use_bias=$(layer.use_bias)"
        if layer.use_bias
            num_params += length(layer.kernels) + length(layer.bias)
        else
            num_params += length(layer.kernels)
        end
    elseif typeof(layer) == MaxPool || typeof(layer) == AvgPool
        summary *= "kernel_size=$(layer.kernel_size), stride=$(layer.stride), padding=$(layer.stride), dilation=$(layer.dilation), activation_function=$(layer.activation_function)"
    elseif typeof(layer) == AdaptiveMaxPool || typeof(layer) == AdaptiveAvgPool
        summary *= "output_size=$(layer.output_size), activation_function=$(layer.activation_function)"
    elseif typeof(layer) == Fc
        summary *= "in_features=$(layer.in_features), out_features=$(layer.out_features), activation_function=$(layer.activation_function), use_bias=$(layer.use_bias)"
        if layer.use_bias
            num_params += length(layer.weights) + length(layer.bias)
        else
            num_params += length(layer.weights)
        end
    elseif typeof(layer) == Reshape
        summary *= "out_shape=$(layer.out_shape), activation_function=$(layer.activation_function)"
    elseif typeof(layer) == BatchNorm2d
        summary *= "num_features=$(layer.num_features), epsilon=$(layer.epsilon), momentum=$(layer.momentum), affine=$(layer.affine), track_running_stats=$(layer.track_running_stats), test_mode=$(layer.test_mode), activation_function=$(layer.activation_function)"
        num_params += length(layer.weight_gamma) + length(layer.weight_beta)
    elseif typeof(layer) == Softmax
        summary *= "dim=$(layer.dim)"
    end
    
    return summary, num_params
end

# returns a string (and the total number of parameters) with an overview of the sc
function summarize_sc(sc::SequentialContainer, sub_counter::String)
    summary = "SequentialContainer\n(\n"
    num_params = 0
    for (i, layer) in enumerate(sc.layer_stack)
        if sub_counter == ""
            sub_counter_new = string(i)
        else
            sub_counter_new = sub_counter * "." * string(i)
        end
        if typeof(layer) != SequentialContainer
            layer_summary, layer_num_params = get_layer_summary(layer)
            summary *= "($sub_counter_new) $layer_summary\n"
        else
            layer_summary, layer_num_params = summarize_sc(layer, sub_counter_new) # works recursively
            lines = split(layer_summary, "\n")
            summary *= "($sub_counter_new) " * lines[1] * "\n"
            for line in lines[2:end]
                summary *= "     " * line * "\n"
            end
        end
        num_params += layer_num_params
    end
    summary *= ")\n"

    return summary, num_params
end

# returns a string (and the total number of parameters) intended for printing with an overview of the model and its number of parameters
function summarize_model(sc::SequentialContainer)
    summary, num_params = summarize_sc(sc, "")
    summary *= "Total Layers: $(length(extract_layers(sc, [])))\n"
    summary *= "Total Parameters: $num_params"

    return summary, num_params
end

end # end of module "Layers"