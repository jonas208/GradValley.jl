module Layers
using ..Functional
# make Functional accessible via gv_functional
gv_functional = Functional

# export all layers and nearly all functions
export Conv, DepthwiseConv, ConvTranspose, Fc, BatchNorm2d, MaxPool, AdaptiveMaxPool, AvgPool, AdaptiveAvgPool, Reshape, Softmax, Identity, SequentialContainer, GraphContainer
export forward, backward, zero_gradients, trainmode!, testmode!, summarize_model

#=
Documentation for functions with many but very similar methods
=#

"""
    forward(layer, input::Array{Float64})

The forward function for computing the output of a module. For every layer/container, an individual method exists.
However, all these methods work exactly the same. They all take the layer/container as the first argument and the input data
as the second argument. The output is returned. 
All layers/containers currently (!) only accept Float64 array inputs, so all methods also expect a Float64 array input, the number
of dimensions can differ.

# Examples
```julia-repl
# define some layers and containers
julia> layer = Conv(3, 6, (5, 5))
julia> container = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# create some random input data
julia> layer_input = rand(32, 3, 50, 50)
julia> container_input = rand(32, 1000)
# compute the output of the modules
julia> layer_output = forward(layer, layer_input)
julia> container_output = forward(container, container_input)
```
"""
function forward end

"""
    backward(layer, next_layer)

The backward function for computing the gradients for a layer. Also well known as backpropagation. For every layer, an individual method exists.
However, all these methods work exactly the same. They all take the current layer for which the gradients should be computed as the first argument 
and next layer containing the backpropagated losses used to compute the gradients for the current layer.
No gradients are returned, they are just saved in the layer.

!!! warning
    Note that this backward function differs from the backward functions for containers. As a user, it is highly recommended to use containers
    for model building because they create the forward and backward pass automatically. Calling the backward functions for all the layers
    individually is normally not necessary and also not recommended.

# Examples
```julia-repl
# define two layers  
julia> layer_1 = Fc(500, 250)
julia> layer_2 = Fc(250, 125)
# compute the output of the layers (with random inputs)
julia> output = forward(layer_1, rand(32, 500))
julia> output = forward(layer_2, output)
# use a loss function (with random data as target values) and save the derivative of the loss
julia> loss, derivative_loss = mse_loss(output, rand(32, 125)) # note that GradValley.Optimization.mse_loss must be imported
# before the gradients are recalculated, the old gradients should always be reset first
julia> zero_gradients(layer_1)
julia> zero_gradients(layer_2)
# backpropagation (compute the gradients to the weights and backpropagate the losses)
# because there exists no next layer after the last layer to take the backpropagted losses from, we will have to manually store the derivative of the loss in the last layer
julia> layer_2.losses = derivative_loss
# than we can compute the gradients to the weights and backpropagate the losses
compute_gradients(layer_2)
compute_previous_losses(layer_2)
# now we can go on with the actual backward function
backward(layer_1, layer_2)
```
"""
function backward end

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
Convolution-Layers (Conv & DepthwiseConv ConvTranspose)
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
- `use_bias::Bool=true`: if true, adds a learnable bias to the output

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
julia> output = forward(m, input)
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
- `use_bias::Bool=true`: if true, adds a learnable bias to the output

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
julia> output = forward(m, input)
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

@doc raw"""
    ConvTranspose(in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=(1, 1), padding::Tuple{Int, Int}=(0, 0), output_padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), groups::Int=1, activation_function::Union{Nothing, String}=nothing, init_mode::String="default_uniform", use_bias::Bool=true)

A transpose convolution layer (also known as fractionally-strided convolution or deconvolution). Apply a 2D transposed convolution over an input signal with additional batch and channel dimensions.
This layer currently (!) only accepts Float64 array inputs. 

# Arguments
- `in_channels::Int`: the number of channels in the input image
- `out_channels::Int`: the number of channels produced by the convolution
- `kernel_size::Tuple{Int, Int}`: the size of the convolving kernel
- `stride::Tuple{Int, Int}=(1, 1)`: the stride of the convolution
- `padding::Tuple{Int, Int}=(0, 0)`: because transposed convolution can be seen as a partly (not true) inverse of convolution, padding means is this case to cut off the desired number of pixels on each side (instead of adding pixels)
- `output_padding::Tuple{Int, Int}=(0, 0)`: additional size added to one side of each dimension in the output shape (note that output_padding is only used to calculate the output shape, but does not actually add zero-padding to the output)
- `dilation::Tuple{Int, Int}=(1, 1)`: the spacing between kernel elements
- `groups::Int=1`: the number of blocked connections from input channels to output channels (in-channels and out-channels must both be divisible by groups)
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the output after the convolution 
- `init_mode::String="default_uniform"`: the initialization mode of the weights
    (can be `"default_uniform"`, `"default"`, `"kaiming_uniform"`, `"kaiming"`, `"xavier_uniform"` or `"xavier"`)
- `use_bias::Bool=true`: if true, adds a learnable bias to the output

# Shapes
- Input: ``(N, C_{in}, H_{in}, W_{in})``
- Weight: ``(C_{in}, \frac{C_{out}}{groups}, H_{w}, W_{w})``
- Bias: ``(C_{out}, )``
- Output: ``(N, C_{out}, H_{out}, W_{out})``, where 
    - ``H_{out} = (H_{in} - 1) \cdot stride[1] - 2 \cdot padding[1] + dilation[1] \cdot (H_w - 1) + output\_padding[1] + 1``
    - ``W_{out} = (W_{in} - 1) \cdot stride[2] - 2 \cdot padding[2] + dilation[2] \cdot (W_w - 1) + output\_padding[2] + 1``

# Useful Fields/Variables
- `kernels::Array{Float64, 4}`: the learnable weights of the layer
- `bias::Vector{Float64}`: the learnable bias of the layer (used when `use_bias=true`)
- `gradients::Array{Float64, 4}`: the current gradients of the weights/kernels
- `bias_gradients::Vector{Float64}`: the current gradients of the bias

# Definition
A transposed convolution can be seen as the gradient of a normal convolution with respect to its inputs. 
The forward pass of a transposed convolution is the backward pass of a normal convolution, so the forward pass
of a normal convolution becomes the backward pass of a transposed convolution (with respect to its inputs). 
For more detailed information, you can look at the [source code of (transposed) convolution](https://github.com/jonas208/GradValley.jl/blob/main/src/functional/gv_convolution.jl).
A nice looking visualization of (transposed) convolution can be found [here](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md).

# Examples
```julia-repl
# square kernels and fully default values of keyword arguments
julia> m = ConvTranspose(6, 3, (5, 5))
# upsampling an output from normal convolution like in DCGANS, Unet, etc.
julia> input = forward(Conv(3, 6, (5, 5)), rand(32, 3, 50, 50))
julia> output = forward(m, input)
# the size of the output of the transposed convolution is equal to the size of the original input of the normal convolution
julia> size(output)
(32, 3, 50, 50)
```
"""
mutable struct ConvTranspose
    # characteristics of the layer
    in_channels::Int
    out_channels::Int
    kernel_size::Tuple{Int, Int}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    output_padding::Tuple{Int, Int}
    dilation::Tuple{Int, Int}
    groups::Int
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Function, Int} # derivative of activation function
    use_bias::Bool
    # data
    inputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    kernels::Array{Float64, 4} # weights
    bias::Vector{Float64} # bias
    outputs_no_activation::Union{Nothing, Array{Float64, 4}} # can be nothing
    outputs::Union{Nothing, Array{Float64, 4}} # can be nothing
    losses::Union{Nothing, Array{Float64, 4}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64, 4}} # losses for the previous layer, can be nothing
    gradients::Array{Float64, 4} # gradients of the kernels/weights
    bias_gradients::Vector{Float64}
    # custom constructor
    function ConvTranspose(in_channels::Int, out_channels::Int, kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=(1, 1), padding::Tuple{Int, Int}=(0, 0), output_padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), groups::Int=1, activation_function::Union{Nothing, String}=nothing, init_mode::String="default_uniform", use_bias::Bool=true) # init_mode::String="kaiming"
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
            error("GradValley: ConvTranspose: output_padding must be smaller than either stride or dilation, but got invalid values: y_output_padding: $y_out_padding x_ou_padding: $x_out_padding y_stride: $y_stride x_stride: $x_stride y_dilation: $y_dilation x_dilation: $x_dilation")
        end

        # initialize kernels/weights
        kernels_shape = (in_channels, convert(Int, out_channels / groups), kernel_size[1], kernel_size[2])
        bias_shape = (out_channels, )
        kernels, bias = general_weight_and_bias_init(kernels_shape, bias_shape, init_mode, gain)
        # initialize gradients of kernels/weights and bias
        gradients = zeros(kernels_shape)
        bias_gradients = zeros(bias_shape)

        # placeholders
        inputs = nothing
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
            output_padding,
            dilation, 
            groups,
            new_activation_function, 
            df,
            use_bias,
            inputs,
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

# forward function for a transpose convolution layer (ConvTranspose)
# Shape of inputs: (batch_size, in_channels, height, width)
function forward(conv_layer::ConvTranspose, inputs::Array{Float64, 4})
    # inputs = copy(inputs)
    conv_layer.inputs = inputs
    outputs_no_activation = gv_functional.multichannel_conv_transpose(inputs, conv_layer.kernels, conv_layer.bias, conv_layer.use_bias, stride=conv_layer.stride, padding=conv_layer.padding, output_padding=conv_layer.output_padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
    if !(isnothing(conv_layer.activation_function))
        outputs = conv_layer.activation_function(outputs_no_activation)
    else
        outputs = outputs_no_activation
    end

    # saving the results of forward computation in the layer struct (mutable)
    conv_layer.outputs_no_activation = outputs_no_activation
    conv_layer.outputs = outputs

    return outputs
end

# save current losses function for a transpose convolution layer (ConvTranspose)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(conv_layer::ConvTranspose, next_layer)
    conv_layer.losses = next_layer.previous_losses
end

# compute previous losses function for a transpose convolution layer (ConvTranspose)
# computes the losses for the previous layer
function compute_previous_losses(conv_layer::ConvTranspose)
    conv_layer.previous_losses = gv_functional.multichannel_conv_transpose_losses(conv_layer)
end

# compute gradients function for a transpose convolution layer (ConvTranspose)
# computes the gradients of the kernels/weights of the given layer -> the gradients are not reset but added to the existing gradients
# conv_layer.losses must have a valid content (not nothing), usally losses were given by calling save_current_losses() before
function compute_gradients(conv_layer::ConvTranspose)
    conv_layer.gradients += gv_functional.multichannel_conv_transpose_gradients(conv_layer)
    # conv_layer.gradients = gv_functional.multichannel_conv_transpose_gradients(conv_layer)
    if conv_layer.use_bias
        conv_layer.bias_gradients = gv_functional.multichannel_conv_transpose_bias_gradients(conv_layer)
    end
end

# zero gradients function for a transpose convolution layer (ConvTranspose)
# resets the gradients of the given layer
function zero_gradients(conv_layer::ConvTranspose)
    conv_layer.gradients = zeros(size(conv_layer.kernels))
    conv_layer.bias_gradients = zeros(size(conv_layer.bias))
end

# backward function for a transpose convolution layer (ConvTranspose)
# calls save_current_losses(), compute_previous_losses(), compute_gradients() -> so like a shortcut for calling these functions seperatly
function backward(conv_layer::ConvTranspose, next_layer)
    save_current_losses(conv_layer, next_layer)
    compute_previous_losses(conv_layer)
    compute_gradients(conv_layer)
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
julia> output = forward(m, input)
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
julia> output = forward(m, input)
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
julia> output = forward(m, input)
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
julia> output = forward(m, input)
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

@doc raw"""
    Fc(in_features::Int, out_features::Int; activation_function::Union{Nothing, String}=nothing, init_mode::String="default_uniform", use_bias::Bool=true)

A fully connected layer (sometimes also known as dense or linear). Apply a linear transformation (matrix multiplication) to the input signal with additional batch dimension.
This layer currently (!) only accepts Float64 array inputs. 

# Arguments
- `in_features::Int`: the size of each input sample (*"number of input neurons"*)
- `out_features::Int`: the size of each output sample (*"number of output neurons"*)
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the output
- `init_mode::String="default_uniform"`: the initialization mode of the weights
    (can be `"default_uniform"`, `"default"`, `"kaiming_uniform"`, `"kaiming"`, `"xavier_uniform"` or `"xavier"`)
`use_bias::Bool=true`: if true, adds a learnable bias to the output

# Shapes
- Input: ``(N, in\_features)``
- Weight: ``(out\_features, in\_features)``
- Bias: ``(out\_features, )``
- Output: ``(N, out\_features)``

# Useful Fields/Variables
- `weights::Array{Float64, 2}`: the learnable weights of the layer
- `bias::Vector{Float64}`: the learnable bias of the layer (used when `use_bias=true`)
- `gradients::Array{Float64, 2}`: the current gradients of the weights
- `bias_gradients::Vector{Float64}`: the current gradients of the bias

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
julia> input = rand(32, 784)
julia> output = forward(m, input)
```
"""
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

@doc raw"""
    Reshape(out_shape; activation_function::Union{Nothing, String}=nothing)

A reshape layer (probably mostly used as a flatten layer). Reshape the input signal (effects all dimensions except the batch dimension).
This layer currently (!) only accepts Float64 array inputs. 

# Arguments
- `out_shape`: the target output size (the output has the same data as the input and must have the same number of elements)
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the output

# Shapes
- Input: ``(N, *)``, where * means any number of dimensions
- Output: ``(N, out\_shape...)``

# Definition
This layer uses the standard [reshape function](https://docs.julialang.org/en/v1/base/arrays/#Base.reshape) inbuilt in Julia.

# Examples
```julia-repl
# flatten the input of size 1*28*28 to a vector of length 784 (each plus batch dimension of course)
julia> m = Reshape((784, ))
# computing the output of the layer (with random inputs)
julia> input = rand(32, 1, 28, 28)
julia> output = forward(m, input)
```
"""
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

@doc raw"""
    BatchNorm2d(num_features::Int; epsilon::Float64=1e-05, momentum::Float64=0.1, affine::Bool=true, track_running_stats::Bool=true, activation_function::Union{Nothing, String}=nothing)

A batch normalization layer. Apply a batch normalization over a 4D input signal (a mini-batch of 2D inputs with additional channel dimension).
This layer currently (!) only accepts Float64 array inputs. 

This layer has two modes: training mode and test mode. If `track_running_stats::Bool=true`, this layer behaves differently in the two modes.
During training, this layer always uses the currently calculated batch statistics. If `track_running_stats::Bool=true`, the running mean and variance are tracked
during training and will be used while testing. If `track_running_stats::Bool=false`, even in test mode, the currently calculated batch statistics are used.
The mode can be switched with [`trainmode!`](@ref) or [`testmode!`](@ref) respectively. The training mode is active by default.

# Arguments
- `num_features::Int`: the number of channels
- `epsilon::Float64=1e-05`: a value added to the denominator for numerical stability
- `momentum::Float64=0.1`: the value used for the running mean and running variance computation
- `affine::Bool=true`: if true, this layer uses learnable affine parameters/weights (``\gamma`` and ``\beta``)
- `track_running_stats::Bool=true`: if true, this layer tracks the running mean and variance during training and will use them for testing/evaluation, if false, such statistics are not tracked and, even in test mode, the batch statistics are always recalculated for each new input
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the output

# Shapes
- Input: ``(N, C, H, W)``
- ``\gamma`` Weight, ``\beta`` Bias: ``(C, )``
- Running Mean/Variance: ``(C, )``
- Output: ``(N, C, H, W)`` (same shape as input)

# Useful Fields/Variables
## Weights (used if `affine::Bool=true`)
- `weight_gamma::Vector{Float64}`: ``\gamma``, a learnabele parameter for each channel, initialized with ones
- `weight_beta::Vector{Float64}`: ``\beta``, a learnabele parameter for each channel, initialized with zeros
## Gradients of weights (used if `affine::Bool=true`)
- `gradient_gamma::Vector{Float64}`: the gradients of ``\gamma``
- `gradient_beta::Vector{Float64}`: the gradients of ``\beta``
## Running statistics (used if `rack_running_stats::Bool=true`)
- `running_mean::Vector{Float64}`: the continuously updated batch statistics of the mean
- `running_variance::Vector{Float64}`: the continuously updated batch statistics of the variance

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
julia> m = BatchNorm2d(3)
# the mode can be switched with trainmode! or testmode!
julia> trainmode!(m)
julia> testmode!(m)
# computing the output of the layer (with random inputs)
julia> input = rand(32, 1, 28, 28)
julia> output = forward(m, input)
```
"""
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

@doc raw"""
    Softmax(; dim::Integer=1)

A softmax activation function layer (probably mostly used at the "end" of a classifier model). Apply the softmax function to an n-dimensional input array.
The softmax will be computed along the given dimension (`dim::Integer`), so every slice along that dimension will sum to 1.
This layer currently (!) only accepts Float64 array inputs. 

!!! note
    Note that this is the only activation function in form of a layer. All other activation functions can be used with the `activation_function::String`
    keyword argument nearly every layer provides. All the activation functions which can be used that way are simple element-wise activation functions.
    Softmax is currently the only non-element-wise activation function. Besides it is very important to be able to select a specific dimension along the 
    softmax should be computed. That would also not work well with the use of simple keyword argument taking only a string which is the name of the function.

# Arguments
- `dim::Integer=1`: the dimension along the softmax will be computed (so every slice along that dimension will sum to 1)

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
# the softmax will be computed along the second dimension
julia> m = Softmax(dim=2)
# computing the output of the layer 
# (with random input data which could represent a batch of unnormalized output values from a classifier)
julia> input = rand(32, 10)
julia> output = forward(m, input)
# summing up the values in the output along the second dimension result in a batch of 32 ones
julia> sum(output, dims=2)
32x1 Matrix{Float64}:
1.0
1.0
...
1.0
```
"""
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

#=
Identity layer (with optional activation function)
=#

@doc raw"""
    Identity(; activation_function::Union{Nothing, String}=nothing)

An identity layer (can be used as an activation function layer). If no activation function is used, this layer does not change the signal in any way.
However, if an activation function is used, the activation function will be applied to the inputs element-wise. 
This layer currently (!) only accepts Float64 array inputs. 

!!! tip
    This layer is helpful to apply an element-wise activation independent of a "normal" computational layer.

# Arguments
- `activation_function::Union{Nothing, String}=nothing`: the element-wise activation function which will be applied to the inputs

# Shapes
- Input: ``(*)``, where ``*`` means any number of dimensions
- Output: ``(*)`` (same shape as input)

# Definition
A placeholder identity operator, except the optional activation function, the input signal is not changed in any way.
If an activation function is used, the activation function will be applied to the inputs element-wise. 

# Examples
```julia-repl
# an independent relu activation
julia> m = Identity(activation_function="relu")
# computing the output of the layer (with random inputs)
julia> input = rand(32, 10)
julia> output = forward(m, input)
```
"""
mutable struct Identity
    # characteristics of the layer
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Function, Int} # derivative of activation function
    # data
    outputs::Union{Nothing, Array{Float64}} # can be nothing
    outputs_no_activation::Union{Nothing, Array{Float64}} # can be nothing
    losses::Union{Nothing, Array{Float64}} # can be nothing
    previous_losses::Union{Nothing, Array{Float64}} # losses for the previous layer, can be nothing
    # custom constructor
    function Identity(; activation_function::Union{Nothing, String}=nothing)
        # setting up the activation function
        new_activation_function, df, gain = general_activation_function_init(activation_function)

        # placeholders
        outputs = nothing
        outputs_no_activation = nothing
        losses = nothing
        previous_losses = nothing

        # create new instance/object
        new(new_activation_function, df, outputs, outputs_no_activation, losses, previous_losses)
    end
end

# forward function for a identity layer (Identity)
# Shape of inputs: (*) where * means an arbitrary number of dimensions
function forward(identity_layer::Identity, inputs::Array{Float64}) # inputs::Array, inputs::Array{Float64, N}) where N
    # inputs = copy(inputs)
    outputs_no_activation = inputs
    if !(isnothing(identity_layer.activation_function))
        outputs = identity_layer.activation_function(outputs_no_activation)
    else
        outputs = outputs_no_activation
    end

    # saving the results of forward computation in the layer struct (mutable)
    identity_layer.outputs_no_activation = outputs_no_activation
    identity_layer.outputs = outputs

    return outputs
end

# save current losses function for a identity layer (Identity)
# saves the losses for the current (given) layer in this layer (losses were previously calculated in the next layer)
function save_current_losses(identity_layer::Identity, next_layer)
    identity_layer.losses = next_layer.previous_losses
end

# compute previous losses function for identity layer (Identity)
# computes the losses for the previous layer
function compute_previous_losses(identity_layer::Identity)
    # calculating the derivative of the out_losses
    out_losses = identity_layer.losses
    if identity_layer.df != 1
        out_losses = out_losses .* identity_layer.df(identity_layer.outputs_no_activation)
    end
    identity_layer.previous_losses = out_losses
end

# backward function for a identity layer (Identity)
# calls save_current_losses(), compute_previous_losses() -> so like a shortcut for calling these functions seperatly
function backward(identity_layer::Identity, next_layer)
    save_current_losses(identity_layer, next_layer)
    compute_previous_losses(identity_layer)
end

# struct SequentialContainer
@doc raw"""
    SequentialContainer(layer_stack::Vector{<: Any})

A sequential container (recommended method for building models). A SequtialContainer can take a vector of layers or other SequentialContainers (submodules).
While forward-pass, the given inputs are *sequentially* propagated through every layer (or submodule) and the output will be returned.
The execution order during forward pass is of course the same as the order in the vector containing the layers or submodules.
This container currently (!) only accepts Float64 array inputs. 

!!! note
    You can use a SequentialContainer in a GraphContainer (and vice versa).
    You can also use a SequentialContainer in a SequentialContainer (nesting allowed).

# Arguments
- `layer_stack::Vector{<: Any}`: the vector containing the layers (or submodules, so other Containers), the order of the modules in the vector corresponds to the execution order

# Indexing and Iteration 
The sequential container is indexable and iterable. Indexing one element/iterating behaves like indexing one element of/iterating over 
the `sequential_container.layer_stack` passed to the container at initialization. However, if the index is a range (UnitRange{<: Integer}), 
a new SequentialContainer containing all the requested submodules/layers is initialized and returned. 
`length(sequential_container)` and `size(sequential_container)` both just return the number of modules in the layers vector (equivalent to `length(sequential_container.layer_stack)`).

# Examples
```julia-repl
# a simple chain of fully connected layers
julia> m = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# computing the output of the module (with random inputs)
julia> input = rand(32, 1000)
julia> output = forward(m, input)

# a more complicated example with with nested submodules
julia> feature_extractor_part_1 = SequentialContainer([Conv(1, 6, (5, 5), activation_function="relu"), AvgPool((2, 2))])
julia> feature_extractor_part_2 = SequentialContainer([Conv(6, 16, (5, 5), activation_function="relu"), AvgPool((2, 2))])
julia> feature_extractor = SequentialContainer([feature_extractor_part_1, feature_extractor_part_2])
julia> classifier = SequentialContainer([Fc(256, 120, activation_function="relu"), Fc(120, 84, activation_function="relu"), Fc(84, 10)])
julia> m = SequentialContainer([feature_extractor, Reshape((256, )), classifier, Softmax(dim=2)])
# computing the output of the module (with random inputs)
julia> input = rand(32, 1, 28, 28)
julia> output = forward(m, input)

# indexing 
julia> m[start] # returns the feature_extractor_part_1 submodule (SequentialContainer)
julia> m[end] # returns the softmax layer (Softmax)
julia> m[start:end-1] # returns the entire model except the softmax layer (a new SequentialContainer is initialized and returned) 

# if a SequentialContainer contains BatchNorm layers (regardless of whether they are nested somewhere in a submodule or not), 
# the mode of all these layers at once can be switched as follows
julia> trainmode!(m)
julia> testmode!(m)

# if a SequentialContainer contains layers with trainable parameters/weights (what is hopefully in nearly all situations the case),
# regardless of whether they are nested somewhere in a submodule or not, the gradients of all these layers at once can be reset as follows
julia> zero_gradients(m)
```
"""
mutable struct SequentialContainer # pay attention when mutable because when last_stack will be changed during training, the backpropagation process will may not work correctly
    layer_stack::Vector{<: Any} # Union{Conv, MaxPool, AvgPool, Reshape, Fc, BatchNorm2d}
    num_layers::Int
    # previous_losses::Array # required for backpropagation when other sequential containers are contained in an outer sequential container
    previous_losses::Array{Float64} # required for backpropagation when other sequential containers are contained in an outer sequential container
    # custom constructor
    function SequentialContainer(layer_stack::Vector{<: Any}) # Union{Conv, MaxPool, AvgPool, Reshape, Fc, BatchNorm2d}, Union{Conv, MaxPool, AvgPool, Reshape, Fc, BatchNorm2d, SequentialContainer, GraphConainer}
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

"""
    backward(sc::SequentialContainer, derivative_loss::Array{Float64})

The backward function for computing the gradients for a SequentialContainer (highly recommend for model building). The function takes the container (so mostly the whole model)
as the first argument and the derivative of the loss as the second argument. No gradients are returned, they are just saved in the layers the container contains.
All layers/containers currently (!) only accept Float64 array inputs, therefore `derivative_loss::Array{Float64}` must also be a Float64 array (the number of dimensions can differ).

# Examples
```julia-repl
# define a model
julia> m = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# compute the output of the model (with random inputs)
julia> output = forward(m, rand(32, 1000))
# use a loss function (with random data as target values) and save the derivative of the loss
julia> loss, derivative_loss = mse_loss(output, rand(32, 125)) # note that GradValley.Optimization.mse_loss must be imported
# before the gradients are recalculated, the old gradients should always be reset first
zero_gradients(m)
# backpropagation 
julia> backward(model, derivative_loss)
```
"""
function backward(sc::SequentialContainer, derivative_loss::Array{Float64}) # sc stands for SequentialContainer
    # println("called")
    layer_stack = sc.layer_stack
    # last_layer = sc.layer_stack[sc.num_layers]
    last_layer = sc.layer_stack[end]
    if typeof(last_layer) != SequentialContainer && typeof(last_layer) != GraphContainer
        last_layer.losses = derivative_loss
        # if !(typeof(last_layer) == AvgPool || typeof(last_layer) == MaxPool || typeof(last_layer) == AdaptiveAvgPool || typeof(last_layer) == AdaptiveMaxPool || typeof(last_layer) == Reshape || typeof(last_layer) == Softmax)
        if typeof(last_layer) == Conv || typeof(last_layer) == DepthwiseConv || typeof(last_layer) == ConvTranspose || typeof(last_layer) == Fc || typeof(last_layer) == BatchNorm2d            
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
        if typeof(current_layer) != SequentialContainer && typeof(current_layer) != GraphContainer
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

#=
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
=#

### Experimental Features: AD and GraphContainer

include("gv_auto_diff.jl")

# GraphConainer behaves similar to SequentialContainer
# a GraphContainer can be used in itself, a SequentialContainer can be also used in itself
# a SequentialContainer can be used in a GraphContainer and a GraphConainer can be used in a SequentialContainer
# it containes all the layers and their parameters, so saving the model is easy like saving a SequentialContainer
@doc raw"""
    GraphContainer(forward_pass::Function, layer_stack::Vector{<: Any})

A computational graph container (recommended method for building models). A GraphContainer can take a function representing the forward pass of a model
and a vector of layers or other containers (submodules).
While forward-pass, a tracked version of the given inputs are passed through the given forward pass function and the output will be returned.
During forward pass, the computational graph is build by a function overload based automatic differentiation system (AD). During backward pass, this computational graph 
is used to compute the gradients.
This container currently (!) only accepts Float64 array inputs. 

!!! note
    You can use a GraphContainer in a SequentialContainer (and vice versa).
    You can also use a GraphContainer in a GraphContainer (nesting allowed).

!!! warning
    Note that the GraphContainer is an experimental feature. The behavior of this module could change dramatically in the future.
    Using this module can may cause problems.

# Arguments
- `forward_pass::Function`: the function representing the forward pass of a model
- `layer_stack::Vector{<: Any}`: the vector containing the layers (or submodules, so other Containers), the order doesn't matter

# Guidelines
GradValley has its own little, rudimentary function overload based automatic differentiation system based on [ChainRules.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).
It was designed to allow simple modifications of a normal sequential signal flow, which is the basis of most neural networks. 
For example, to be able to implement ResNet's residual connections. So it represents an alternative to data flow layers known from other Deep Learning packages.
In a way, it is similar to the forward function known from every [PyTorch model](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#define-the-class).
Since the AD does not offer that much functionality at this point in time, the following guidelines must be observed:
- The forward pass function must take at least two arguments. The first is the vector containing the layers (which was passed to GraphContainer at initialization). The following arguments (the last could also be a Vararg argument) are the data inputs.
- The forward pass function must be written generically enough to accept arrays of type T<:AbstractArray/real numbers of type T<:Real as input (starting with the second argument).
- Array inputs that are being differentiated cannot be mutated.
- The initialization of new arrays (for example with `zeros` or `rand`) and their use in mix with the input passed to the forward function is not allowed.
- Avoid dot syntax in most cases, there only exist a few differentiation rules for the most basic vectorized operators (.+, .-, .*, ./, .^).

# Examples
```julia-repl
# a simple chain of fully connected layers (equivalent to the first example of SequentialContainer)
julia> layers = [Fc(1000, 500), Fc(500, 250), Fc(250, 125)]
julia> function forward_pass(layers::Vector, input::AbstractArray)
           fc_1, fc_2, fc_3 = layers
           output = forward(fc_1, input)
           output = forward(fc_2, output)
           output = forward(fc_3, output)
           return output
       end
julia> m = GraphContainer(forward_pass, layers)
# computing the output of the module (with random inputs)
julia> input = rand(32, 1000)
julia> output = forward(m, input)

# a more complicated example: implementation of an inverted residual block
julia> layers = [Conv(16, 64, (1, 1), activation_function="relu"), 
                 DepthwiseConv(64, 64, (3, 3), padding=(1, 1), activation_function="relu"), 
                 Conv(64, 16, (1, 1), activation_function="relu")]
julia> function forward_pass(layers::Vector, input::AbstractArray)
           conv_1, depthwise_conv, conv_2 = layers
           output = forward(conv_1, input)
           output = forward(depthwise_conv, output)
           output = forward(conv_2, output)
           output = output + input # residual/skipped connection
           return output
       end
julia> m = GraphContainer(forward_pass, layers)
# computing the output of the module (with random inputs)
julia> input = rand(32, 16, 50, 50)
julia> output = forward(m, input)

# a simple example with a polynomial, just to show that it is possible to use the GraphContainer like an automatic differentiation (AD) tool 
julia> f(layers, x) = 0.5x^3 - 2x^2 + 10
julia> df(x) = 1.5x^2 - 4x # checking the result of the AD with this manually written derivation 
julia> m = GraphContainer(f, [])
julia> y = forward(m, 3)
julia> dydx = backward(m, 1) # in this case, no loss function was used, so we have no gradient information, therefore we use 1 as the so-called seed
1-element Vector{Float64}:
 1.5
julia> manual_dydx = df(3)
1.5
julia> isapprox(dydx[1], manual_dydx)
true

# if a GraphContainer contains BatchNorm layers (regardless of whether they are nested somewhere in a submodule or not), 
# the mode of all these layers at once can be switched as follows
julia> trainmode!(m)
julia> testmode!(m)

# if a GraphContainer contains layers with trainable parameters/weights (what is hopefully in nearly all situations the case),
# regardless of whether they are nested somewhere in a submodule or not, the gradients of all these layers at once can be reset as follows
julia> zero_gradients(m)
```
"""
mutable struct GraphContainer
    layer_stack::Vector{<: Any} # Union{Conv, MaxPool, AvgPool, Reshape, Fc, BatchNorm2d}
    num_layers::Int
    # the function which defines the forward pass, 
    # any variables are used in the function should either be passed directly to the function or reside in the layers vector
    forward_pass::Function 
    tracked_inputs::Vector{Union{TrackedReal, TrackedArray}} # saved in the grc just because to acces the gradients to the inputs after backward pass easily 
    tracked_output::Union{TrackedReal, TrackedArray} # contains the computational graph 
    previous_losses::Union{Array{Float64}, Real} # required for backpropagation when other containers are contained in an outer container, ::Array{Float64}
    # custom constructor
    function GraphContainer(forward_pass::Function, layer_stack::Vector{<: Any}) # Union{Conv, MaxPool, AvgPool, Reshape, Fc, BatchNorm2d}, Union{Conv, MaxPool, AvgPool, Reshape, Fc, BatchNorm2d, SequentialContainer, GraphConainer}
        num_layers = length(layer_stack)
        # create new instance/object
        new(layer_stack, num_layers, forward_pass)
    end
end

function forward(grc::GraphContainer, inputs::Vararg{Union{Array{Float64}, Real}})
    tracked_inputs = [TrackedWithGradient(input) for input in inputs]
    grc.tracked_inputs = tracked_inputs
    tracked_output = grc.forward_pass(grc.layer_stack, tracked_inputs...)
    grc.tracked_output = tracked_output
    primal_output = primal(tracked_output)

    return primal_output
end

"""
    backward(grc::GraphContainer, derivative_loss::Union{Array{Float64}, Real})

The backward function for computing the gradients for a GraphContainer (recommend for model building). The function takes the container (so mostly the whole model)
as the first argument and the derivative of the loss as the second argument. The gradients are returned (in a vector, in the same order as the inputs were passed to the `forward` function), they are also saved in the layers the container contains.
All layers/containers currently (!) only accept Float64 array inputs, therefore `derivative_loss::Union{Array{Float64}, Real}` must also be a Float64 array (the number of dimensions can differ).

# Examples
```julia-repl
# define a model
julia> layers = [Fc(1000, 500), Fc(500, 250), Fc(250, 125)]
julia> function forward_pass(layers::Vector, input::AbstractArray)
           fc_1, fc_2, fc_3 = layers
           output = forward(fc_1, input)
           output = forward(fc_2, output)
           output = forward(fc_3, output)
           return output
       end
julia> m = GraphContainer(forward_pass, layers)
# compute the output of the model (with random inputs)
julia> output = forward(m, rand(32, 1000))
# use a loss function (with random data as target values) and save the derivative of the loss
julia> loss, derivative_loss = mse_loss(output, rand(32, 125)) # note that GradValley.Optimization.mse_loss must be imported
# before the gradients are recalculated, the old gradients should always be reset first
zero_gradients(m)
# backpropagation 
julia> input_gradient = backward(model, derivative_loss)
```
"""
function backward(grc::GraphContainer, derivative_loss::Union{Array{Float64}, Real})
    tracked_backward(grc.tracked_output, derivative_loss)
    # save the gradients to the inputs (arguments) of the forward pass function (exept the first "layers" argument)
    gradients = [tracked_input.gradient for tracked_input in grc.tracked_inputs]
    # save the first gradients in the grc, so a GraphContainer behaves somewhat like a layer
    grc.previous_losses = gradients[1]

    return gradients
end

# rrules and special forward functions for layers and containers from GradValley.Layers  
include("gv_auto_diff_rules.jl")

#=
Functions for SequentialContainer and GraphContainer
(were combined because they would otherwise be very similar to each other)
=#

# resets the the gradients of all Conv/DepthwiseConv/ConvTranspose/Fc/BatchNorm2d layers in the given container
function zero_gradients(container::Union{SequentialContainer, GraphContainer})
    for layer in container.layer_stack
        if typeof(layer) == Conv || typeof(layer) == DepthwiseConv || typeof(layer) == ConvTranspose || typeof(layer) == Fc || typeof(layer) == BatchNorm2d || typeof(layer) == SequentialContainer || typeof(layer) == GraphContainer
            zero_gradients(layer)
        end
    end
end

# if the given Container contains BatchNorm2d layers, there mode will be set to trainmode
function trainmode!(container::Union{SequentialContainer, GraphContainer})
    for layer in container.layer_stack
        if typeof(layer) == BatchNorm2d || typeof(layer) == SequentialContainer || typeof(layer) == GraphContainer
            trainmode!(layer)
        end
    end
end

# if the given Container contains BatchNorm2d layers, there mode will be set to testmode
function testmode!(container::Union{SequentialContainer, GraphContainer})
    for layer in container.layer_stack
        if typeof(layer) == BatchNorm2d || typeof(layer) == SequentialContainer || typeof(layer) == GraphContainer
            testmode!(layer)
        end
    end
end

# extracts all layers in a stack of (nested) containers and layers recursively, returns a vector only containing all the pure layers
function extract_layers(container::Union{SequentialContainer, GraphContainer}, layer_stack)
    for layer in container.layer_stack
        if typeof(layer) == SequentialContainer || typeof(layer) == GraphContainer
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
    elseif typeof(layer) == Identity
        summary *= "activation_function=$(layer.activation_function)"
    end
    
    return summary, num_params
end

# returns a string (and the total number of parameters) with an overview of the container (currently doesn't show an visualization of the computational graph)
function summarize_container(container::Union{SequentialContainer, GraphContainer}, sub_counter::String)
    if typeof(container) == SequentialContainer
        summary = "SequentialContainer\n(\n"
    else # GraphContainer
        summary = "GraphContainer\n(\n"
    end
    # summary = "SequentialContainer\n(\n"
    num_params = 0
    for (i, layer) in enumerate(container.layer_stack)
        if sub_counter == ""
            sub_counter_new = string(i)
        else
            sub_counter_new = sub_counter * "." * string(i)
        end
        #=
        if typeof(layer) != SequentialContainer
            layer_summary, layer_num_params = get_layer_summary(layer)
            summary *= "($sub_counter_new) $layer_summary\n"
        else
            layer_summary, layer_num_params = summarize_container(layer, sub_counter_new) # works recursively
            lines = split(layer_summary, "\n")
            summary *= "($sub_counter_new) " * lines[1] * "\n"
            for line in lines[2:end]
                summary *= "     " * line * "\n"
            end
        end
        =#
        if typeof(layer) == SequentialContainer || typeof(layer) == GraphContainer
            layer_summary, layer_num_params = summarize_container(layer, sub_counter_new) # works recursively
            lines = split(layer_summary, "\n")
            summary *= "($sub_counter_new) " * lines[1] * "\n"
            for line in lines[2:end]
                summary *= "     " * line * "\n"
            end
        else
            layer_summary, layer_num_params = get_layer_summary(layer)
            summary *= "($sub_counter_new) $layer_summary\n"
        end
        num_params += layer_num_params
    end
    summary *= ")" # ")\n"

    return summary, num_params
end

# returns a string (and the total number of parameters) intended for printing with an overview of the model (currently doesn't show an visualization of the computational graph) and its number of parameters
"""
    summarize_model(container::Union{SequentialContainer, GraphContainer})

Return a string (and the total number of parameters) intended for printing with an overview of the model 
(currently doesn't show an visualization of the computational graph) and its number of parameters.
"""
function summarize_model(container::Union{SequentialContainer, GraphContainer})
    summary, num_params = summarize_container(container, "")
    summary *= "\n"
    summary *= "Total Layers: $(length(extract_layers(container, [])))\n"
    summary *= "Total Parameters: $num_params"

    return summary, num_params
end

end # end of module "Layers"