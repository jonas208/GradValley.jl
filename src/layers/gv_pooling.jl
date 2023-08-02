#= 
Pooling Layers (MaxPool & AvgPool)
=#

@doc raw"""
    MaxPool(kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), activation_function::Union{Nothing, AbstractString}=nothing)

A maximum pooling layer. Apply a 2D maximum pooling over an input signal with additional batch and channel dimensions.

# Arguments
- `kernel_size::NTuple{2, Int}`: the size of the window to take the maximum over
- `stride::NTuple{2, Int}=kernel_size`: the stride of the window
- `padding::NTuple{2, Int}=(0, 0)`: the zero padding added to all four sides of the input
- `dilation::NTuple{2, Int}=(1, 1)`: the spacing between the window elements
- `activation_function::Union{Nothing, AbstractString}=nothing`: the element-wise activation function which will be applied to the output after the pooling

# Shapes
- Input: ``(W_{in}, H_{in}, C, N)``
- Output: ``(W_{out}, H_{out}, C, N)``
    - ``H_{out} = {\frac{H_{in} + 2 \cdot padding[1] - dilation[1] \cdot (H_w - 1) - 1}{stride[1]}} + 1``
    - ``W_{out} = {\frac{W_{in} + 2 \cdot padding[2] - dilation[2] \cdot (W_w - 1) - 1}{stride[2]}} + 1``

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
julia> input = rand(50, 50, 3, 32)
julia> output = forward(m, input)
```
"""
mutable struct MaxPool <: AbstractNonParamLayer
    # characteristics of the layer
    kernel_size::NTuple{2, Int}
    stride::NTuple{2, Int}
    padding::NTuple{2, Int}
    dilation::NTuple{2, Int}
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Nothing, Function} # derivative of activation function
    # custom constructor
    function MaxPool(kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), activation_function::Union{Nothing, AbstractString}=nothing)
        kernel_height, kernel_width = kernel_size
        padding_height, padding_width = padding
        if !(padding_height <= kernel_height ÷ 2 && padding_width <= kernel_width ÷ 2) # how to round?
            error("GradValley: MaxPool: padding should be smaller than or equal to half of kernel size, but got padding_height = $padding_height, padding_width = $padding_width,
                kernel_height = $kernel_height, kernel_width = $kernel_width")
        end

        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # create new instance/object
        new(kernel_size, 
            stride, 
            padding,
            dilation,
            new_activation_function, 
            df
        )
    end
end

function forward(pool_layer::MaxPool, input::AbstractArray{T, 4}) where T
    output_no_activation = Functional.maximum_pooling2d(input, pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
    if !isnothing(pool_layer.activation_function)
        output = pool_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    return output
end

function (pool_layer::MaxPool)(input::AbstractArray{T, 4}) where T
    return forward(pool_layer, input)
end

function forward(layer::MaxPool, input::TrackedArray{T, 4}) where T
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), pool_layer::MaxPool, input::AbstractArray{T, 4}) where T
    # doing the forward pass of the layer

    if typeof(input) <: CuArray
        output_no_activation = Functional.maximum_pooling2d(input, pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
    else
        # positions are only necessary for backpropagation on the cpu, stores the positions of the max values (for every output)
        output_no_activation, positions = Functional.maximum_pooling2d(input, pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation, return_data_for_backprop=true)
    end

    if !isnothing(pool_layer.activation_function)
        output = pool_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    function forward_pullback(output_gradient::AbstractArray{T, 4}) where T
        # doing the backpropagation of the pool_layer

        if !isnothing(pool_layer.df)
            output_gradient = output_gradient .* pool_layer.df(output_no_activation)
        end

        if typeof(input) <: CuArray
            input_gradient = Functional.maximum_pooling2d_backward(output_gradient, output_no_activation, input, pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
        else
            input_gradient = Functional.maximum_pooling2d_backward(output_gradient, input, positions, padding=pool_layer.padding)
        end

        # defining gradients
        forward_gradient = NoTangent()
        pool_layer_gradient = NoTangent()

        return forward_gradient, pool_layer_gradient, input_gradient
    end

    return output, forward_pullback
end

@doc raw"""
    AvgPool(kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), activation_function::Union{Nothing, AbstractString}=nothing)

An average pooling layer. Apply a 2D average pooling over an input signal with additional batch and channel dimensions.

# Arguments
- `kernel_size::NTuple{2, Int}`: the size of the window to take the average over
- `stride::NTuple{2, Int}=kernel_size`: the stride of the window
- `padding::NTuple{2, Int}=(0, 0)`: the zero padding added to all four sides of the input
- `dilation::NTuple{2, Int}=(1, 1)`: the spacing between the window elements
- `activation_function::Union{Nothing, AbstractString}=nothing`: the element-wise activation function which will be applied to the output after the pooling

# Shapes
- Input: ``(W_{in}, H_{in}, C, N)``
- Output: ``(W_{out}, H_{out}, C, N)``
    - ``H_{out} = {\frac{H_{in} + 2 \cdot padding[1] - dilation[1] \cdot (H_w - 1) - 1}{stride[1]}} + 1``
    - ``W_{out} = {\frac{W_{in} + 2 \cdot padding[2] - dilation[2] \cdot (W_w - 1) - 1}{stride[2]}} + 1``

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
julia> input = rand(50, 50, 3, 32)
julia> output = forward(m, input)
```
"""
mutable struct AvgPool <: AbstractNonParamLayer
    # characteristics of the layer
    kernel_size::NTuple{2, Int}
    stride::NTuple{2, Int}
    padding::NTuple{2, Int}
    dilation::NTuple{2, Int}
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Nothing, Function} # derivative of activation function
    # custom constructor
    function AvgPool(kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), activation_function::Union{Nothing, AbstractString}=nothing)
        kernel_height, kernel_width = kernel_size
        padding_height, padding_width = padding
        if !(padding_height <= kernel_height ÷ 2 && padding_width <= kernel_width ÷ 2) # how to round?
            error("GradValley: MaxPool: padding should be smaller than or equal to half of kernel size, but got padding_height = $padding_height, padding_width = $padding_width,
                kernel_height = $kernel_height, kernel_width = $kernel_width")
        end

        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # create new instance/object
        new(kernel_size, 
            stride, 
            padding,
            dilation,
            new_activation_function, 
            df
        )
    end
end

function forward(pool_layer::AvgPool, input::AbstractArray{T, 4}) where T
    output_no_activation = Functional.average_pooling2d(input, pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
    if !isnothing(pool_layer.activation_function)
        output = pool_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    return output
end

function (pool_layer::AvgPool)(input::AbstractArray{T, 4}) where T
    return forward(pool_layer, input)
end

function forward(layer::AvgPool, input::TrackedArray{T, 4}) where T
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), pool_layer::AvgPool, input::AbstractArray{T, 4}) where T
    # doing the forward pass of the layer

    output_no_activation = Functional.average_pooling2d(input, pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
    if !isnothing(pool_layer.activation_function)
        output = pool_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    function forward_pullback(output_gradient::AbstractArray{T, 4}) where T
        # doing the backpropagation of the pool_layer

        if !isnothing(pool_layer.df)
            output_gradient = output_gradient .* pool_layer.df(output_no_activation)
        end

        if typeof(input) <: CuArray
            input_gradient = Functional.average_pooling2d_backward(output_gradient, output_no_activation, input, pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
        else
            input_gradient = Functional.average_pooling2d_backward(output_gradient, input, pool_layer.kernel_size, stride=pool_layer.stride, padding=pool_layer.padding, dilation=pool_layer.dilation)
        end

        # defining gradients
        forward_gradient = NoTangent()
        pool_layer_gradient = NoTangent()

        return forward_gradient, pool_layer_gradient, input_gradient
    end

    return output, forward_pullback
end

#= 
Adaptive Pooling Layers (AdaptiveMaxPool & AdaptiveAvgPool)
=#

@doc raw"""
    AdaptiveMaxPool(output_size::NTuple{2, Int}; activation_function::Union{Nothing, AbstractString}=nothing)

An adaptive maximum pooling layer. Apply a 2D adaptive maximum pooling over an input signal with additional batch and channel dimensions.
For any input size, the spatial size of the output is always equal to the specified ``output\_size``.

# Arguments
- `output_size::NTuple{2, Int}`: the target output size of the image (can even be larger than the input size) of the form ``(H_{out}, W_{out})``
- `activation_function::Union{Nothing, AbstractString}=nothing`: the element-wise activation function which will be applied to the output after the pooling

# Shapes
- Input: ``(W_{in}, H_{in}, C, N)``
- Output: ``(W_{out}, H_{out}, C, N)``, where ``(H_{out}, W_{out}) = output\_size``

# Definition
In some cases, the kernel-size and stride could be calculated in a way that the output would have the target size 
(using a standard maximum pooling with the calculated kernel-size and stride, padding and dilation would not 
be used in this case). However, this approach would only work if the input size is an integer multiple of the output size (See this question at stack overflow
for further information: [stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work](https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work)).
A more generic approach is to calculate the indices of the input with an additional algorithm only for adaptive pooling. 
With this approach, it is even possible that the output is larger than the input what is really unusual for pooling simply because that is the opposite
of what pooling actually should do, namely reducing the size. The `function get_in_indices(in_len, out_len)` in 
[`gv_functional.jl`](https://github.com/jonas208/GradValley.jl/blob/main/src/gv_functional.jl)
(line 68 - 85) implements such an algorithm (similar to the one at the stack overflow question), so you could check there on how exactly it is defined.
Thus, the mathematical definition would be identical to the one at [`MaxPool`](@ref) with the difference that the indices ``y_{in}`` and ``x_{in}`` 
have already been calculated beforehand.

# Examples
```julia-repl
# target output size of 5x5
julia> m = AdaptiveMaxPool((5, 5))
# computing the output of the layer (with random inputs)
julia> input = rand(50, 50, 3, 32)
julia> output = forward(m, input)
```
"""
mutable struct AdaptiveMaxPool <: AbstractNonParamLayer
    # characteristics of the layer
    output_size::NTuple{2, Int}
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Nothing, Function} # derivative of activation function
    # custom constructor
    function AdaptiveMaxPool(output_size::NTuple{2, Int}; activation_function::Union{Nothing, AbstractString}=nothing)
        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # create new instance/object
        new(output_size,
            new_activation_function, 
            df
        )
    end
end

function forward(pool_layer::AdaptiveMaxPool, input::AbstractArray{T, 4}) where T
    output_no_activation = Functional.adaptive_maximum_pooling2d(input, pool_layer.output_size)
    if !isnothing(pool_layer.activation_function)
        output = pool_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    return output
end

function (pool_layer::AdaptiveMaxPool)(input::AbstractArray{T, 4}) where T
    return forward(pool_layer, input)
end

function forward(layer::AdaptiveMaxPool, input::TrackedArray{T, 4}) where T
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), pool_layer::AdaptiveMaxPool, input::AbstractArray{T, 4}) where T
    # doing the forward pass of the layer

    if typeof(input) <: CuArray
        output_no_activation = Functional.adaptive_maximum_pooling2d(input, pool_layer.output_size)
    else
        # positions are only necessary for backpropagation on the cpu, stores the positions of the max values (for every output)
        output_no_activation, positions = Functional.adaptive_maximum_pooling2d(input, pool_layer.output_size, return_data_for_backprop=true)
    end

    if !isnothing(pool_layer.activation_function)
        output = pool_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    function forward_pullback(output_gradient::AbstractArray{T, 4}) where T
        # doing the backpropagation of the pool_layer

        if !isnothing(pool_layer.df)
            output_gradient = output_gradient .* pool_layer.df(output_no_activation)
        end

        if typeof(input) <: CuArray
            input_gradient = Functional.adaptive_maximum_pooling2d_backward(output_gradient, input)
        else
            input_gradient = Functional.adaptive_maximum_pooling2d_backward(output_gradient, input, positions)
        end

        # defining gradients
        forward_gradient = NoTangent()
        pool_layer_gradient = NoTangent()

        return forward_gradient, pool_layer_gradient, input_gradient
    end

    return output, forward_pullback
end

@doc raw"""
    AdaptiveAvgPool(output_size::NTuple{2, Int}; activation_function::Union{Nothing, AbstractString}=nothing)

An adaptive average pooling layer. Apply a 2D adaptive average pooling over an input signal with additional batch and channel dimensions.
For any input size, the size of the output is always equal to the specified ``output\_size``.

# Arguments
- `output_size::NTuple{2, Int}`: the target output size of the image (can even be larger than the input size) of the form ``(H_{out}, W_{out})``
- `activation_function::Union{Nothing, AbstractString}=nothing`: the element-wise activation function which will be applied to the output after the pooling

# Shapes
- Input: ``(W_{in}, H_{in}, C, N)``
- Output: ``(W_{out}, H_{out}, C, N)``, where ``(H_{out}, W_{out}) = output\_size``

# Definition
In some cases, the kernel-size and stride could be calculated in a way that the output would have the target size 
(using a standard average pooling with the calculated kernel-size and stride, padding and dilation would not 
be used in this case). However, this approach would only work if the input size is an integer multiple of the output size (See this question at stack overflow
for further information: [stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work](https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work)).
A more generic approach is to calculate the indices of the input with an additional algorithm only for adaptive pooling. 
With this approach, it is even possible that the output is larger than the input what is really unusual for pooling simply because that is the opposite
of what pooling actually should do, namely reducing the size. The `function get_in_indices(in_len, out_len)` in 
[`gv_functional.jl`](https://github.com/jonas208/GradValley.jl/blob/main/src/gv_functional.jl)
(line 68 - 85) implements such an algorithm (similar to the one at the stack overflow question), so you could check there on how exactly it is defined.
Thus, the mathematical definition would be identical to the one at [`AvgPool`](@ref) with the difference that the indices ``y_{in}`` and ``x_{in}`` 
have already been calculated beforehand.

# Examples
```julia-repl
# target output size of 5x5
julia> m = AdaptiveAvgPool((5, 5))
# computing the output of the layer (with random inputs)
julia> input = rand(50, 50, 3, 32)
julia> output = forward(m, input)
```
"""
mutable struct AdaptiveAvgPool <: AbstractNonParamLayer
    # characteristics of the layer
    output_size::NTuple{2, Int}
    activation_function::Union{Nothing, Function} # can be nothing
    df::Union{Nothing, Function} # derivative of activation function
    # custom constructor
    function AdaptiveAvgPool(output_size::NTuple{2, Int}; activation_function::Union{Nothing, AbstractString}=nothing)
        # setting up the activation function
        new_activation_function, df, _ = general_activation_function_init(activation_function) # _ is the received gain (not used in this layer)

        # create new instance/object
        new(output_size,
            new_activation_function, 
            df
        )
    end
end

function forward(pool_layer::AdaptiveAvgPool, input::AbstractArray{T, 4}) where T
    output_no_activation = Functional.adaptive_average_pooling2d(input, pool_layer.output_size)
    if !isnothing(pool_layer.activation_function)
        output = pool_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    return output
end

function (pool_layer::AdaptiveAvgPool)(input::AbstractArray{T, 4}) where T
    return forward(pool_layer, input)
end

function forward(layer::AdaptiveAvgPool, input::TrackedArray{T, 4}) where T
    tracked_args = (layer, input)
    output, pullback = rrule(forward, layer, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(forward), pool_layer::AdaptiveAvgPool, input::AbstractArray{T, 4}) where T
    # doing the forward pass of the layer

    output_no_activation = Functional.adaptive_average_pooling2d(input, pool_layer.output_size)
    if !isnothing(pool_layer.activation_function)
        output = pool_layer.activation_function(output_no_activation)
    else
        output = output_no_activation
    end

    function forward_pullback(output_gradient::AbstractArray{T, 4}) where T
        # doing the backpropagation of the pool_layer

        if !isnothing(pool_layer.df)
            output_gradient = output_gradient .* pool_layer.df(output_no_activation)
        end

        input_gradient = Functional.adaptive_average_pooling2d_backward(output_gradient, input)

        # defining gradients
        forward_gradient = NoTangent()
        pool_layer_gradient = NoTangent()

        return forward_gradient, pool_layer_gradient, input_gradient
    end

    return output, forward_pullback
end