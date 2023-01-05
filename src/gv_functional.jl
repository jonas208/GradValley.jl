module gv_functional
using LoopVectorization
using LinearAlgebra

#=
General convention:
in_channels, out_channels: indices for the respective channel types
in_channel, out_channel: numbers for the respective channel types
y_w, x_w are indexes used for kernels/weights (and their gradients)
x_in, y_in are indexes used for inputs
y_out, x_out are indexes used for outputs
m, n are indexes which were calculated by get_input_position()
=#

#=
Internal functions, Internals
=#

# returns the position in an input matrix given by the position in output (e.g. usefull for conv, pool and diffrent	backward-passes)
# output_position and stride must be tuples
function get_input_position(output_position::Tuple, stride::Tuple)
    m = output_position[1] + (stride[1] - 1) * (output_position[1] - 1)
    n = output_position[2] + (stride[2] - 1) * (output_position[2] - 1)

    return m, n
end

# returns the shape of an output from a pooling- or convolution-operation
function calculate_output_shape(input_height::Int, input_width::Int, kernel_height::Int, kernel_width::Int; stride::Tuple=(1, 1), padding::Tuple=(0, 0), dilation::Tuple=(1, 1)) # dilation::Tuple{Int, Int}
    output_height = (input_height + 2 * padding[1] - dilation[1] * (kernel_height - 1) - 1) / stride[1] + 1
    output_width = (input_width + 2 * padding[2] - dilation[2] * (kernel_width - 1) - 1) / stride[2] + 1

    output_height = convert(Int, trunc(output_height))
    output_width = convert(Int, trunc(output_width))

    return output_height, output_width
end

# combine many tuples
tuplejoin(t1::Tuple, t2::Tuple, t3...) = tuplejoin((t1..., t2...), t3...)
tuplejoin(t::Tuple) = t

# creates an array of iterators (UnitRanges) for each dimension without the itertor for the given dimension (dim), the given size_tuple is the size of the orginal array, e.g. useful for dim_sum or softmax & backward_softmax
function get_iters_without_at_dim(size_tuple, dim)
    # checks if dim is a valid value
    num_dims = length(size_tuple)
    if dim == 0 || abs(dim) > num_dims
        error("GradValley: dim_sum: the given dim is out of bounce")
    end
    if dim < 0
        dim = num_dims + 1 - dim
    end
    iterators = UnitRange{Int}[]
    for (index, dim_size) in enumerate(size_tuple)
        if index != dim
            push!(iterators, 1:dim_size)
        end
    end

    return iterators
end

# calculates the sum along a specific dim (removes this dimension in the output),
# negative dim starts counting at the end of all dimensions, so dim=-1 for example is the last dimension
function dim_sum(input; dim=1)
    input_size = size(input)
    num_dims = length(input_size)
    # checks if dim is a valid value
    if dim == 0 || abs(dim) > num_dims
        error("GradValley: dim_sum: the given dim is out of bounce")
    end
    if dim < 0
        dim = num_dims + 1 - dim
    end
    dim_size = input_size[dim]
    output_size = Int[]
    for (dim_index, dim_size) in enumerate(input_size)
        if dim_index != dim
            push!(output_size, dim_size)
        end
    end
    output_size = Tuple(output_size)
    # println(output_size)
    output = zeros(eltype(input), output_size)
    indices_array = Union{UnitRange{Int}, Int}[1:dim_size for dim_size in input_size]
    for index_dim in 1:dim_size
        indices_array[dim] = index_dim
        output .+= input[indices_array...]
    end
    
    return output
end

# internal function for adaptive pooling, returns a list of ranges containing the indices for reading the input array with the correct pooling kernels/windows
function get_in_indices(in_len, out_len)
    get_start_index(a, b, c) = floor((a * c) / b)
    get_end_index(a, b, c) = ceil(((a + 1) * c) / b)
    # indices = UnitRange{Int}[]
    indices = Vector{UnitRange{Int}}(undef, out_len)
    for index_out in 0:out_len - 1
        start_index = get_start_index(index_out, out_len, in_len)
        end_index = get_end_index(index_out, out_len, in_len)
        difference = end_index - start_index
        range_start = convert(Int, trunc(start_index + 1))
        range_end = convert(Int, trunc(start_index + difference)) # + 1
        indices_range = range_start:range_end
        # push!(indices, indices_range)
        indices[index_out + 1] = indices_range
    end

    return indices
end

#=
Padding-Operations
=#

# Performing a padding-operation (nd, number of dimensions doesn't matter) as is usual with neural networks: equal padding one each "end" of an axis/dimension
# Shape of input: (d1, d2, ..., dn)
# padding must always be a tuple with length of the number of dimensions of input: (d1, d2, ..., dn)
# Shape of output: (d1 + padding[0] * 2, d2 + padding[2] * 2, ..., dn + padding[n] * 2)
function zero_pad_nd(input::Array, padding::Tuple)
    # calculate shape of output
    input_shape = size(input)
    output_shape = ()
    for (dim, in_dim_size) in enumerate(input_shape)
        out_dim_size = in_dim_size + padding[dim] * 2
        output_shape = tuplejoin(output_shape, (out_dim_size, ))
    end
    
    output = zeros(output_shape)

    # actual computation
    for position in CartesianIndices(input)
        output_positon = () 
        for (dim, index) in enumerate(Tuple(position))
            output_index = index + padding[dim]
            output_positon = tuplejoin(output_positon, (output_index, ))
        end
        output[CartesianIndex(output_positon)] = input[position]
    end

    return output
end

# MISSING: an optimized version of the zero_pad_nd for (exactly) two dimensions

#=
Convolution-Operations: Forward & Backward
=#

# Performing a single channel convolution
# Shape of input: (height, width)
# Shape of kernel: (height, width)
# stride and padding must be always given as tuples of length 2
function conv(input::AbstractMatrix{<: AbstractFloat}, kernel::AbstractMatrix{<: AbstractFloat}; stride::NTuple{2, Integer}=(1, 1), padding::NTuple{2, Integer}=(0, 0), dilation::NTuple{2, Integer}=(0, 0))
    # storing all the necessary shapes
    input_height, input_width = size(input)
    kernel_height, kernel_width = size(kernel)
    # calculating shape of output
    output_height = (input_height + 2 * padding[1] - dilation[1] * (kernel_height - 1) - 1) / stride[1] + 1
    output_width = (input_width + 2 * padding[2] - dilation[2] * (kernel_width - 1) - 1) / stride[2] + 1

    output_height = convert(Int, trunc(output_height))
    output_width = convert(Int, trunc(output_width))

    output = Array{Float64}(undef, output_height, output_width)

    # performing padding
    if padding != (0, 0)
        input = zero_pad_nd(input, padding)
    end

    y_stride, x_stride = stride
    y_dilation, x_dilation = dilation
    # actual computation
    @turbo for y_out in axes(output, 1), x_out in axes(output, 2) # @tturbo
        m = y_out + (y_stride - 1) * (y_out - 1)
        n = x_out + (x_stride - 1) * (x_out - 1)
        value = 0.00
        for y_w in axes(kernel, 1), x_w in axes(kernel, 1)
            y_in = m + (y_w - 1) * y_dilation
            x_in = n + (x_w - 1) * x_dilation
            value += input[y_in, x_in] * kernel[y_w, x_w]
        end
        output[y_out, x_out] = value
    end

    return output
end

# Performing a multichannel convolution (on a hole batch)
# Shape of input: (batch_size, in_channels, height, width)
# Shape of kernels: (out_channels, in_channels, height, width)
# stride and padding must be always given as tuples of length 2
function multichannel_conv(inputs::Array{Float64, 4}, kernels::Array{Float64, 4}, bias::Vector{Float64}, use_bias::Bool; stride::Tuple{Int, Int}=(1, 1), padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), groups::Int=1)
    # inputs = copy(inputs)
    # storing all the necessary shapes
    current_batch_size::Int, in_channels::Int, input_height::Int, input_width::Int = size(inputs)
    out_channels::Int, in_channels_kernels::Int, kernel_height::Int, kernel_width::Int = size(kernels)
    # calculating shape of output
    output_height::Int, output_width::Int = calculate_output_shape(input_height, input_width, kernel_height, kernel_width, stride=stride, padding=padding, dilation=dilation)

    # performing padding
    if padding != (0, 0)
        new_inputs = Array{Float64}(undef, current_batch_size, in_channels, input_height + 2 * padding[1], input_width + 2 * padding[2])
        for index_batch in 1:current_batch_size
            # padding each in_channel individually
            if padding != (0, 0)
                for in_channel in 1:in_channels
                    new_inputs[index_batch, in_channel, :, :] = zero_pad_nd(inputs[index_batch, in_channel, :, :], padding)
                end
            end
        end
    inputs = new_inputs
    end

    # output::Array{Float64, 4} = Array{Float64, 4}(undef, current_batch_size, out_channels, output_height, output_width)
    output = zeros(current_batch_size, out_channels, output_height, output_width)

    if !use_bias
        bias = zeros(size(bias))
    end
    
    y_stride = stride[1]
    x_stride = stride[2]
    y_dilation = dilation[1]
    x_dilation = dilation[2]
    out_channels_per_group = out_channels ÷ groups
    # actual computation
    @inbounds Threads.@threads for index_batch in 1:current_batch_size
        @turbo for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, y_out in 1:output_height, x_out in 1:output_width
            # m, n = get_input_position((y_out, x_out), stride)
            m = y_out + (y_stride - 1) * (y_out - 1) # ::Int
            n = x_out + (x_stride - 1) * (x_out - 1) # ::Int
            out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
            value = 0.00
            for in_channel_kernel in 1:in_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                # y_in = m + y_w - 1 # no dilation (dilation = 1)
                # x_in = n + x_w - 1 # no dilation (dilation = 1)
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                in_channel_input = in_channel_kernel + (group - 1) * in_channels_kernels
                value += inputs[index_batch, in_channel_input, y_in, x_in] * kernels[out_channel, in_channel_kernel, y_w, x_w]
            end
            # output[index_batch, out_channel, y_out, x_out] = value
            output[index_batch, out_channel, y_out, x_out] = value + bias[out_channel]
        end
    end

    return output, inputs
end

# Functions used for Backpropagation (Convolution)
# The only input each function takes is an instance of a conv layer struct (Conv)
# Because a layer is given, these functions directly work on the hole batch

# Computes the derivative of the inputs on the given layer, the results are used as the losses for the previous layer
function multichannel_conv_losses(conv_layer) # maybe works not yet with padding -> TO TEST!
    # storing all the necessary shapes
    # current_batch_size::Int, in_channels::Int, input_height::Int, input_width::Int = size(conv_layer.inputs)
    current_batch_size::Int, in_channels::Int, input_height::Int, input_width::Int = size(conv_layer.inputs_padded)
    current_batch_size, out_channels::Int, output_height::Int, output_width::Int = size(conv_layer.outputs)
    out_channels, in_channels_kernels::Int, kernel_height::Int, kernel_width::Int = size(conv_layer.kernels)

    # losses for the previous layer
    losses::Array{Float64, 4} = zeros(current_batch_size, in_channels, input_height, input_width)

    out_losses::Array{Float64, 4} = conv_layer.losses
    if conv_layer.df != 1
        out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
    end

    stride::Tuple{Int, Int} = conv_layer.stride
    dilation::Tuple{Int, Int} = conv_layer.dilation
    groups::Int = conv_layer.groups
    y_stride = stride[1]
    x_stride = stride[2]
    y_dilation = dilation[1]
    x_dilation = dilation[2]
    out_channels_per_group = out_channels ÷ groups
    # actual computation
    @inbounds Threads.@threads for index_batch in 1:current_batch_size
    # @tturbo for index_batch in 1:current_batch_size

        @turbo for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, y_out in 1:output_height, x_out in 1:output_width
         # for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, y_out in 1:output_height, x_out in 1:output_width
            # m::Int, n::Int = get_input_position((y_out, x_out), stride)
            m = y_out + (y_stride - 1) * (y_out - 1) # ::Int
            n = x_out + (x_stride - 1) * (x_out - 1) # ::Int
            out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
            for in_channel_kernel in 1:in_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                # y_in = m + y_w - 1 # no dilation (dilation = 1)
                # x_in = n + x_w - 1 # no dilation (dilation = 1)
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                in_channel_input = in_channel_kernel + (group - 1) * in_channels_kernels
                losses[index_batch, in_channel_input, y_in, x_in] += conv_layer.kernels[out_channel, in_channel_kernel, y_w, x_w] * out_losses[index_batch, out_channel, y_out, x_out]
            end
        end

    end
    #=
    @inbounds Threads.@threads for index_batch in 1:current_batch_size
        @turbo for y_out in 1:output_height, x_out in 1:output_width
            m = y_out + (y_stride - 1) * (y_out - 1)
            n = x_out + (x_stride - 1) * (x_out - 1)
            for y_w in 1:kernel_height, x_w in 1:kernel_width
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                for group in 1:groups, in_channel_kernel in 1:in_channels_kernels
                    in_channel_input = in_channel_kernel + (group - 1) * in_channels_kernels
                    losses_value = 0.00
                    for out_channel_per_group in 1:out_channels_per_group
                        out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                        losses_value += conv_layer.kernels[out_channel, in_channel_kernel, y_w, x_w] * out_losses[index_batch, out_channel, y_out, x_out]
                    end
                    losses[index_batch, in_channel_input, y_in, x_in] = losses_value
                end
            end
        end
    end
    =#

    if conv_layer.padding != (0, 0)
        # println(size(losses))
        y_pad = conv_layer.padding[1]
        x_pad = conv_layer.padding[2]
        # @views losses = losses[:, :, y_pad+1:input_height-y_pad, x_pad+1:input_width-x_pad] # scheint auch ohne @views gut und performant zu funktionieren
        losses = losses[:, :, y_pad+1:input_height-y_pad, x_pad+1:input_width-x_pad]
        # println(size(losses))
        # println(size(conv_layer.inputs))
    end
   
    return losses
end

# Computes the derivative of the kernels/weights on the given layer, the results are used to optimize the kernels/weights
function multichannel_conv_gradients(conv_layer)
    # storing all the necessary shapes
    current_batch_size::Int, in_channels::Int, input_height::Int, input_width::Int = size(conv_layer.inputs)
    current_batch_size, out_channels::Int, output_height::Int, output_width::Int = size(conv_layer.outputs)
    out_channels, in_channels_kernels::Int, kernel_height::Int, kernel_width::Int = size(conv_layer.kernels)

    # storing often used data which will be modified
    inputs_padded::Array{Float64, 4} = conv_layer.inputs_padded
    # losses::Array{Float64, 4} = conv_layer.losses

    # calculating the derivative of the out_losses
    out_losses::Array{Float64, 4} = conv_layer.losses
    if conv_layer.df != 1
        out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
    end

    gradients::Array{Float64, 4} = zeros(out_channels, in_channels_kernels, kernel_height, kernel_width)
    stride::Tuple{Int, Int} = conv_layer.stride
    dilation::Tuple{Int, Int} = conv_layer.dilation
    groups::Int = conv_layer.groups
    y_stride::Int = stride[1]
    x_stride::Int = stride[2]
    y_dilation = dilation[1]
    x_dilation = dilation[2]
    out_channels_per_group = out_channels ÷ groups
    #= ANOTHER VERSION
    # actual computation
    # @turbo for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, in_channel_kernel in 1:in_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width # @tturbo
    # @inbounds for group in 1:groups, out_channel_per_group in 1:out_channels_per_group
    @inbounds Threads.@threads for out_channel_per_group in 1:out_channels_per_group
    for group in 1:groups
        @turbo for in_channel_kernel in 1:in_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width # @tturbo
            value = 0.00
            for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                # y_in = m + y_w - 1 # no dilation (dilation = 1)
                # x_in = n + x_w - 1 # no dilation (dilation = 1)
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                in_channel_input = in_channel_kernel + (group - 1) * in_channels_kernels
                value += inputs_padded[index_batch, in_channel_input, y_in, x_in] * out_losses[index_batch, out_channel, y_out, x_out]
            end
            gradients[out_channel, in_channel_kernel, y_w, x_w] = value
        end
    end
    end
    =#
    @inbounds Threads.@threads for out_channel_per_group in 1:out_channels_per_group
        @turbo for group in 1:groups, in_channel_kernel in 1:in_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
            value = 0.00
            for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                in_channel_input = in_channel_kernel + (group - 1) * in_channels_kernels
                value += inputs_padded[index_batch, in_channel_input, y_in, x_in] * out_losses[index_batch, out_channel, y_out, x_out]
            end
            gradients[out_channel, in_channel_kernel, y_w, x_w] = value
        end
    end
    
    return gradients
end

# Computes the derivative of the bias on the given layer, the results are used to optimize the bias
function multichannel_conv_bias_gradients(conv_layer)
    current_batch_size, out_channels, output_height, output_width = size(conv_layer.outputs)

    # calculating derivative of the activation function
    out_losses::Array{Float64, 4} = conv_layer.losses
    if conv_layer.df != 1
        out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
    end

    bias_gradients::Vector{Float64} = conv_layer.bias_gradients
    @turbo for out_channel in 1:out_channels # @inbounds Threads.@threads
        value = 0.00
        for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
            value += out_losses[index_batch, out_channel, y_out, x_out]
        end
        bias_gradients[out_channel] += value
    end

    return bias_gradients
end

#=
Pooling-Operations: Forward & Backward
=#

# Performing a single channel pooling operation (max or avg)
# Shape of input: (height, width)
# kernel_size, stride and padding must be always given as tuples of length 2
function pool(input::Array, kernel_size::Tuple, mode::String; stride::Tuple=nothing, padding::Tuple=(0, 0))
    # setting stride who it is commenly used (when it is not defined)
    if isnothing(stride)
        stride = kernel_size
    end

    # storing all the necessary shapes
    input_height, input_width = size(input)
    kernel_height, kernel_width = kernel_size
    # calculating shape of output
    output_height, output_width = calculate_output_shape(input_height, input_width, kernel_height, kernel_width, stride=stride, padding=padding)

    output = Array{Float64}(undef, output_height, output_width)
    # println(size(output))

    # performing padding
    if padding != (0, 0)
        input = zero_pad_nd(input, padding)
    end

    # positions are necessary for backpropagation (only for max pooling)
    # each position in the output is assigned the position in the input with the highest value
    # (which is also the value in the output matrix at the given position in output)
    y_positions = Array{Int}(undef, output_height, output_width)
    x_positions = Array{Int}(undef, output_height, output_width)
    positions = (y_positions, x_positions)
    # performing the actual computation
    for y_out in 1:output_height, x_out in 1:output_width
        values = []
        m, n = get_input_position((y_out, x_out), stride)
        for y_w in 1:kernel_height, x_w in 1:kernel_width
            value = input[m + y_w - 1, n + x_w - 1]
            # the positions are saved
            if length(values) != 0 && value > maximum(values) # max
                y_positions[y_out, x_out] = m + y_w - 1
                x_positions[y_out, x_out] = n + x_w - 1
            elseif length(values) == 0
                y_positions[y_out, x_out] = m + y_w - 1
                x_positions[y_out, x_out] = n + x_w - 1
            end
            push!(values, value)
        end
        if mode == "max"
            output[y_out, x_out] = maximum(values) # max
        elseif mode == "avg"
            output[y_out, x_out] = sum(values) / length(values)
        else
            error("""pool: mode muste be "max" or "avg" """)
        end
    end

    return output, positions
end

# Performing a multichannel pooling (on a hole batch)
# Shape of input: (batch_size, in_channels, height, width)
# kernel_size, stride and padding must be always given as tuples of length 2
function multichannel_pool_old(inputs::Array{Float64, 4}, kernel_size::Tuple{Int, Int}, mode::String; stride::Union{Nothing, Tuple{Int, Int}}=nothing, padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1))
    # setting stride who it is commenly used (when it is not defined)
    if isnothing(stride)
        stride::Tuple{Int, Int} = kernel_size
    end

    # storing all the necessary shapes
    current_batch_size::Int, channels::Int, input_height::Int, input_width::Int = size(inputs)
    kernel_height::Int, kernel_width::Int = kernel_size
    output_height::Int, output_width::Int = calculate_output_shape(input_height, input_width, kernel_height, kernel_width, stride=stride, padding=padding, dilation=dilation)

    output::Array{Float64, 4} = Array{Float64, 4}(undef, current_batch_size, channels, output_height, output_width)

    # performing padding
    if padding != (0, 0)
        new_inputs = Array{Float64, 4}(undef, current_batch_size, channels, input_height + 2 * padding[1], input_width + 2 * padding[2]) # Array{Float64}
        for index_batch in 1:current_batch_size
            # padding each in_channel individually
            if padding != (0, 0)
                for channel in 1:channels
                    new_inputs[index_batch, channel, :, :] = zero_pad_nd(inputs[index_batch, channel, :, :], padding)
                end
            end
        end
    inputs = new_inputs
    end

    y_stride = stride[1]
    x_stride = stride[2]
    y_dilation = dilation[1]
    x_dilation = dilation[2]
    # positions are necessary for backpropagation (only for max pooling)
    # each position in the output is assigned the position in the input with the highest value
    # (which is also the value in the output matrix at the given position in output)
    y_positions::Array{Int, 4} = Array{Int, 4}(undef, current_batch_size, channels, output_height, output_width)
    x_positions::Array{Int, 4} = Array{Int, 4}(undef, current_batch_size, channels, output_height, output_width)
    positions::Tuple{Array{Int, 4}, Array{Int, 4}} = (y_positions, x_positions)
    # going throw all data in batch
    @inbounds Threads.@threads for index_batch in 1:current_batch_size

        # compute pooling for each channel seperatly
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width # @turbo 
            values = Float64[] # ::Array{Float64}, ::Vector{Float64}
            ## values = zeros(kernel_height, kernel_width)
            # m, n = get_input_position((y_out, x_out), stride)
            m = y_out + (y_stride - 1) * (y_out - 1) # ::Int
            n = x_out + (x_stride - 1) * (x_out - 1) # ::Int
            max_y = 0
            max_x = 0
            for y_w in 1:kernel_height, x_w in 1:kernel_width
                # y_in = m + y_w - 1 # ::Int, no dilation (dilation = 1)
                # x_in = n + x_w - 1 # ::Int, no dilation (dilation = 1)
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                value = inputs[index_batch, channel, y_in, x_in] # ::Float64
                # the positions are saved
                if length(values) != 0 && value > maximum(values)
                    # y_positions[index_batch, channel, y_out, x_out] = y_in
                    # x_positions[index_batch, channel, y_out, x_out] = x_in
                    max_y = y_in
                    max_x = x_in
                elseif length(values) == 0
                ## elseif sum(values) == 0
                    # y_positions[index_batch, channel, y_out, x_out] = y_in
                    # x_positions[index_batch, channel, y_out, x_out] = x_in
                    max_y = y_in
                    max_x = x_in
                end
                push!(values, value)
                ## values[y_w, x_w] = value
            end
            # output[index_batch, channel, y_out, x_out] = maximum(values)
            # y_positions[index_batch, channel, y_out, x_out] = max_y
            # x_positions[index_batch, channel, y_out, x_out] = max_x
            if mode == "max"
                output[index_batch, channel, y_out, x_out] = maximum(values)
                y_positions[index_batch, channel, y_out, x_out] = max_y
                x_positions[index_batch, channel, y_out, x_out] = max_x
            elseif mode == "avg"
                output[index_batch, channel, y_out, x_out] = sum(values) / length(values)
                # y_positions[index_batch, channel, y_out, x_out] = max_y
                # x_positions[index_batch, channel, y_out, x_out] = max_x
            else
                error("""pool: mode muste be "max" or "avg" """)
            end
        end

    end

    return output, positions, inputs
end

# Performing a multichannel maximum pooling (on a hole batch)
# Shape of input: (batch_size, in_channels, height, width)
# kernel_size, stride and padding must be always given as tuples of length 2
function multichannel_maxpool(inputs::Array{Float64, 4}, kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=kernel_size, padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1))
    # return multichannel_pool_old(inputs, kernel_size, "max", stride=stride, padding=padding, dilation=dilation)

    current_batch_size::Int, channels::Int, input_height::Int, input_width::Int = size(inputs)
    kernel_height::Int, kernel_width::Int = kernel_size
    output_height::Int, output_width::Int = calculate_output_shape(input_height, input_width, kernel_height, kernel_width, stride=stride, padding=padding, dilation=dilation)

    output::Array{Float64, 4} = Array{Float64, 4}(undef, current_batch_size, channels, output_height, output_width)

    # performing padding
    if padding != (0, 0)
        new_inputs = Array{Float64, 4}(undef, current_batch_size, channels, input_height + 2 * padding[1], input_width + 2 * padding[2]) # Array{Float64}
        for index_batch in 1:current_batch_size
            # padding each in_channel individually
            if padding != (0, 0)
                for channel in 1:channels
                    new_inputs[index_batch, channel, :, :] = zero_pad_nd(inputs[index_batch, channel, :, :], padding)
                end
            end
        end
    inputs = new_inputs
    end

    # positions are necessary for backpropagation (only for max pooling)
    # each position in the output is assigned the position in the input with the highest value
    # (which is also the value in the output matrix at the given position in output)
    y_positions::Array{Int, 4} = Array{Int, 4}(undef, current_batch_size, channels, output_height, output_width)
    x_positions::Array{Int, 4} = Array{Int, 4}(undef, current_batch_size, channels, output_height, output_width)
    positions::Tuple{Array{Int, 4}, Array{Int, 4}} = (y_positions, x_positions)

    y_stride = stride[1]
    x_stride = stride[2]
    y_dilation = dilation[1]
    x_dilation = dilation[2]
    # going throw all data in batch
    # for index_batch in 1:current_batch_size
    @inbounds Threads.@threads for index_batch in 1:current_batch_size

        # compute pooling for each channel seperatly
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
            m = y_out + (y_stride - 1) * (y_out - 1)
            n = x_out + (x_stride - 1) * (x_out - 1)
            # values = Float64[]
            values = zeros(kernel_height, kernel_width)
            max_y = 0
            max_x = 0
            for y_w in 1:kernel_height, x_w in 1:kernel_width
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                value = inputs[index_batch, channel, y_in, x_in]
                # if length(values) != 0 && value > maximum(values)
                if value > maximum(values)
                    max_y = y_in
                    max_x = x_in
                # elseif length(values) == 0
                elseif sum(values) == 0
                    max_y = y_in
                    max_x = x_in
                end
                # push!(values, value)
                values[y_w, x_w] = value
            end
            output[index_batch, channel, y_out, x_out] = maximum(values)
            # the positions are saved
            y_positions[index_batch, channel, y_out, x_out] = max_y
            x_positions[index_batch, channel, y_out, x_out] = max_x
        end

    end

    return output, positions, inputs
end

# Performing a multichannel average pooling (on a hole batch)
# Shape of input: (batch_size, in_channels, height, width)
# kernel_size, stride and padding must be always given as tuples of length 2
function multichannel_avgpool(inputs::Array{Float64, 4}, kernel_size::Tuple{Int, Int}; stride::Tuple{Int, Int}=kernel_size, padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1))
    # storing all the necessary shapes
    current_batch_size::Int, channels::Int, input_height::Int, input_width::Int = size(inputs)
    kernel_height::Int, kernel_width::Int = kernel_size
    output_height::Int, output_width::Int = calculate_output_shape(input_height, input_width, kernel_height, kernel_width, stride=stride, padding=padding, dilation=dilation)

    output::Array{Float64, 4} = Array{Float64, 4}(undef, current_batch_size, channels, output_height, output_width)

    # performing padding
    if padding != (0, 0)
        new_inputs = Array{Float64, 4}(undef, current_batch_size, channels, input_height + 2 * padding[1], input_width + 2 * padding[2]) # Array{Float64}
        for index_batch in 1:current_batch_size
            # padding each in_channel individually
            if padding != (0, 0)
                for channel in 1:channels
                    new_inputs[index_batch, channel, :, :] = zero_pad_nd(inputs[index_batch, channel, :, :], padding)
                end
            end
        end
    inputs = new_inputs
    end

    y_stride = stride[1]
    x_stride = stride[2]
    y_dilation = dilation[1]
    x_dilation = dilation[2]
    # going throw all data in batch
    @inbounds Threads.@threads for index_batch in 1:current_batch_size
    # @tturbo for index_batch in 1:current_batch_size # eigentlich schneller als Threads.@threads gepaart mit @turbo -> vielleicht doch nicht!

        # compute pooling for each channel seperatly
        @turbo for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
        # for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width # kein @turbo hier, stattdessen oben @tturbo, weil eigentlich schneller -> vielleicht doch nicht!
            m = y_out + (y_stride - 1) * (y_out - 1)
            n = x_out + (x_stride - 1) * (x_out - 1)
            kernel_sum = 0.00 # kernel_sum = 0 
            for y_w in 1:kernel_height, x_w in 1:kernel_width
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                kernel_sum += inputs[index_batch, channel, y_in, x_in] # ::Float64
            end
            output[index_batch, channel, y_out, x_out] = kernel_sum / (kernel_height * kernel_width)
        end

    end

    return output, inputs
end

# Functions used for Backpropagation (Pooling)
# The only input each function takes is an instance of a pool layer struct (MaxPool or AvgPool)
# Because a layer is given, these functions directly work on the hole batch

# Computes the backward-pass (derivative of inputs) of a max pooling operation
# The computed losses are the losses for the previous layer
function multichannel_maxpool_backward(pool_layer)
    # storing all the necessary shapes
    current_batch_size::Int, channels::Int, input_height::Int, input_width::Int = size(pool_layer.inputs)
    current_batch_size, channels, output_height::Int, output_width::Int = size(pool_layer.outputs)
    positions::Tuple{Array{Int, 4}, Array{Int, 4}} = pool_layer.positions

    # calculating the derivative of the out_losses
    out_losses::Array{Float64, 4} = pool_layer.losses
    if pool_layer.df != 1
        out_losses = out_losses .* pool_layer.df(pool_layer.outputs_no_activation)
    end

    # losses = Array{Float64}(undef, current_batch_size, channels, input_height, input_width)
    losses::Array{Float64, 4} = zeros(current_batch_size, channels, input_height, input_width)

    # going throw all data in batch
    # for index_batch in 1:current_batch_size
    @inbounds Threads.@threads for index_batch in 1:current_batch_size

        # going throw every output
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
        # for channel in 1:channels
        # Threads.@threads for y_out in 1:output_height
        # Threads.@threads for x_out in 1:output_width
            value::Float64 = out_losses[index_batch, channel, y_out, x_out]
            y_position::Int = positions[1][index_batch, channel, y_out, x_out]
            x_position::Int = positions[2][index_batch, channel, y_out, x_out]
            if !(y_position <= pool_layer.padding[1])
                y_position -= pool_layer.padding[1] # * 2
            end
            if !(x_position <= pool_layer.padding[2])
                x_position -= pool_layer.padding[2] # * 2
            end
            # losses[index_batch, channel, y_position, x_position] = value
            losses[index_batch, channel, y_position, x_position] += value
        end
        # end
        # end 
    
    end

    return losses
end

# Computes the backward-pass (derivative of inputs) of a avg pooling operation
# The computed losses are the losses for the previous layer
function multichannel_avgpool_backward(pool_layer)
    # storing all the necessary shapes
    current_batch_size::Int, channels::Int, input_height::Int, input_width::Int = size(pool_layer.inputs)
    current_batch_size, channels, output_height::Int, output_width::Int = size(pool_layer.outputs)
    # kernel_width, kernel_height = pool_layer.kernel_size
    kernel_height::Int, kernel_width::Int = pool_layer.kernel_size

    out_losses::Array{Float64, 4} = pool_layer.losses
    if pool_layer.df != 1
        out_losses = out_losses .* pool_layer.df(pool_layer.outputs_no_activation)
    end

    # losses = Array{Float64}(undef, current_batch_size, channels, input_height, input_width)
    losses::Array{Float64, 4} = zeros(current_batch_size, channels, input_height, input_width)

    # going throw all data in batch
    # for index_batch in 1:current_batch_size
    @inbounds Threads.@threads for index_batch in 1:current_batch_size

        # going throw every output
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
        # Threads.@threads for channel in 1:channels
        # Threads.@threads for y_out in 1:output_height
        # Threads.@threads for x_out in 1:output_width
            m, n = get_input_position((y_out, x_out), pool_layer.stride)
            m::Int -= pool_layer.padding[1]
            n::Int -= pool_layer.padding[2]
            for y_w in 1:kernel_height, x_w in 1:kernel_width
                y_in = m + (y_w - 1) * pool_layer.dilation[1]
                x_in = n + (x_w - 1) * pool_layer.dilation[2]
                if 1 <= y_in <= input_height && 1 <= x_in <= input_width
                    losses[index_batch, channel, y_in, x_in] += out_losses[index_batch, channel, y_out, x_out]
                end
            end
        end
        # end
        # end

    end
    # factor = (1/(kernel_width * kernel_width))
    factor::Float64 = (1/(kernel_height * kernel_width))
    losses *= factor
    # losses = losses .* factor

    # pool_layer.previous_losses = losses
    return losses
end

#=
Adaptive Pooling-Operations: Forward & Backward
=#

# Performing a multichannel adaptive maximum pooling (on a hole batch)
# Shape of input: (batch_size, in_channels, height, width)
# output_size must be always given as a tuple of length 2
function multichannel_adaptive_maxpool(inputs::Array{Float64, 4}, output_size::Tuple{Int, Int})
    # storing all the necessary shapes
    current_batch_size::Int, channels::Int, input_height::Int, input_width::Int = size(inputs)
    output_height::Int, output_width::Int = output_size

    output::Array{Float64, 4} = Array{Float64, 4}(undef, current_batch_size, channels, output_height, output_width)

    # positions are necessary for backpropagation (only for max pooling)
    # each position in the output is assigned the position in the input with the highest value
    # (which is also the value in the output matrix at the given position in output)
    y_positions::Array{Int, 4} = Array{Int, 4}(undef, current_batch_size, channels, output_height, output_width)
    x_positions::Array{Int, 4} = Array{Int, 4}(undef, current_batch_size, channels, output_height, output_width)
    positions::Tuple{Array{Int, 4}, Array{Int, 4}} = (y_positions, x_positions)

    y_in_indices = get_in_indices(input_height, output_height)
    x_in_indices = get_in_indices(input_width, output_width)
    # going throw all data in batch
    for index_batch in 1:current_batch_size # Threads.@threads, @turbo
        # compute pooling for each channel seperatly
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width # @turbo
            values = inputs[index_batch, channel, y_in_indices[y_out], x_in_indices[x_out]]
            max_index = argmax(values)
            output[index_batch, channel, y_out, x_out] = values[max_index] # maximum(values)
            y_positions[index_batch, channel, y_out, x_out] = y_in_indices[y_out][1] + max_index[1] - 1
            x_positions[index_batch, channel, y_out, x_out] = x_in_indices[x_out][1] + max_index[2] - 1
        end
    end

    return output, positions
end

# Performing a multichannel adaptive average pooling (on a hole batch)
# Shape of input: (batch_size, in_channels, height, width)
# output_size must be always given as a tuple of length 2
function multichannel_adaptive_avgpool(inputs::Array{Float64, 4}, output_size::Tuple{Int, Int})
    # storing all the necessary shapes
    current_batch_size::Int, channels::Int, input_height::Int, input_width::Int = size(inputs)
    output_height::Int, output_width::Int = output_size

    output::Array{Float64, 4} = Array{Float64, 4}(undef, current_batch_size, channels, output_height, output_width)

    y_in_indices = get_in_indices(input_height, output_height)
    x_in_indices = get_in_indices(input_width, output_width)
    # going throw all data in batch
    #=
    for index_batch in 1:current_batch_size # Threads.@threads, @turbo
        # compute pooling for each channel seperatly
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width # @turbo
            kernel_sum = 0.00
            for y_in in y_in_indices[y_out], x_in in x_in_indices[x_out]
                kernel_sum += inputs[index_batch, channel, y_in, x_in] # ::Float64
            end
            output[index_batch, channel, y_out, x_out] = kernel_sum / (length(y_in_indices[y_out]) * length(x_in_indices[x_out]))
        end
    end
    =#
    for y_out in 1:output_height, x_out in 1:output_width
        current_y_in_indices = y_in_indices[y_out]
        current_x_in_indices = x_in_indices[x_out]
        @turbo for index_batch in 1:current_batch_size, channel in 1:channels
            kernel_sum = 0.00
            for y_in in current_y_in_indices, x_in in current_x_in_indices
                kernel_sum += inputs[index_batch, channel, y_in, x_in] # ::Float64
            end
            output[index_batch, channel, y_out, x_out] = kernel_sum / (length(current_y_in_indices) * length(current_x_in_indices))
        end
    end

    return output
end

# Functions used for Backpropagation (Adaptive Pooling)
# The only input each function takes is an instance of a pool layer struct (AdaptiveMaxPool or AdaptiveAvgPool)
# Because a layer is given, these functions directly work on the hole batch

# Computes the backward-pass (derivative of inputs) of a adaptive max pooling operation
# The computed losses are the losses for the previous layer
function multichannel_adaptive_maxpool_backward(pool_layer)
    # storing all the necessary shapes
    current_batch_size::Int, channels::Int, input_height::Int, input_width::Int = size(pool_layer.inputs)
    current_batch_size, channels, output_height::Int, output_width::Int = size(pool_layer.outputs)
    positions::Tuple{Array{Int, 4}, Array{Int, 4}} = pool_layer.positions

    out_losses::Array{Float64, 4} = pool_layer.losses
    if pool_layer.df != 1
        out_losses = out_losses .* pool_layer.df(pool_layer.outputs_no_activation)
    end

    # losses = Array{Float64}(undef, current_batch_size, channels, input_height, input_width)
    losses::Array{Float64, 4} = zeros(current_batch_size, channels, input_height, input_width)

    # going throw all data in batch
    @turbo for index_batch in 1:current_batch_size

        # going throw every output
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
            #=
            value::Float64 = out_losses[index_batch, channel, y_out, x_out]
            y_position::Int = positions[1][index_batch, channel, y_out, x_out]
            x_position::Int = positions[2][index_batch, channel, y_out, x_out]
            =#
            value = out_losses[index_batch, channel, y_out, x_out]
            y_position = positions[1][index_batch, channel, y_out, x_out]
            x_position = positions[2][index_batch, channel, y_out, x_out]
            losses[index_batch, channel, y_position, x_position] += value
        end
    
    end

    return losses
end

# Computes the backward-pass (derivative of inputs) of a adaptive avg pooling operation
# The computed losses are the losses for the previous layer
function multichannel_adaptive_avgpool_backward(pool_layer)
    #=
    # calculating and setting up kernel_size and stride
    input_height, input_width = size(inputs)[3:4]
    output_height, output_width = pool_layer.output_size

    y_stride = input_height ÷ output_height
    x_stride = input_width ÷ output_width

    kernel_height = input_height - (output_height - 1) * y_stride
    kernel_width = input_width - (output_width - 1) * x_stride

    stride = (y_stride, x_stride)
    kernel_size = (kernel_height, kernel_width)
    =#
    # storing all the necessary shapes
    current_batch_size::Int, channels::Int, input_height::Int, input_width::Int = size(pool_layer.inputs)
    current_batch_size, channels, output_height::Int, output_width::Int = size(pool_layer.outputs)

    out_losses::Array{Float64, 4} = pool_layer.losses
    if pool_layer.df != 1
        out_losses = out_losses .* pool_layer.df(pool_layer.outputs_no_activation)
    end

    # losses = Array{Float64}(undef, current_batch_size, channels, input_height, input_width)
    losses::Array{Float64, 4} = zeros(current_batch_size, channels, input_height, input_width)
    y_in_indices = get_in_indices(input_height, output_height)
    x_in_indices = get_in_indices(input_width, output_width)
    #=
    # going throw all data in batch
    for index_batch in 1:current_batch_size # Threads.@threads, @turbo
        # compute pooling for each channel seperatly
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width # @turbo
            for y_in in y_in_indices[y_out], x_in in x_in_indices[x_out]
                losses[index_batch, channel, y_in, x_in] += out_losses[index_batch, channel, y_out, x_out] / (length(y_in_indices[y_out]) * length(x_in_indices[x_out]))
            end
        end
    end
    =#
    for y_out in 1:output_height, x_out in 1:output_width
        current_y_in_indices = y_in_indices[y_out]
        current_x_in_indices = x_in_indices[x_out]
        @turbo for index_batch in 1:current_batch_size, channel in 1:channels
            for y_in in current_y_in_indices, x_in in current_x_in_indices
                losses[index_batch, channel, y_in, x_in] += out_losses[index_batch, channel, y_out, x_out] / (length(current_y_in_indices) * length(current_x_in_indices))
            end
        end
    end

    # pool_layer.previous_losses = losses
    return losses
end

#=
Fully connected: Forward & Backward
=#

# Forward propagation for a fully connected layer (Fc)
function fc_forward(inputs::Array{Float64, 2}, weights::Array{Float64, 2}, bias::Vector{Float64}, use_bias::Bool)
    # return rand(size(inputs)[1], size(weights)[1])
    current_batch_size = size(inputs)[1]
    out_features, in_features = size(weights)
    outputs = Array{Float64}(undef, current_batch_size, out_features)
    #=
    # for index_batch in 1:current_batch_size
    @inbounds @views Threads.@threads for index_batch in 1:current_batch_size
        # outputs[index_batch, :] = weights * inputs[index_batch, :]
        # println(size(outputs[index_batch, :]), typeof(outputs[index_batch, :]))
        # println(size(weights * inputs[index_batch, :]), typeof(weights * inputs[index_batch, :]))
        # println(size(outputs), typeof(outputs))
        # exit()
        if use_bias
            outputs[index_batch, :] = weights * inputs[index_batch, :] + bias # .+
        else
            outputs[index_batch, :] = weights * inputs[index_batch, :]
        end
    end
    =#
    # NOCH NICHT GEGEN GETESTET
    if !use_bias
        bias = zeros(out_features)
    end
    @turbo for index_batch in 1:current_batch_size
        for out_feature in 1:out_features
            output_value = 0.00
            for in_feature in 1:in_features
                output_value += inputs[index_batch, in_feature] * weights[out_feature, in_feature]
            end
            outputs[index_batch, out_feature] = output_value + bias[out_feature]
        end
    end

    return outputs
end

# Functions used for Backpropagation (Fc)
# The only input each function takes is an instance of a fc layer struct (Fc)
# Because a layer is given, these functions directly work on the hole batch

# Backpropagation of a fully connected layer, the resuls are the losses for the previous layer
function fc_losses(fc_layer)
    #= OLD VERSION WITH MATRIX MULTIPLICATION
    # return rand(size(fc_layer.inputs)[1], size(fc_layer.inputs)[2])
    current_batch_size = size(fc_layer.inputs)[1]
    out_features, in_features = size(fc_layer.weights)
    losses = Array{Float64}(undef, current_batch_size, in_features)
    # fc_layer.losses .*= fc_layer.activation_function(fc_layer.outputs_no_activation) # maybe . (dot) is not the right way of doing an element-wise multiplication)
    if fc_layer.df != 1
        out_losses = fc_layer.losses .* fc_layer.df(fc_layer.outputs_no_activation)
    else
        out_losses = fc_layer.losses
    end
    weights_transposed = fc_layer.weights'
    # println("Size transposed weights: ", size(weights_transposed))
    # println("Size original weights: ", size(fc_layer.weights))
    # for index_batch in 1:current_batch_size
    @inbounds @views Threads.@threads for index_batch in 1:current_batch_size
        # losses[index_batch, :] = weights_transposed * fc_losses[index_batch, :]
        losses[index_batch, :] = weights_transposed * out_losses[index_batch, :]
    end
    =#
    out_losses = fc_layer.losses
    weights = fc_layer.weights
    inputs = fc_layer.inputs
    if fc_layer.df != 1
        out_losses = out_losses .* fc_layer.df(fc_layer.outputs_no_activation)
    end
    current_batch_size = size(inputs)[1]
    out_features::Int, in_features::Int = size(weights)
    losses = zeros(current_batch_size, in_features)

    @turbo for index_batch in 1:current_batch_size # @tturbo
        for in_feature in 1:in_features
            losses_value = 0.00
            for out_feature in 1:out_features
                losses_value += weights[out_feature, in_feature] * out_losses[index_batch, out_feature]
            end
            losses[index_batch, in_feature] = losses_value
        end
    end

    return losses
end

# Computes the derivative of the weights/bias on the given layer, the results are used to optimize the weights/bias
function fc_gradients(fc_layer)
    #= OLD VERSION WITH MATRIX MULTIPLICATION
    # return rand(size(fc_layer.weights)...)
    current_batch_size = size(fc_layer.inputs)[1]
    out_features, in_features = size(fc_layer.weights)
    # gradients = Array{Float64}(undef, out_features, in_features)
    gradients = zeros(out_features, in_features)
    ## gradients = fc_layer.gradients
    bias_gradients = zeros(out_features)
    # bias_gradients = fc_layer.bias_gradients
    if fc_layer.df != 1
        # df = fc_layer.losses .* fc_layer.df(fc_layer.outputs_no_activation)
        df = fc_layer.df(fc_layer.outputs_no_activation)
    else
        df = ones(current_batch_size, out_features)
    end
    out_losses = fc_layer.losses
    inputs = fc_layer.inputs
    # for index_batch in 1:current_batch_size
    @inbounds @views Threads.@threads for index_batch in 1:current_batch_size
        single_out_losses_vector = out_losses[index_batch, :]
        single_out_losses = reshape(single_out_losses_vector, out_features, 1)
        single_inputs = reshape(inputs[index_batch, :], in_features, 1)
        gradients += (single_out_losses .* df[index_batch, :]) * inputs[index_batch, :]'
        bias_gradients += single_out_losses_vector .* df[index_batch, :]
    end
    =#
    # NOCH NICHT GESTESTET
    out_losses = fc_layer.losses
    weights = fc_layer.weights
    inputs = fc_layer.inputs
    if fc_layer.df != 1
        out_losses = out_losses .* fc_layer.df(fc_layer.outputs_no_activation)
    end
    current_batch_size = size(inputs)[1]
    out_features::Int, in_features::Int = size(weights)
    gradients = zeros(out_features, in_features)

    @turbo for index_batch in 1:current_batch_size # @tturbo
        for in_feature in 1:in_features
            for out_feature in 1:out_features
                gradients[out_feature, in_feature] += inputs[index_batch, in_feature] * out_losses[index_batch, out_feature]
            end
        end
    end
    bias_gradients = sum(out_losses, dims=1)[1, :]

    return gradients, bias_gradients
end

function fc_backward(fc_layer)
    out_losses = fc_layer.losses
    inputs = fc_layer.inputs
    weights = fc_layer.weights
    if fc_layer.df != 1
        out_losses = out_losses .* fc_layer.df(fc_layer.outputs_no_activation)
    end
    # weights_transposed = weights'
    current_batch_size = size(inputs)[1]
    out_features::Int, in_features::Int = size(weights)
    gradients = zeros(out_features, in_features)
    # bias_gradients = zeros(out_features)
    losses = zeros(current_batch_size, in_features)

    @turbo for index_batch in 1:current_batch_size # @tturbo
        for in_feature in 1:in_features
            losses_value = 0.00
            for out_feature in 1:out_features
                losses_value += weights[out_feature, in_feature] * out_losses[index_batch, out_feature]
                gradients[out_feature, in_feature] += inputs[index_batch, in_feature] * out_losses[index_batch, out_feature]
                # gradients[out_feature, in_feature] += inputs[index_batch, out_feature] * out_losses[index_batch, out_feature]
            end
            losses[index_batch, in_feature] = losses_value
        end
    end
    bias_gradients = sum(out_losses, dims=1)[1, :]
    # println(size(bias_gradients))

    return gradients, bias_gradients, losses
end

#=
Reshape: Forward & Backward
=#

function reshape_forward(inputs::Array, out_shape::Tuple)
    current_batch_size = size(inputs)[1]
    output_shape = tuplejoin((current_batch_size, ), out_shape)
    # outputs = Array{Float64}(undef, output_shape...)
    outputs = reshape(inputs, output_shape)

    return outputs
end

# Functions used for Backpropagation (Reshape)
# The only input each function takes is an instance of a reshape layer struct (Reshape)
# Because a layer is given, these functions directly work on the hole batch

function reshape_backward(reshape_layer)
    if reshape_layer.df != 1
        out_losses = reshape_layer.losses .* reshape_layer.df(reshape_layer.outputs_no_activation)
    else
        out_losses = reshape_layer.losses
    end
    in_shape = size(reshape_layer.inputs)
    losses = reshape(out_losses, in_shape)

    return losses
end

#=
Batch normalization for 4d-inputs (2d batch normalization): Forward & Backward
=#

# calculates the variance of a 3-dimensional array, the mean must be given (necessary for batch normalization)
# function batchNorm2d_calculate_variance(channels_over_batch::Array{Float64, 3}, mean::Float64)
function batchNorm2d_calculate_variance(channels_over_batch::AbstractArray{Float64, 3}, mean::Float64)
    variance = 0.00
    for value in channels_over_batch
        variance += (value - mean)^2
    end
    variance /= length(channels_over_batch) # (1 / length(channels_over_batch)) * 

    return variance
end

# Performing a multichannel batch normalization (on a hole batch)
# Shape of input: (batch_size, channels, height, width)
# Shape of weight_gamma/weight_beta/running_mean/running_variance: (channels, )
function batchNorm2d_forward(inputs::Array{Float64, 4}, weight_gamma::Vector{Float64}, weight_beta::Vector{Float64}, track_running_stats::Bool, momentum::Float64, running_mean::Vector{Float64}, running_variance::Vector{Float64}, test_mode::Bool; epsilon::Float64=1e-5) # ::Array{Float64, 1}
    current_batch_size::Int, channels::Int, height::Int, width::Int = size(inputs)
    outputs = Array{Float64, 4}(undef, current_batch_size, channels, height, width)
    outputs_no_weights_applied = Array{Float64, 4}(undef, current_batch_size, channels, height, width)
    
    if !test_mode # executed only in training mode
        current_mean = Vector{Float64}(undef, channels)
        current_variance = Vector{Float64}(undef, channels)
        # for channel in 1:channels
        @inbounds @views Threads.@threads for channel in 1:channels
            channels_over_batch = inputs[:, channel, :, :]
            mean = sum(channels_over_batch) / length(channels_over_batch)
            variance = batchNorm2d_calculate_variance(channels_over_batch, mean)
            output_channels_over_batch = (channels_over_batch .- mean) / sqrt(variance + epsilon) # - epsilon
            outputs_no_weights_applied[:, channel, :, :] = output_channels_over_batch
            output_channels_over_batch = weight_gamma[channel] * output_channels_over_batch .+ weight_beta[channel]
            outputs[:, channel, :, :] = output_channels_over_batch
            current_mean[channel] = mean
            current_variance[channel] = variance
        end
        if track_running_stats
            running_mean = (1 - momentum) * running_mean .+ momentum * current_mean
            running_variance = (1 - momentum) * running_variance .+ momentum * current_variance
        end
    else # executed only in test mode
        for channel in 1:channels
            channels_over_batch = inputs[:, channel, :, :]
            if !track_running_stats # uses batch statistics when no running estimates were tracked during training
                mean = sum(channels_over_batch) / length(channels_over_batch)
                variance = batchNorm2d_calculate_variance(channels_over_batch, mean)
            else # using the during training computed running estimates
                mean = running_mean[channel]
                variance = running_variance[channel]
            end
            output_channels_over_batch = (channels_over_batch .- mean) / sqrt(variance + epsilon) # - epsilon
            output_channels_over_batch = weight_gamma[channel] * output_channels_over_batch .+ weight_beta[channel]
            outputs[:, channel, :, :] = output_channels_over_batch
        end
    end

    return outputs, running_mean, running_variance, outputs_no_weights_applied
end

# Functions used for Backpropagation (BatchNorm2d)
# The only input each function takes is an instance of a batch normalization layer struct (BatchNorm2d)
# Because a layer is given, these functions directly work on the hole batch

# Computes the derivative of the inputs on the given layer, the results are used as the losses for the previous layer
function batchNorm2d_losses(batchnorm_layer)
    current_batch_size, channels, height, width = size(batchnorm_layer.outputs)
    losses = Array{Float64, 4}(undef, current_batch_size, channels, height, width)

    if batchnorm_layer.df != 1
        out_losses = batchnorm_layer.losses .* batchnorm_layer.df(batchnorm_layer.outputs_no_activation)
    else
        out_losses = batchnorm_layer.losses
    end

    # for channel in 1:channels
    @inbounds @views Threads.@threads for channel in 1:channels
        channels_over_batch = batchnorm_layer.inputs[:, channel, :, :]
        num_values = length(channels_over_batch)
        mean = sum(channels_over_batch) / length(channels_over_batch)
        variance = batchNorm2d_calculate_variance(channels_over_batch, mean)
        denominator = sqrt(variance + batchnorm_layer.epsilon)
        xhat_batch = batchnorm_layer.outputs_no_weights_applied[:, channel, :, :] # xhat are the outputs without weights applied
        dxhat_batch = out_losses[:, channel, :, :] * batchnorm_layer.weight_gamma[channel] # derivative of out losses with respect to outputs without weights applied
        losses[:, channel, :, :] = ((num_values * dxhat_batch) .- sum(dxhat_batch) .- (xhat_batch * sum(dxhat_batch .* xhat_batch))) / (num_values * denominator)
    end

    return losses
end

# Computes the derivative of the weights on the given layer, the results are used to optimize the weights
function batchNorm2d_gradients(batchnorm_layer)
    gradient_gamma = zeros(size(batchnorm_layer.weight_gamma))
    gradient_beta = zeros(size(batchnorm_layer.weight_beta)) # weight_gamma
    if !batchnorm_layer.affine
        return gradient_gamma, gradient_beta
    else
        channels = length(gradient_gamma)

        if batchnorm_layer.df != 1
            out_losses = batchnorm_layer.losses .* batchnorm_layer.df(batchnorm_layer.outputs_no_activation)
        else
            out_losses = batchnorm_layer.losses
        end
        
        # for channel in 1:channels
        @inbounds @views Threads.@threads for channel in 1:channels
            gradient_beta[channel] = sum(out_losses[:, channel, :, :])
            # channel_outputs_no_weights = (channel_outputs .- batchnorm_layer.weight_beta[channel]) / batchnorm_layer.weight_gamma[channel]
            channel_outputs_no_weights = batchnorm_layer.outputs_no_weights_applied[:, channel, :, :]
            gradient_gamma[channel] = sum(out_losses[:, channel, :, :] .* channel_outputs_no_weights)
        end
    end
    
    return gradient_gamma, gradient_beta
end

#=
Activation functions: Forward & Backward (their derivatives)
All functions take an array (nd, number of dimensions doesn't matter) and return a new array with the modified values
(The given array will not be modified)
The prefix "d_" stands for the derivative (of the activation function)
=#

# appplies a element-wise relu activation on a copy of inputs
function relu(inputs::Array)
    outputs = copy(inputs)
    # for (index, value) in enumerate(outputs)
    for index in eachindex(outputs)
        value = outputs[index]
        if value < 0
            outputs[index] = 0
        end
    end
    # outputs = max.(0, inputs)

    return outputs
end

# appplies the element-wise derivative of relu activation on a copy of inputs
function d_relu(inputs::Array)
    outputs = copy(inputs)
    # for (index, value) in enumerate(outputs)
    for index in eachindex(outputs)
        value = outputs[index]
        if value < 0
            outputs[index] = 0
        else
            outputs[index] = 1
        end
    end

    return outputs
end

# appplies a element-wise sigmoid activation on a copy of inputs
function sigmoid(inputs::Array)
    outputs = copy(inputs)
    # for (index, value) in enumerate(outputs)
    # sig(x) = 1 / (1 + exp(-x))
    # outputs = map(sig, outputs)
    for index in eachindex(outputs)
        value = outputs[index]
        outputs[index] = 1 / (1 + exp(-value))
    end

    return outputs
end

# appplies the element-wise derivative of sigmoid activation on a copy of inputs
function d_sigmoid(inputs::Array)
    outputs = copy(inputs)
    sig(x) = 1 / (1 + exp(-x))
    # for (index, value) in enumerate(outputs)
    for index in eachindex(outputs)
        value = outputs[index]
        outputs[index] = sig(value) * (1 - sig(value))
    end

    return outputs
end

# appplies the element-wise derivative of tanh activation on a copy of inputs
function gv_tanh(inputs::Array)
    #=
    outputs = copy(inputs)
    for index in eachindex(outputs)
        value = outputs[index]
        outputs[index] = tanh(value)
    end
    =#
    outputs = tanh.(inputs)

    return outputs
end

# DEPRECTED: there is no further tanh function because Julia already has a built-in tanh function
# appplies the element-wise derivative of tanh activation on a copy of inputs
function d_tanh(inputs::Array)
    outputs = copy(inputs)
    # for (index, value) in enumerate(outputs)
    for index in eachindex(outputs)
        value = outputs[index]
        outputs[index] = 1 - tanh(value)^2
    end

    return outputs
end

#=
Softmax along a specific dimension (dim): Forward & Backward
=#

# computes the softmax along a specific dimension
function softmax(input; dim=1)
    #=
    output = copy(input)
    exps_sum = dim_sum(exp.(input); dim=dim)
    dim_size = size(input)[dim]
    indices_array = Union{UnitRange{Int}, Int}[1:dim_size for dim_size in size(input)]
    for index_dim in 1:dim_size
        indices_array[dim] = index_dim
        output[indices_array...] = exp.(input[indices_array...]) ./ exps_sum
    end
    # output = exp.(input) ./ exps_sum # only works for matrices (exp.(input)) followed by a vector (exps_sum)
    =#

    # truly iterates over all slices (vectors)

    output = zeros(size(input))
    input_size = size(input)
    num_dims = length(input_size)
    # checks if dim is a valid value
    if dim == 0 || abs(dim) > num_dims
        error("GradValley: dim_sum: the given dim is out of bounce")
    end
    if dim < 0
        dim = num_dims + 1 - dim
    end
    iterators = get_iters_without_at_dim(input_size, dim)
    for indices_tuple in Base.product(iterators...)
        indices_array = collect(Union{UnitRange{Int}, Int}, indices_tuple)
        # insert!(indices_array, dim, :)
        indices_array = insert!(indices_array, dim, 1:input_size[dim])
        input_array_slice = input[indices_array...] # input_array_slice is always a vector
        exps_sum = sum(exp.(input_array_slice))
        output[indices_array...] = exp.(input_array_slice) ./ exps_sum
    end


    return output
end

# Functions used for Backpropagation (Softmax)
# The only input each function takes is an instance of a softmax layer struct (Softmax)
# Because a layer is given, these functions directly work on the hole batch

# computes the derivative of softmax activation on the given layer
function softmax_backward(softmax_layer)
    out_losses = softmax_layer.losses
    softmax_output = softmax_layer.outputs
    dim = softmax_layer.dim
    out_losses_size = size(out_losses)
    dim_size = out_losses_size[dim]
    losses = zeros(eltype(out_losses), out_losses_size)
    iterators = get_iters_without_at_dim(out_losses_size, dim)
    for indices_tuple in Base.product(iterators...)
        indices_array = collect(Union{UnitRange{Int}, Int}, indices_tuple)
        insert!(indices_array, dim, 1:dim_size)

        softmax_output_array_slice = softmax_output[indices_array...] # is always a vector
        n = length(softmax_output_array_slice)
        replicated_softmax_output = zeros(n, n)
        for x in 1:n
            replicated_softmax_output[:, x] = softmax_output_array_slice
        end
        identity = Matrix(1.0I, n, n) # Identity matrix of Float64 type
        jacobian_matrix = (replicated_softmax_output .* (identity - transpose(replicated_softmax_output)))

        losses[indices_array...] = jacobian_matrix * out_losses[indices_array...]
    end

    return losses
end

#=
Weight initialization:
All functions take a tuple which is the shape of the returned weight, gain is also a necessary paramter
(choosen by the activation function, usually given by a layer struct)
=#

# calculates fan_mode for all types of weight initializations
function calculate_fan_mode(weight_shape::Tuple)
    if length(weight_shape) == 4 # Convolution layer (Conv)
        in_channels = weight_shape[2]
        out_channels = weight_shape[1]
        size_kernel = weight_shape[3] * weight_shape[4]
        fan_in = in_channels * size_kernel
        fan_out = out_channels * size_kernel
    elseif length(weight_shape) == 2 # Fully connected layer (Fc)
        fan_in = weight_shape[2]
        fan_out = weight_shape[1]
    else
        error("GradValley: calculate_fan_mode: invalid weight_shape")
    end

    return fan_in, fan_out
end

# default initialization (normal distribution)
function default_init(weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        std = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        std = gain / sqrt(fan_out)
    else
        error("GradValley: default_init: invalid fan_mode")
    end
    weight = randn(weight_shape) * std

    return weight
end

# default initialization (uniform distribution)
function default_uniform_init(weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        bound = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        bound = gain / sqrt(fan_out)
    else
        error("GradValley: default_uniform_init: invalid fan_mode")
    end
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    weight = rand(weight_shape...) * 2 * bound .- bound

    return weight
end

# kaiming (he) initialization (normal distribution)
function kaiming_init(weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        std = gain * sqrt(1 / fan_in)
    elseif fan_mode == "fan_out"
        std = gain * sqrt(1 / fan_out)
    else
        error("GradValley: kaiming_init: invalid fan_mode")
    end
    weight = randn(weight_shape) * std

    return weight
end

# xavier (glorot) initialization (normal distribution)
function xavier_init(weight_shape::Tuple, gain::Real)
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    std = gain * sqrt(2 / (fan_in + fan_out))
    weight = randn(weight_shape) * std

    return weight
end

# kaiming (he) initialization (uniform distribution)
function kaiming_uniform_init(weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        bound = gain * sqrt(3 / fan_in)
    elseif fan_mode == "fan_out"
        bound = gain * sqrt(3 / fan_out)
    else
        error("GradValley: kaiming_init: invalid fan_mode")
    end
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    weight = rand(weight_shape...) * 2 * bound .- bound

    return weight
end

# xavier (glorot) initialization (uniform distribution)
function xavier_uniform_init(weight_shape::Tuple, gain::Real)
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    bound = gain * sqrt(6 / (fan_in + fan_out))
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    weight = rand(weight_shape...) * 2 * bound .- bound

    return weight
end

# bias initialization (normal distribution)
function bias_init(bias_shape::Tuple, weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # bias = rand(bias_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        std = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        std = gain / sqrt(fan_out)
    else
        error("GradValley: bias_normal_init: invalid fan_mode")
    end
    bias = randn(bias_shape) * std

    return bias
end

# bias initialization (uniform distribution)
function bias_uniform_init(bias_shape::Tuple, weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # bias = rand(bias_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        bound = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        bound = gain / sqrt(fan_out)
    else
        error("GradValley: bias_uniform_init: invalid fan_mode")
    end
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    bias = rand(bias_shape...) * 2 * bound .- bound

    return bias
end

end # end of module "gv_functional"