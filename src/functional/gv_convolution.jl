#=
Convolution-Operations: Forward & Backward
=#

# Performing a single channel convolution
# Shape of input: (height, width)
# Shape of kernel: (height, width)
# stride, padding and dilation must be always given as tuples of length 2
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
# Shape of input: (batch_size, in_channels / groups, height, width)
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

# Performing a multichannel transpose convolution (on a hole batch)
# Shape of input: (batch_size, in_channels, height, width)
# Shape of kernels: (in_channels, out_channels / groups, height, width)
# stride, padding and dilation must be always given as tuples of length 2
function multichannel_conv_transpose(inputs::Array{Float64, 4}, kernels::Array{Float64, 4}, bias::Vector{Float64}, use_bias::Bool; stride::Tuple{Int, Int}=(1, 1), padding::Tuple{Int, Int}=(0, 0), output_padding::Tuple{Int, Int}=(0, 0), dilation::Tuple{Int, Int}=(1, 1), groups::Int=1)
    # storing all the necessary shapes
    current_batch_size, in_channels, input_height, input_width = size(inputs)
    in_channels, out_channels_kernels, kernel_height, kernel_width = size(kernels)

    # splitting up the hyperparameters per dimension
    y_stride, x_stride = stride
    y_padding, x_padding = padding
    y_out_padding, x_out_padding = output_padding
    y_dilation, x_dilation = dilation

    if !(y_out_padding < y_stride || y_out_padding < y_dilation) || !(x_out_padding < x_stride || x_out_padding < x_dilation)
        error("output_padding must be smaller than either stride or dilation, but got invalid values: y_output_padding: $y_out_padding x_ou_padding: $x_out_padding y_stride: $y_stride x_stride: $x_stride y_dilation: $y_dilation x_dilation: $x_dilation")
    end

    output_height = (input_height - 1) * y_stride + y_dilation * (kernel_height - 1) + y_out_padding + 1
    output_width = (input_width - 1) * x_stride + x_dilation * (kernel_width - 1) + x_out_padding + 1

    outputs = zeros(current_batch_size, out_channels_kernels * groups, output_height, output_width)

    in_channels_per_group = in_channels ÷ groups
    # actual computation
    @inbounds Threads.@threads for index_batch in 1:current_batch_size

        @turbo for group in 1:groups, in_channel_per_group in 1:in_channels_per_group, y_in in 1:input_height, x_in in 1:input_width
            m = y_in + (y_stride - 1) * (y_in - 1)
            n = x_in + (x_stride - 1) * (x_in - 1)
            in_channel = (group * in_channels_per_group + 1) - in_channel_per_group
            for out_channel_kernel in 1:out_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                y_out = m + (y_w - 1) * y_dilation
                x_out = n + (x_w - 1) * x_dilation
                out_channel_output = out_channel_kernel + (group - 1) * out_channels_kernels
                outputs[index_batch, out_channel_output, y_out, x_out] += kernels[in_channel, out_channel_kernel, y_w, x_w] * inputs[index_batch, in_channel, y_in, x_in]
            end
        end

        # adding bias if necessary
        if use_bias
            @turbo for out_channel in 1:out_channels_kernels * groups
                bias_value = bias[out_channel]
                for y_out in 1:output_height, x_out in 1:output_width
                    outputs[index_batch, out_channel, y_out, x_out] += bias_value
                end
            end
        end

    end

    if padding != (0, 0)
        # @views outputs = outputs[:, :, y_padding+1:output_height-y_padding, x_padding+1:output_width-x_padding] # scheint auch ohne @views gut und performant zu funktionieren
        outputs = outputs[:, :, y_padding+1:output_height-y_padding, x_padding+1:output_width-x_padding]
    end
   
    return outputs
end

# Computes the derivative of the inputs on the given layer, the results are used as the losses for the previous layer
function multichannel_conv_transpose_losses(conv_layer)
    # storing all the necessary shapes
    current_batch_size, in_channels, input_height, input_width = size(conv_layer.inputs)
    in_channels, out_channels_kernels, kernel_height, kernel_width = size(conv_layer.kernels)

    # calculating the derivative of the out_losses
    out_losses = conv_layer.losses
    if conv_layer.df != 1
        out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
    end

    # splitting up the hyperparameters per dimension
    y_stride, x_stride = conv_layer.stride
    y_dilation, x_dilation = conv_layer.dilation

    # performing padding
    if conv_layer.padding != (0, 0)
        out_losses = zero_pad_2d(out_losses, conv_layer.padding)
    end

    losses = zeros(current_batch_size, in_channels, input_height, input_width)

    in_channels_per_group = in_channels ÷ conv_layer.groups
    @inbounds Threads.@threads for index_batch in 1:current_batch_size

        @turbo for group in 1:conv_layer.groups, in_channel_per_group in 1:in_channels_per_group, y_in in 1:input_height, x_in in 1:input_width
            m = y_in + (y_stride - 1) * (y_in - 1)
            n = x_in + (x_stride - 1) * (x_in - 1)
            in_channel = (group * in_channels_per_group + 1) - in_channel_per_group
            value = 0.00
            for out_channel_kernel in 1:out_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                y_out = m + (y_w - 1) * y_dilation
                x_out = n + (x_w - 1) * x_dilation
                out_channel_output = out_channel_kernel + (group - 1) * out_channels_kernels
                value += conv_layer.kernels[in_channel, out_channel_kernel, y_w, x_w] * out_losses[index_batch, out_channel_output, y_out, x_out]
            end
            losses[index_batch, in_channel, y_in, x_in] = value
        end

    end

    return losses
end

# Computes the derivative of the kernels/weights on the given layer, the results are used to optimize the kernels/weights
function multichannel_conv_transpose_gradients(conv_layer)
    # storing all the necessary shapes
    current_batch_size, in_channels, input_height, input_width = size(conv_layer.inputs)
    in_channels, out_channels_kernels, kernel_height, kernel_width = size(conv_layer.kernels)

    # calculating the derivative of the out_losses
    out_losses = conv_layer.losses
    if conv_layer.df != 1
        out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
    end

    # splitting up the hyperparameters per dimension
    y_stride, x_stride = conv_layer.stride
    y_dilation, x_dilation = conv_layer.dilation

    # performing padding
    if conv_layer.padding != (0, 0)
        out_losses = zero_pad_2d(out_losses, conv_layer.padding)
    end

    gradients = zeros(in_channels, out_channels_kernels, kernel_height, kernel_width)

    in_channels_per_group = in_channels ÷ conv_layer.groups
    #= ANOTHER VERSION
    @turbo for index_batch in 1:current_batch_size # turbo

        for group in 1:conv_layer.groups, in_channel_per_group in 1:in_channels_per_group, y_in in 1:input_height, x_in in 1:input_width
            m = y_in + (y_stride - 1) * (y_in - 1)
            n = x_in + (x_stride - 1) * (x_in - 1)
            in_channel = (group * in_channels_per_group + 1) - in_channel_per_group
            for out_channel_kernel in 1:out_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                y_out = m + (y_w - 1) * y_dilation
                x_out = n + (x_w - 1) * x_dilation
                out_channel_output = out_channel_kernel + (group - 1) * out_channels_kernels
                gradients[in_channel, out_channel_kernel, y_w, x_w] += conv_layer.inputs[index_batch, in_channel, y_in, x_in] * out_losses[index_batch, out_channel_output, y_out, x_out]
            end
        end

    end
    =#
    @inbounds Threads.@threads for in_channel_per_group in 1:in_channels_per_group

        @turbo for group in 1:conv_layer.groups, out_channel_kernel in 1:out_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
            in_channel = (group * in_channels_per_group + 1) - in_channel_per_group
            out_channel_output = out_channel_kernel + (group - 1) * out_channels_kernels
            value = 0.00
            for y_in in 1:input_height, x_in in 1:input_width
                m = y_in + (y_stride - 1) * (y_in - 1)
                n = x_in + (x_stride - 1) * (x_in - 1)
                y_out = m + (y_w - 1) * y_dilation
                x_out = n + (x_w - 1) * x_dilation
                for index_batch in 1:current_batch_size
                    value += conv_layer.inputs[index_batch, in_channel, y_in, x_in] * out_losses[index_batch, out_channel_output, y_out, x_out]
                end
            end
            gradients[in_channel, out_channel_kernel, y_w, x_w] = value
        end

    end
    
    return gradients
end


# Computes the derivative of the bias on the given layer, the results are used to optimize the bias
function multichannel_conv_transpose_bias_gradients(conv_layer)
    # storing all the necessary shapes
    current_batch_size, in_channels, input_height, input_width = size(conv_layer.inputs)
    in_channels, out_channels_kernels, kernel_height, kernel_width = size(conv_layer.kernels)

    # calculating the derivative of the out_losses
    out_losses = conv_layer.losses
    if conv_layer.df != 1
        out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
    end

    # performing padding
    if conv_layer.padding != (0, 0)
        out_losses = zero_pad_2d(out_losses, conv_layer.padding)
    end
    # getting output size before the "padding" applied in forward pass (actually the opposite of normal padding) 
    current_batch_size, out_channels, output_height, output_width = size(out_losses)

    bias_gradients = conv_layer.bias_gradients

    @turbo for out_channel in 1:out_channels # @inbounds Threads.@threads
        value = 0.00
        for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
            value += out_losses[index_batch, out_channel, y_out, x_out]
        end
        bias_gradients[out_channel] += value
    end

    return bias_gradients
end