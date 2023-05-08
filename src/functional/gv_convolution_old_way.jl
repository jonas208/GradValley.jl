#=
Convolution-Operations: Forward & Backward
=#

@doc raw"""
    convolution2d!(outputs::AbstractArray{T, 4}, inputs::AbstractArray{T, 4}, kernels::AbstractArray{T, 4}, bias::AbstractVector{T}, use_bias::Bool; stride::NTuple{2, T2}=(1, 1), padding::NTuple{2, T2}=(0, 0), dilation::NTuple{2, T2}=(1, 1), groups::T2=1) where {T <: Real, T2 <: Integer}

Non-allocating version of the forward function for the associated 2d-Convolution layer, see Conv for details.
"""
function convolution2d!(outputs::AbstractArray{T, 4}, inputs::AbstractArray{T, 4}, kernels::AbstractArray{T, 4}, bias::AbstractVector{T}, use_bias::Bool; stride::NTuple{2, T2}=(1, 1), padding::NTuple{2, T2}=(0, 0), dilation::NTuple{2, T2}=(1, 1), groups::T2=1) where {T <: Real, T2 <: Integer}
    # storing all the necessary shapes
    current_batch_size, in_channels, input_height, input_width = size(inputs)
    out_channels, in_channels_kernels, kernel_height, kernel_width = size(kernels)
    output_height, output_width = size(outputs)[3:4]

    # performing padding
    if padding != (0, 0)
        inputs = zero_pad_2d(inputs, padding)
    end

    # check if to use bias 
    if !use_bias
        bias = zeros(size(bias))
    end
    
    y_stride = stride[1]
    x_stride = stride[2]
    y_dilation = dilation[1]
    x_dilation = dilation[2]
    out_channels_per_group = out_channels ÷ groups
    # actual computation
    if groups == 1 && stride == (1, 1) && dilation == (1, 1) # very specialized case for maximum performance
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                value = 0.00
                for in_channel in 1:in_channels, y_w in 1:kernel_height, x_w in 1:kernel_width
                    value += inputs[index_batch, in_channel, y_out + y_w - 1, x_out + x_w - 1] * kernels[out_channel, in_channel, y_w, x_w]
                end
                outputs[index_batch, out_channel, y_out, x_out] = value + bias[out_channel]
            end
        end
    elseif groups == 1 # second specialized case for better performance
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                value = 0.00
                for in_channel in 1:in_channels, y_w in 1:kernel_height, x_w in 1:kernel_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    value += inputs[index_batch, in_channel, y_in, x_in] * kernels[out_channel, in_channel, y_w, x_w]
                end
                outputs[index_batch, out_channel, y_out, x_out] = value + bias[out_channel]
            end
        end
    else # general case for any convolution 
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                value = 0.00
                for in_channel_kernel in 1:in_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    in_channel_input = in_channel_kernel + (group - 1) * in_channels_kernels
                    value += inputs[index_batch, in_channel_input, y_in, x_in] * kernels[out_channel, in_channel_kernel, y_w, x_w]
                end
                outputs[index_batch, out_channel, y_out, x_out] = value + bias[out_channel]
            end
        end
    end

    return outputs, inputs
end

@doc raw"""
    convolution2d(inputs::AbstractArray{T, 4}, kernels::AbstractArray{T, 4}, bias::AbstractVector{T}, use_bias::Bool; stride::NTuple{2, T2}=(1, 1), padding::NTuple{2, T2}=(0, 0), dilation::NTuple{2, T2}=(1, 1), groups::T2=1) where {T <: Real, T2 <: Integer}

Allocating version of the forward function for the associated 2d-Convolution layer, see Conv for details.
"""
function convolution2d(inputs::AbstractArray{T, 4}, kernels::AbstractArray{T, 4}, bias::AbstractVector{T}, use_bias::Bool; stride::NTuple{2, T2}=(1, 1), padding::NTuple{2, T2}=(0, 0), dilation::NTuple{2, T2}=(1, 1), groups::T2=1) where {T <: Real, T2 <: Integer}
    # storing all the necessary shapes
    current_batch_size::Int, in_channels::Int, input_height::Int, input_width::Int = size(inputs)
    out_channels::Int, in_channels_kernels::Int, kernel_height::Int, kernel_width::Int = size(kernels)
    # calculating shape of output
    output_height::Int, output_width::Int = calculate_output_shape(input_height, input_width, kernel_height, kernel_width, stride=stride, padding=padding, dilation=dilation)

    outputs = zeros(current_batch_size, out_channels, output_height, output_width)

    return convolution2d!(outputs, inputs, kernels, bias, use_bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

@doc raw"""
    convolution2d!(conv_layer, inputs::AbstractArray{T, 4}; no_grad::Bool=false) where {T <: Real}

The forward function for the associated 2d-Convolution layer that the layer's forward function directly calls, see Conv for details..
"""
function convolution2d!(conv_layer, inputs::AbstractArray{T, 4}; no_grad::Bool=false) where {T <: Real}
    # storing all the necessary shapes
    current_batch_size, in_channels, input_height, input_width = size(inputs)
    out_channels, in_channels_kernels, kernel_height, kernel_width = size(conv_layer.kernels)
    # calculating shape of output
    output_height, output_width = calculate_output_shape(input_height, input_width, kernel_height, kernel_width, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation)

    # check if the sizes match the pre-allocated arrays to avoid unnecessary allocations
    if !isnothing(conv_layer.outputs_no_activation) && size(conv_layer.outputs_no_activation) == (current_batch_size, out_channels, output_height, output_width)
        outputs_no_activation, inputs_padded = convolution2d!(conv_layer.outputs_no_activation, inputs, conv_layer.kernels, conv_layer.bias, conv_layer.use_bias, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
    else
        outputs_no_activation, inputs_padded = convolution2d(inputs, conv_layer.kernels, conv_layer.bias, conv_layer.use_bias, stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups)
    end

    # check if it is possible to use pre-allocated arrays 
    if !(isnothing(conv_layer.activation_function))
        if !isnothing(conv_layer.outputs) && size(conv_layer.outputs) == (current_batch_size, out_channels, output_height, output_width)
            outputs = conv_layer.activation_function(conv_layer.outputs, outputs_no_activation)
        else
            outputs = conv_layer.activation_function(outputs_no_activation)
        end
    else
        outputs = outputs_no_activation
    end

    if !no_grad
        # saving the results of forward computation in the layer struct (mutable)
        conv_layer.outputs_no_activation = outputs_no_activation
        conv_layer.outputs = outputs
        conv_layer.inputs_padded = inputs_padded
        conv_layer.inputs = inputs
    end

    return outputs
end

# Functions used for Backpropagation (Convolution)
# The only input each function takes is an instance of a conv layer struct (Conv or DepthwiseConv or ConvTranspose)
# Because a layer is given, these functions directly work on the hole batch

# Computes the derivative of the inputs on the given layer, the results are used as the losses for the previous layer
function multichannel_conv_losses(conv_layer) # maybe works not yet with padding -> TO TEST!
    # storing all the necessary shapes
    current_batch_size::Int, in_channels::Int, input_height::Int, input_width::Int = size(conv_layer.inputs_padded)
    current_batch_size, out_channels::Int, output_height::Int, output_width::Int = size(conv_layer.outputs_no_activation) # size(conv_layer.outputs)
    out_channels, in_channels_kernels::Int, kernel_height::Int, kernel_width::Int = size(conv_layer.kernels)

    # losses for the previous layer
    losses::Array{Float64, 4} = zeros(current_batch_size, in_channels, input_height, input_width)

    out_losses::Array{Float64, 4} = conv_layer.losses
    if conv_layer.df != 1
        activation_function_gradients = conv_layer.df(conv_layer.outputs, conv_layer.outputs_no_activation) # outputs becomes activation_function_gradients (for allocating as little as possible)
        out_losses = activation_function_gradients .= (*).(out_losses, activation_function_gradients)
        # out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
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
    if groups == 1 && stride == (1, 1) && dilation == (1, 1) # very specialized case for maximum performance
        # println("Correct Impl")
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                for in_channel in 1:in_channels, y_w in 1:kernel_height, x_w in 1:kernel_width
                    losses[index_batch, in_channel, y_out + y_w - 1, x_out + x_w - 1] += conv_layer.kernels[out_channel, in_channel, y_w, x_w] * out_losses[index_batch, out_channel, y_out, x_out]
                end
            end
        end
    elseif groups == 1 # second specialized case for better performance
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                for in_channel in 1:in_channels, y_w in 1:kernel_height, x_w in 1:kernel_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    losses[index_batch, in_channel, y_in, x_in] += conv_layer.kernels[out_channel, in_channel, y_w, x_w] * out_losses[index_batch, out_channel, y_out, x_out]
                end
            end
        end
    else # general case for any convolution 
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                for in_channel_kernel in 1:in_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    in_channel_input = in_channel_kernel + (group - 1) * in_channels_kernels
                    losses[index_batch, in_channel_input, y_in, x_in] += conv_layer.kernels[out_channel, in_channel_kernel, y_w, x_w] * out_losses[index_batch, out_channel, y_out, x_out]
                end
            end
        end
    end

    # depad 
    if conv_layer.padding != (0, 0)
        y_pad = conv_layer.padding[1]
        x_pad = conv_layer.padding[2]
        losses = losses[:, :, y_pad+1:input_height-y_pad, x_pad+1:input_width-x_pad]
    end
   
    return losses
end

# Computes the derivative of the kernels/weights on the given layer, the results are used to optimize the kernels/weights
function multichannel_conv_gradients(conv_layer)
    # storing all the necessary shapes
    current_batch_size::Int, in_channels::Int, input_height::Int, input_width::Int = size(conv_layer.inputs)
    current_batch_size, out_channels::Int, output_height::Int, output_width::Int = size(conv_layer.outputs_no_activation) # size(conv_layer.outputs)
    out_channels, in_channels_kernels::Int, kernel_height::Int, kernel_width::Int = size(conv_layer.kernels)

    # storing often used data which will be modified
    inputs_padded::Array{Float64, 4} = conv_layer.inputs_padded

    # calculating the derivative of the out_losses
    out_losses::Array{Float64, 4} = conv_layer.losses
    if conv_layer.df != 1
        activation_function_gradients = conv_layer.df(conv_layer.outputs, conv_layer.outputs_no_activation) # outputs becomes activation_function_gradients (for allocating as little as possible)
        out_losses = activation_function_gradients .= (*).(out_losses, activation_function_gradients)
        # out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
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
    # actual computation 
    if groups == 1 && stride == (1, 1) && dilation == (1, 1) # very specialized case for maximum performance
        @inbounds Threads.@threads for out_channel in 1:out_channels
            @turbo for in_channel in 1:in_channels, y_w in 1:kernel_height, x_w in 1:kernel_width
                value = 0.00
                for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                    value += inputs_padded[index_batch, in_channel, y_out + y_w - 1, x_out + x_w - 1] * out_losses[index_batch, out_channel, y_out, x_out]
                end
                gradients[out_channel, in_channel, y_w, x_w] = value
            end
        end
    elseif groups == 1 # second specialized case for better performance
        @inbounds Threads.@threads for out_channel in 1:out_channels
            @turbo for in_channel in 1:in_channels, y_w in 1:kernel_height, x_w in 1:kernel_width
                value = 0.00
                for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                    m = y_out + (y_stride - 1) * (y_out - 1)
                    n = x_out + (x_stride - 1) * (x_out - 1)
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    value += inputs_padded[index_batch, in_channel, y_in, x_in] * out_losses[index_batch, out_channel, y_out, x_out]
                end
                gradients[out_channel, in_channel, y_w, x_w] = value
            end
        end
    else # general case for any convolution 
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
    end
    
    return gradients
end

# Computes the derivative of the bias on the given layer, the results are used to optimize the bias
function multichannel_conv_bias_gradients(conv_layer)
    current_batch_size, out_channels, output_height, output_width = size(conv_layer.outputs_no_activation) # size(conv_layer.outputs)

    # calculating derivative of the activation function
    out_losses::Array{Float64, 4} = conv_layer.losses
    if conv_layer.df != 1
        activation_function_gradients = conv_layer.df(conv_layer.outputs, conv_layer.outputs_no_activation) # outputs becomes activation_function_gradients (for allocating as little as possible)
        out_losses = activation_function_gradients .= (*).(out_losses, activation_function_gradients)
        # out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
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
        error("output_padding must be smaller than either stride or dilation, but got invalid values: y_output_padding: $y_out_padding x_output_padding: $x_out_padding y_stride: $y_stride x_stride: $x_stride y_dilation: $y_dilation x_dilation: $x_dilation")
    end

    output_height = (input_height - 1) * y_stride + y_dilation * (kernel_height - 1) + y_out_padding + 1
    output_width = (input_width - 1) * x_stride + x_dilation * (kernel_width - 1) + x_out_padding + 1

    outputs = zeros(current_batch_size, out_channels_kernels * groups, output_height, output_width)

    in_channels_per_group = in_channels ÷ groups
    # actual computation
    @inbounds Threads.@threads for index_batch in 1:current_batch_size

        if groups == 1 && stride == (1, 1) && dilation == (1, 1) # very specialized case for maximum performance
            # println("forward: Case 1")
            @turbo for in_channel in 1:in_channels, y_in in 1:input_height, x_in in 1:input_width
                for out_channel in 1:out_channels_kernels * groups, y_w in 1:kernel_height, x_w in 1:kernel_width
                    outputs[index_batch, out_channel, y_in + y_w - 1, x_in + x_w - 1] += kernels[in_channel, out_channel, y_w, x_w] * inputs[index_batch, in_channel, y_in, x_in]
                end
            end
        elseif groups == 1 # second specialized case for better performance
            # println("forward: Case 2")
            @turbo for in_channel in 1:in_channels, y_in in 1:input_height, x_in in 1:input_width
                m = y_in + (y_stride - 1) * (y_in - 1)
                n = x_in + (x_stride - 1) * (x_in - 1)
                for out_channel in 1:out_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                    y_out = m + (y_w - 1) * y_dilation
                    x_out = n + (x_w - 1) * x_dilation
                    outputs[index_batch, out_channel, y_out, x_out] += kernels[in_channel, out_channel, y_w, x_w] * inputs[index_batch, in_channel, y_in, x_in]
                end
            end
        else # general case for any convolution 
            # println("forward: Case 3")
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
        activation_function_gradients = conv_layer.df(conv_layer.outputs, conv_layer.outputs_no_activation) # outputs becomes activation_function_gradients (for allocating as little as possible)
        out_losses = activation_function_gradients .= (*).(out_losses, activation_function_gradients)
        # out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
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

        if conv_layer.groups == 1 && conv_layer.stride == (1, 1) && conv_layer.dilation == (1, 1) # very specialized case for maximum performance
            @turbo for in_channel in 1:in_channels, y_in in 1:input_height, x_in in 1:input_width
                value = 0.00
                for out_channel in 1:out_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                    value += conv_layer.kernels[in_channel, out_channel, y_w, x_w] * out_losses[index_batch, out_channel, y_in + y_w - 1, x_in + x_w - 1]
                end
                losses[index_batch, in_channel, y_in, x_in] = value
            end
        elseif conv_layer.groups == 1 # second specialized case for better performance
            @turbo for in_channel in 1:in_channels, y_in in 1:input_height, x_in in 1:input_width
                m = y_in + (y_stride - 1) * (y_in - 1)
                n = x_in + (x_stride - 1) * (x_in - 1)
                value = 0.00
                for out_channel in 1:out_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                    y_out = m + (y_w - 1) * y_dilation
                    x_out = n + (x_w - 1) * x_dilation
                    value += conv_layer.kernels[in_channel, out_channel, y_w, x_w] * out_losses[index_batch, out_channel, y_out, x_out]
                end
                losses[index_batch, in_channel, y_in, x_in] = value
            end
        else # general case for any convolution 
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
        activation_function_gradients = conv_layer.df(conv_layer.outputs, conv_layer.outputs_no_activation) # outputs becomes activation_function_gradients (for allocating as little as possible)
        out_losses = activation_function_gradients .= (*).(out_losses, activation_function_gradients)
       #  out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
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

    if conv_layer.groups == 1 && conv_layer.stride == (1, 1) && conv_layer.dilation == (1, 1) # very specialized case for maximum performance
        @inbounds Threads.@threads for in_channel in 1:in_channels

            @turbo for out_channel in 1:out_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                value = 0.00
                for y_in in 1:input_height, x_in in 1:input_width
                    for index_batch in 1:current_batch_size
                        value += conv_layer.inputs[index_batch, in_channel, y_in, x_in] * out_losses[index_batch, out_channel, y_in + y_w - 1, x_in + x_w - 1]
                    end
                end
                gradients[in_channel, out_channel, y_w, x_w] = value
            end
    
        end
    elseif conv_layer.groups == 1 # second specialized case for better performance
        @inbounds Threads.@threads for in_channel in 1:in_channels

            @turbo for out_channel in 1:out_channels_kernels, y_w in 1:kernel_height, x_w in 1:kernel_width
                value = 0.00
                for y_in in 1:input_height, x_in in 1:input_width
                    m = y_in + (y_stride - 1) * (y_in - 1)
                    n = x_in + (x_stride - 1) * (x_in - 1)
                    y_out = m + (y_w - 1) * y_dilation
                    x_out = n + (x_w - 1) * x_dilation
                    for index_batch in 1:current_batch_size
                        value += conv_layer.inputs[index_batch, in_channel, y_in, x_in] * out_losses[index_batch, out_channel, y_out, x_out]
                    end
                end
                gradients[in_channel, out_channel, y_w, x_w] = value
            end
    
        end
    else # general case for any convolution 
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
        activation_function_gradients = conv_layer.df(conv_layer.outputs, conv_layer.outputs_no_activation) # outputs becomes activation_function_gradients (for allocating as little as possible)
        out_losses = activation_function_gradients .= (*).(out_losses, activation_function_gradients)
        # out_losses = out_losses .* conv_layer.df(conv_layer.outputs_no_activation)
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