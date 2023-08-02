function convolution2d!(output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}, bias::AbstractVector{T}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, in_channels, current_batch_size = size(input)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)
    output_width, output_height = size(output)[1:2]

    # performing padding
    if padding != (0, 0)
        input = zero_pad_2d(input, padding)
    end
    
    y_stride, x_stride = stride
    y_dilation, x_dilation = dilation
    out_channels_per_group = out_channels ÷ groups
    typed_zero = zero(eltype(output))
    # actual computation
    if groups == 1 && stride == (1, 1) && dilation == (1, 1) # very specialized case for maximum performance
        # println("very specialized case for maximum performance")
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                value = typed_zero
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    value += input[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value + bias[out_channel]
            end
        end
    elseif groups == 1 # second specialized case for better performance
        # println("second specialized case for better performance")
        #=
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                value = typed_zero
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    value += input[x_in, y_in, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value + bias[out_channel]
            end
        end
        =#

        input_m_indices = zeros(Int, output_height)
        input_n_indices = zeros(Int, output_width)
        @turbo for y_out in 1:output_height
            m = y_out + (y_stride - 1) * (y_out - 1)
            input_m_indices[y_out] = m
        end
        @turbo for x_out in 1:output_width
            n = x_out + (x_stride - 1) * (x_out - 1)
            input_n_indices[x_out] = n
        end

        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                m = input_m_indices[y_out]
                n = input_n_indices[x_out]
                value = typed_zero
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    value += input[x_in, y_in, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value + bias[out_channel]
            end
        end

    else # general case for any convolution 
        # println("general case for any convolution")
        #=
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                value = typed_zero
                for in_channel_weight in 1:in_channels_weight, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    in_channel_input = in_channel_weight + (group - 1) * in_channels_weight
                    value += input[x_in, y_in, in_channel_input, index_batch] * weight[x_w, y_w, in_channel_weight, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value + bias[out_channel]
            end
        end
        =#

        input_m_indices = zeros(Int, output_height)
        input_n_indices = zeros(Int, output_width)
        @turbo for y_out in 1:output_height
            m = y_out + (y_stride - 1) * (y_out - 1)
            input_m_indices[y_out] = m
        end
        @turbo for x_out in 1:output_width
            n = x_out + (x_stride - 1) * (x_out - 1)
            input_n_indices[x_out] = n
        end

        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, y_out in 1:output_height, x_out in 1:output_width
                m = input_m_indices[y_out]
                n = input_n_indices[x_out]
                out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                value = typed_zero
                for in_channel_weight in 1:in_channels_weight, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    in_channel_input = in_channel_weight + (group - 1) * in_channels_weight
                    value += input[x_in, y_in, in_channel_input, index_batch] * weight[x_w, y_w, in_channel_weight, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value + bias[out_channel]
            end
        end
    end

    return output
end

function convolution2d(input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}, bias::AbstractVector{T}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, in_channels, current_batch_size = size(input)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)
    # calculating shape of output
    output_height, output_width = calculate_output_shape(input_height, input_width, weight_height, weight_width, stride=stride, padding=padding, dilation=dilation)

    output = zeros(eltype(input), output_width, output_height, out_channels, current_batch_size)

    return convolution2d!(output, input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function convolution2d_data_backward!(input_gradient::AbstractArray{T, 4}, output_gradient::AbstractArray{T, 4}, weight::AbstractArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)
    # because in the actual computation section, values are added, it's saver to reset the given input_gradient first
    input_gradient .= zero(eltype(input_gradient))
    # check if input_gradient must be padded 
    if padding != (0, 0)
        input_gradient_padded = zero_pad_2d(input_gradient, padding)
    else
        input_gradient_padded = input_gradient
    end
    # store the size of input after padding 
    input_width, input_height, in_channels, current_batch_size = size(input_gradient_padded) # size after padding 

    y_stride, x_stride = stride
    y_dilation, x_dilation = dilation
    out_channels_per_group = out_channels ÷ groups
    # actual computation
    if groups == 1 && stride == (1, 1) && dilation == (1, 1) # very specialized case for maximum performance
        # println("very specialized case for maximum performance")
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    input_gradient_padded[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] += weight[x_w, y_w, in_channel, out_channel] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
            end
        end
    elseif groups == 1 # second specialized case for better performance
        # println("second specialized case for better performance")
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    input_gradient_padded[x_in, y_in, in_channel, index_batch] += weight[x_w, y_w, in_channel, out_channel] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
            end
        end
    else # general case for any convolution 
        # println("general case for any convolution")
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            for out_channel_per_group in 1:out_channels_per_group 
                @turbo for group in 1:groups, y_out in 1:output_height, x_out in 1:output_width # @turbo 
                    m = y_out + (y_stride - 1) * (y_out - 1)
                    n = x_out + (x_stride - 1) * (x_out - 1)
                    out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                    for in_channel_weight in 1:in_channels_weight, y_w in 1:weight_height, x_w in 1:weight_width
                        y_in = m + (y_w - 1) * y_dilation
                        x_in = n + (x_w - 1) * x_dilation
                        in_channel_input = in_channel_weight + (group - 1) * in_channels_weight
                        input_gradient_padded[x_in, y_in, in_channel_input, index_batch] += weight[x_w, y_w, in_channel_weight, out_channel] * output_gradient[x_out, y_out, out_channel, index_batch]
                    end
                end
            end
        end
    end

    # depad 
    if padding != (0, 0)
        y_pad, x_pad = padding
        input_gradient .= input_gradient_padded[x_pad+1:input_width-x_pad, y_pad+1:input_height-y_pad, :, :]
    end
   
    return input_gradient
end

function convolution2d_data_backward(output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)
    input_width, input_height, in_channels, current_batch_size = size(input)
    # allocate the input_gradient with size of input before padding 
    # input_gradient = zeros(eltype(output_gradient), input_width, input_height, in_channels_weight * groups, current_batch_size)
    input_gradient = zeros(eltype(output_gradient), input_width, input_height, in_channels, current_batch_size)

    return convolution2d_data_backward!(input_gradient, output_gradient, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function convolution2d_filter_backward!(weight_gradient::AbstractArray{T, 4}, output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, in_channels, current_batch_size = size(input)
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)

    # check if input must be padded 
    if padding != (0, 0)
        input_padded = zero_pad_2d(input, padding)
    else
        input_padded = input
    end

    y_stride, x_stride = stride
    y_dilation, x_dilation = dilation
    out_channels_per_group = out_channels ÷ groups
    # actual computation 
    typed_zero = zero(eltype(output_gradient))
    # because in the actual computation section, values are added, it's saver to reset the given weight_gradient first
    ### weight_gradient .= zero(eltype(weight_gradient))
    if groups == 1 && stride == (1, 1) && dilation == (1, 1) # very specialized case for maximum performance
        # println("very specialized case for maximum performance")
        @inbounds Threads.@threads for out_channel in 1:out_channels
            @turbo for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                value = typed_zero
                for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                    value += input_padded[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
                weight_gradient[x_w, y_w, in_channel, out_channel] = value
            end
        end

        #=
        @inbounds @tturbo for index_batch in 1:current_batch_size
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                output_gradient_value = output_gradient[x_out, y_out, out_channel, index_batch]
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    # because values are added here, do this: weight_gradient .= zero(eltype(weight_gradient))
                    weight_gradient[x_w, y_w, in_channel, out_channel] += input_padded[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] * output_gradient_value
                end
            end
        end
        =#

        #=
        @inbounds @tturbo for in_channel in 1:in_channels
            for out_channel in 1:out_channels, y_w in 1:weight_height, x_w in 1:weight_width
                value = typed_zero
                for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                    value += input_padded[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
                weight_gradient[x_w, y_w, in_channel, out_channel] = value
            end
        end
        =#

    elseif groups == 1 # second specialized case for better performance
        # println("second specialized case for better performance")
        @inbounds Threads.@threads for out_channel in 1:out_channels
            @turbo for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                value = typed_zero
                for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                    m = y_out + (y_stride - 1) * (y_out - 1)
                    n = x_out + (x_stride - 1) * (x_out - 1)
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    value += input_padded[x_in, y_in, in_channel, index_batch] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
                weight_gradient[x_w, y_w, in_channel, out_channel] = value
            end
        end
    else # general case for any convolution 
        # println("general case for any convolution")
        @inbounds Threads.@threads for out_channel_per_group in 1:out_channels_per_group
            @turbo for group in 1:groups, in_channel_weight in 1:in_channels_weight, y_w in 1:weight_height, x_w in 1:weight_width
                value = typed_zero
                for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                    m = y_out + (y_stride - 1) * (y_out - 1)
                    n = x_out + (x_stride - 1) * (x_out - 1)
                    out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    in_channel_input = in_channel_weight + (group - 1) * in_channels_weight
                    value += input_padded[x_in, y_in, in_channel_input, index_batch] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
                weight_gradient[x_w, y_w, in_channel_weight, out_channel] = value
            end
        end
    end
    
    return weight_gradient
end

function convolution2d_filter_backward(output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)
    # allocate the input_gradient with size of input before padding 
    weight_gradient = zeros(eltype(output_gradient), weight_width, weight_height, in_channels_weight, out_channels)

    return convolution2d_filter_backward!(weight_gradient, output_gradient, input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function convolution2d_bias_backward!(bias_gradient::AbstractVector{T}, output_gradient::AbstractArray{T, 4}) where {T <: Real}
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)

    typed_zero = zero(eltype(bias_gradient))
    bias_gradient .= typed_zero
    @turbo for out_channel in 1:out_channels
        value = typed_zero
        for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
            value += output_gradient[x_out, y_out, out_channel, index_batch]
        end
        bias_gradient[out_channel] = value
    end

    return bias_gradient
end

function convolution2d_bias_backward(output_gradient::AbstractArray{T, 4}) where {T <: Real}
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    bias_gradient = zeros(eltype(output_gradient), out_channels)

    return convolution2d_bias_backward!(bias_gradient, output_gradient)
end

function deconvolution2d!(output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}, bias::AbstractVector{T}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    output = convolution2d_data_backward!(output, input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # storing all the necessary shapes
    weight_width, weight_height, out_channels_weight, out_channels = size(weight)
    output_width, output_height, out_channels, current_batch_size = size(output)
    # adding bias if necessary
    @turbo for index_batch in 1:current_batch_size, out_channel in 1:out_channels_weight * groups
        bias_value = bias[out_channel]
        for y_out in 1:output_height, x_out in 1:output_width
            output[x_out, y_out, out_channel, index_batch] += bias_value
        end
    end

    return output
end

function deconvolution2d(input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}, bias::AbstractVector{T}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), output_padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, in_channels, current_batch_size = size(input)
    weight_width, weight_height, out_channels_weight, in_channels = size(weight)
    # calculating shape of output
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + dilation[1] * (weight_height - 1) + output_padding[1] + 1
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + dilation[2] * (weight_width - 1) + output_padding[2] + 1

    output = zeros(eltype(input), output_width, output_height, out_channels_weight * groups, current_batch_size)

    return deconvolution2d!(output, input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function deconvolution2d_data_backward!(input_gradient::AbstractArray{T, 4}, output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, in_channels, current_batch_size = size(input_gradient)
    input_gradient = convolution2d!(input_gradient, output_gradient, weight, zeros(eltype(input_gradient), in_channels), stride=stride, padding=padding, dilation=dilation, groups=groups)

    return input_gradient
end

function deconvolution2d_data_backward(output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, in_channels, current_batch_size = size(input)
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)

    input_gradient = zeros(eltype(output_gradient), input_width, input_height, in_channels, current_batch_size)

    return deconvolution2d_data_backward!(input_gradient, output_gradient, input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function deconvolution2d_filter_backward!(weight_gradient::AbstractArray{T, 4}, output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    weight_gradient = convolution2d_filter_backward!(weight_gradient, input, output_gradient, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)

    return weight_gradient
end

function deconvolution2d_filter_backward(output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight::AbstractArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: Real}
    # storing all the necessary shapes
    weight_width, weight_height, out_channels_weight, in_channels = size(weight)

    weight_gradient = zeros(eltype(output_gradient), weight_width, weight_height, out_channels_weight, in_channels)

    return deconvolution2d_filter_backward!(weight_gradient, output_gradient, input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function deconvolution2d_bias_backward!(bias_gradient::AbstractVector{T}, output_gradient::AbstractArray{T, 4}) where {T <: Real}
    return convolution2d_bias_backward!(bias_gradient, output_gradient)
end

function deconvolution2d_bias_backward(output_gradient::AbstractArray{T, 4}) where {T <: Real}
    return convolution2d_bias_backward(output_gradient)
end