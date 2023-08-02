function maximum_pooling2d!(output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), return_data_for_backprop::Bool=false) where {T <: Real}
    input_width, input_height, channels, current_batch_size = size(input)
    output_width, output_height, channels, current_batch_size = size(output)
    kernel_height, kernel_width = kernel_size

    # performing padding
    if padding != (0, 0)
        input = zero_pad_2d(input, padding)
    end

    # positions are necessary for backpropagation (only for maximum pooling)
    # each position in the output is assigned to the position in the input with the largest value
    # (which is also the value in the output matrix at the given position in output)
    y_positions = Array{Int, 4}(undef, output_width, output_height, channels, current_batch_size)
    x_positions = Array{Int, 4}(undef, output_width, output_height, channels, current_batch_size)
    positions = (y_positions, x_positions)

    y_stride, x_stride = stride
    y_dilation, x_dilation = dilation
    # going through all data in batch
    @inbounds Threads.@threads for index_batch in 1:current_batch_size

        # compute pooling for each channel seperatly
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
            m = y_out + (y_stride - 1) * (y_out - 1)
            n = x_out + (x_stride - 1) * (x_out - 1)
            # initialize maximum value and indices
            max_y, max_x = m, n
            max = input[n, m, channel, index_batch]
            for y_w in 1:kernel_height, x_w in 1:kernel_width
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                value = input[x_in, y_in, channel, index_batch]
                if value > max
                    max_y, max_x = y_in, x_in
                    max = value
                end
            end
            output[x_out, y_out, channel, index_batch] = max
            # the positions are saved
            y_positions[x_out, y_out, channel, index_batch] = max_y
            x_positions[x_out, y_out, channel, index_batch] = max_x
        end

    end

    if return_data_for_backprop
        return output, positions
    else
        return output
    end
end

function maximum_pooling2d(input::AbstractArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), return_data_for_backprop::Bool=false) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    kernel_height, kernel_width = kernel_size
    # calculating shape of output
    output_height, output_width = calculate_output_shape(input_height, input_width, kernel_height, kernel_width, stride=stride, padding=padding, dilation=dilation)

    output = zeros(eltype(input), output_width, output_height, channels, current_batch_size)
    
    return maximum_pooling2d!(output, input, kernel_size, stride=stride, padding=padding, dilation=dilation, return_data_for_backprop=return_data_for_backprop)
end

function maximum_pooling2d_backward!(input_gradient::AbstractArray{T, 4}, output_gradient::AbstractArray{T, 4}, data_for_backprop::NTuple{2, Array{Int, 4}}; padding::NTuple{2, Int}=(0, 0)) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input_gradient)
    output_width, output_height, channels, current_batch_size = size(output_gradient)
    positions = data_for_backprop
    y_positions, x_positions = positions

    y_pad, x_pad = padding
    # going through all data in batch
    @inbounds Threads.@threads for index_batch in 1:current_batch_size

        # going through every output
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
            value = output_gradient[x_out, y_out, channel, index_batch]
            y_position = y_positions[x_out, y_out, channel, index_batch]
            x_position = x_positions[x_out, y_out, channel, index_batch]
            if !(y_position <= y_pad)
                y_position -= y_pad
            end
            if !(x_position <= x_pad)
                x_position -= x_pad
            end
            if !(y_position > input_height || x_position > input_width)
                input_gradient[x_position, y_position, channel, index_batch] += value
            end
        end
    
    end

    return input_gradient
end

function maximum_pooling2d_backward(output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}, data_for_backprop::NTuple{2, Array{Int, 4}}; padding::NTuple{2, Int}=(0, 0)) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)

    input_gradient = zeros(eltype(output_gradient), input_width, input_height, channels, current_batch_size)

    return maximum_pooling2d_backward!(input_gradient, output_gradient, data_for_backprop, padding=padding)
end

function average_pooling2d!(output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_width, output_height, channels, current_batch_size = size(output)
    kernel_height, kernel_width = kernel_size

    # performing padding
    if padding != (0, 0)
        input = zero_pad_2d(input, padding)
    end

    y_stride, x_stride = stride
    y_dilation, x_dilation = dilation
    typed_zero = zero(eltype(output))
    # going through all data in batch
    @inbounds Threads.@threads for index_batch in 1:current_batch_size
    # @tturbo for index_batch in 1:current_batch_size # eigentlich schneller als Threads.@threads gepaart mit @turbo -> vielleicht doch nicht!

        # compute pooling for each channel seperatly
        @turbo for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
        # for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width # kein @turbo hier, stattdessen oben @tturbo, weil eigentlich schneller -> vielleicht doch nicht!
            m = y_out + (y_stride - 1) * (y_out - 1)
            n = x_out + (x_stride - 1) * (x_out - 1)
            kernel_sum = typed_zero
            for y_w in 1:kernel_height, x_w in 1:kernel_width
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                kernel_sum += input[x_in, y_in, channel, index_batch]
            end
            output[x_out, y_out, channel, index_batch] = kernel_sum / (kernel_height * kernel_width)
        end

    end

    return output
end

function average_pooling2d(input::AbstractArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    kernel_height, kernel_width = kernel_size
    # calculating shape of output
    output_height, output_width = calculate_output_shape(input_height, input_width, kernel_height, kernel_width, stride=stride, padding=padding, dilation=dilation)

    output = zeros(eltype(input), output_width, output_height, channels, current_batch_size)
    
    return average_pooling2d!(output, input, kernel_size, stride=stride, padding=padding, dilation=dilation)
end

function average_pooling2d_backward!(input_gradient::AbstractArray{T, 4}, output_gradient::AbstractArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input_gradient)
    output_width, output_height, channels, current_batch_size = size(output_gradient)
    kernel_height, kernel_width = kernel_size

    y_stride, x_stride = stride
    y_dilation, x_dilation = dilation
    y_pad, x_pad = padding
    if padding == (0, 0)
        # going through all data in batch
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            # going through every output
            @turbo for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                for y_w in 1:kernel_height, x_w in 1:kernel_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    input_gradient[x_in, y_in, channel, index_batch] += output_gradient[x_out, y_out, channel, index_batch]
                end
            end
        end
    else
        # going through all data in batch
        @inbounds Threads.@threads for index_batch in 1:current_batch_size
            # going through every output
            for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                m -= y_pad
                n -= x_pad
                for y_w in 1:kernel_height, x_w in 1:kernel_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    if 1 <= y_in <= input_height && 1 <= x_in <= input_width
                        input_gradient[x_in, y_in, channel, index_batch] += output_gradient[x_out, y_out, channel, index_batch]
                    end
                end
            end
        end
    end
    factor = (1/(kernel_height * kernel_width))
    input_gradient .= input_gradient * factor

    return input_gradient
end

function average_pooling2d_backward(output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)

    input_gradient = zeros(eltype(output_gradient), input_width, input_height, channels, current_batch_size)

    return average_pooling2d_backward!(input_gradient, output_gradient, kernel_size, stride=stride, padding=padding, dilation=dilation)
end

function adaptive_average_pooling2d!(output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, output_size::NTuple{2, Int}) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_height, output_width = output_size

    y_in_indices = get_in_indices(input_height, output_height)
    x_in_indices = get_in_indices(input_width, output_width)
    typed_zero = zero(eltype(output))
    for y_out in 1:output_height, x_out in 1:output_width
        current_y_in_indices = y_in_indices[y_out]
        current_x_in_indices = x_in_indices[x_out]
        @turbo for index_batch in 1:current_batch_size, channel in 1:channels
            kernel_sum = typed_zero
            for y_in in current_y_in_indices, x_in in current_x_in_indices
                kernel_sum += input[x_in, y_in, channel, index_batch]
            end
            output[x_out, y_out, channel, index_batch] = kernel_sum / (length(current_y_in_indices) * length(current_x_in_indices))
        end
    end

    return output
end

function adaptive_average_pooling2d(input::AbstractArray{T, 4}, output_size::NTuple{2, Int}) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_height, output_width = output_size

    output = zeros(eltype(input), output_width, output_height, channels, current_batch_size)

    return adaptive_average_pooling2d!(output, input, output_size)
end

function adaptive_average_pooling2d_backward!(input_gradient::AbstractArray{T, 4}, output_gradient::AbstractArray{T, 4}) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input_gradient)
    output_width, output_height, channels, current_batch_size = size(output_gradient)

    y_in_indices = get_in_indices(input_height, output_height)
    x_in_indices = get_in_indices(input_width, output_width)
    for y_out in 1:output_height, x_out in 1:output_width
        current_y_in_indices = y_in_indices[y_out]
        current_x_in_indices = x_in_indices[x_out]
        @turbo for index_batch in 1:current_batch_size, channel in 1:channels
            for y_in in current_y_in_indices, x_in in current_x_in_indices
                input_gradient[x_in, y_in, channel, index_batch] += output_gradient[x_out, y_out, channel, index_batch] / (length(current_y_in_indices) * length(current_x_in_indices))
            end
        end
    end

    return input_gradient
end

function adaptive_average_pooling2d_backward(output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_width, output_height, channels, current_batch_size = size(output_gradient)

    input_gradient = zeros(eltype(output_gradient), input_width, input_height, channels, current_batch_size)

    return adaptive_average_pooling2d_backward!(input_gradient, output_gradient)
end

function adaptive_maximum_pooling2d!(output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, output_size::NTuple{2, Int}; return_data_for_backprop::Bool=false) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_height, output_width = output_size

    # positions are necessary for backpropagation (only for maximum pooling)
    # each position in the output is assigned to the position in the input with the largest value
    # (which is also the value in the output matrix at the given position in output)
    y_positions = Array{Int, 4}(undef, output_width, output_height, channels, current_batch_size)
    x_positions = Array{Int, 4}(undef, output_width, output_height, channels, current_batch_size)
    positions = (y_positions, x_positions)

    y_in_indices = get_in_indices(input_height, output_height)
    x_in_indices = get_in_indices(input_width, output_width)
    # going through all data in batch
    Threads.@threads for index_batch in 1:current_batch_size
        # compute pooling for each channel seperatly
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
            values = input[x_in_indices[x_out], y_in_indices[y_out], channel, index_batch]
            max_index = argmax(values)
            output[x_out, y_out, channel, index_batch] = values[max_index] # maximum(values)
            y_positions[x_out, y_out, channel, index_batch] = y_in_indices[y_out][1] + max_index[2] - 1 # max_index[1]
            x_positions[x_out, y_out, channel, index_batch] = x_in_indices[x_out][1] + max_index[1] - 1 # max_index[2]
        end
    end

    if return_data_for_backprop
        return output, positions
    else
        return output
    end
end

function adaptive_maximum_pooling2d(input::AbstractArray{T, 4}, output_size::NTuple{2, Int}; return_data_for_backprop::Bool=false) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_height, output_width = output_size

    output = zeros(eltype(input), output_width, output_height, channels, current_batch_size)

    return adaptive_maximum_pooling2d!(output, input, output_size, return_data_for_backprop=return_data_for_backprop)
end

function adaptive_maximum_pooling2d_backward!(input_gradient::AbstractArray{T, 4}, output_gradient::AbstractArray{T, 4}, data_for_backprop::NTuple{2, Array{Int, 4}}) where {T <: Real}
    # storing all the necessary shapes
    output_width, output_height, channels, current_batch_size = size(output_gradient)
    positions = data_for_backprop
    y_positions, x_positions = positions

    # going through all data in batch
    @turbo for index_batch in 1:current_batch_size

        # going through every output
        for channel in 1:channels, y_out in 1:output_height, x_out in 1:output_width
            value = output_gradient[x_out, y_out, channel, index_batch]
            y_position = y_positions[x_out, y_out, channel, index_batch]
            x_position = x_positions[x_out, y_out, channel, index_batch]
            input_gradient[x_position, y_position, channel, index_batch] += value
        end
    
    end

    return input_gradient
end

function adaptive_maximum_pooling2d_backward(output_gradient::AbstractArray{T, 4}, input::AbstractArray{T, 4}, data_for_backprop::NTuple{2, Array{Int, 4}}) where {T <: Real}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)

    input_gradient = zeros(eltype(output_gradient), input_width, input_height, channels, current_batch_size)

    return adaptive_maximum_pooling2d_backward!(input_gradient, output_gradient, data_for_backprop)
end