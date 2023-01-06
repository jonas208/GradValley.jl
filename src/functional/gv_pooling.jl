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