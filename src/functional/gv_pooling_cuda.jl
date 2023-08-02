using cuDNN: CUDNN_POOLING_MAX, CUDNN_POOLING_MAX_DETERMINISTIC, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN
using cuDNN: scalingParameter, cudnnPoolingDescriptor, cudnnPoolingMode_t, cudnnPoolingForward!, cudnnPoolingForward, cudnnPoolingBackward

function cudnnPoolingDescriptor(input::DenseCuArray{T}, kernel_size::NTuple{2, Int}, mode::cudnnPoolingMode_t; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0)) where {T <: CUDNNFloat}
    nanOpt = CUDNN_NOT_PROPAGATE_NAN
    return cudnnPoolingDescriptor(mode, nanOpt, Cint(ndims(input)-2), collect(Cint, kernel_size), collect(Cint, padding), collect(Cint, stride))
end

function maximum_pooling2d!(output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: CUDNNFloat}
    if dilation != (1, 1)
        error("GradValley: maximum_pooling2d!: for pooling, only dilation = (1, 1) is supported in cuDNN")
    end
    pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_MAX, stride=stride, padding=padding)
    # pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_MAX_DETERMINISTIC, stride=stride, padding=padding)

    return cudnnPoolingForward!(output, input, pool_descriptor)
end

function maximum_pooling2d(input::DenseCuArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: CUDNNFloat}
    if dilation != (1, 1)
        error("GradValley: maximum_pooling2d: for pooling, only dilation = (1, 1) is supported in cuDNN")
    end
    pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_MAX, stride=stride, padding=padding)
    # pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_MAX_DETERMINISTIC, stride=stride, padding=padding)

    return cudnnPoolingForward(input, pool_descriptor)
end

function maximum_pooling2d_backward!(input_gradient::DenseCuArray{T, 4}, output_gradient::DenseCuArray{T, 4}, output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: CUDNNFloat}
    if dilation != (1, 1)
        error("GradValley: maximum_pooling2d_backward!: for pooling, only dilation = (1, 1) is supported in cuDNN")
    end
    pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_MAX, stride=stride, padding=padding)
    # pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_MAX_DETERMINISTIC, stride=stride, padding=padding)

    input_gradient_descriptor = cudnnTensorDescriptor(input_gradient)
    output_gradient_descriptor = cudnnTensorDescriptor(output_gradient)
    output_descriptor = cudnnTensorDescriptor(output)
    input_descriptor = cudnnTensorDescriptor(input)

    alpha, beta = scalingParameter(T, 1), scalingParameter(T, 0)

    cudnnPoolingBackward(cuDNN.handle(), pool_descriptor, alpha, output_descriptor, output, output_gradient_descriptor, output_gradient, input_descriptor, input, beta, input_gradient_descriptor, input_gradient)

    return input_gradient
end

function maximum_pooling2d_backward(output_gradient::DenseCuArray{T, 4}, output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)

    input_gradient = CUDA.zeros(T, input_width, input_height, channels, current_batch_size)

    return maximum_pooling2d_backward!(input_gradient, output_gradient, output, input, kernel_size, stride=stride, padding=padding, dilation=dilation)
end

function average_pooling2d!(output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: CUDNNFloat}
    if dilation != (1, 1)
        error("GradValley: average_pooling2d!: for pooling, only dilation = (1, 1) is supported in cuDNN")
    end
    pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, stride=stride, padding=padding)
    # pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, stride=stride, padding=padding)

    return cudnnPoolingForward!(output, input, pool_descriptor)
end

function average_pooling2d(input::DenseCuArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: CUDNNFloat}
    if dilation != (1, 1)
        error("GradValley: average_pooling2d: for pooling, only dilation = (1, 1) is supported in cuDNN")
    end
    pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, stride=stride, padding=padding)
    # pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, stride=stride, padding=padding)

    return cudnnPoolingForward(input, pool_descriptor)
end

function average_pooling2d_backward!(input_gradient::DenseCuArray{T, 4}, output_gradient::DenseCuArray{T, 4}, output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: CUDNNFloat}
    if dilation != (1, 1)
        error("GradValley: average_pooling2d_backward!: for pooling, only dilation = (1, 1) is supported in cuDNN")
    end
    pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, stride=stride, padding=padding)
    # pool_descriptor = cudnnPoolingDescriptor(input, kernel_size, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, stride=stride, padding=padding)

    input_gradient_descriptor = cudnnTensorDescriptor(input_gradient)
    output_gradient_descriptor = cudnnTensorDescriptor(output_gradient)
    output_descriptor = cudnnTensorDescriptor(output)
    input_descriptor = cudnnTensorDescriptor(input)

    alpha, beta = scalingParameter(T, 1), scalingParameter(T, 0)

    cudnnPoolingBackward(cuDNN.handle(), pool_descriptor, alpha, output_descriptor, output, output_gradient_descriptor, output_gradient, input_descriptor, input, beta, input_gradient_descriptor, input_gradient)

    return input_gradient
end

function average_pooling2d_backward(output_gradient::DenseCuArray{T, 4}, output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, kernel_size::NTuple{2, Int}; stride::NTuple{2, Int}=kernel_size, padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1)) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)

    input_gradient = CUDA.zeros(T, input_width, input_height, channels, current_batch_size)

    return average_pooling2d_backward!(input_gradient, output_gradient, output, input, kernel_size, stride=stride, padding=padding, dilation=dilation)
end

function adaptive_average_pooling2d!_kernel(output::CuDeviceArray{T, 4}, input::CuDeviceArray{T, 4}, output_size::NTuple{2, Int}, y_in_indices, x_in_indices) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_height, output_width = output_size

    channel = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y_out = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    x_out = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    if channel <= channels && y_out <= output_height && x_out <= output_width 
        @inbounds for index_batch in 1:current_batch_size
            kernel_sum = zero(eltype(output)) # 0.00
            for y_in in y_in_indices[y_out], x_in in x_in_indices[x_out]
                kernel_sum += input[x_in, y_in, channel, index_batch]
            end
            output[x_out, y_out, channel, index_batch] = kernel_sum / (length(y_in_indices[y_out]) * length(x_in_indices[x_out]))
        end
    end

    return nothing
end

function adaptive_average_pooling2d!(output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, output_size::NTuple{2, Int}) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_height, output_width = output_size

    y_in_indices = CuArray(get_in_indices(input_height, output_height))
    x_in_indices = CuArray(get_in_indices(input_width, output_width))

    kernel = @cuda launch=false adaptive_average_pooling2d!_kernel(output, input, output_size, y_in_indices, x_in_indices)

    config = CUDA.launch_configuration(kernel.fun)
    
    threads_per_dim = Int(floor(config.threads^(1/3)))
    
    channel_threads = min(channels, threads_per_dim)
    channel_blocks = cld(channels, channel_threads)

    height_threads = min(output_height, threads_per_dim)
    height_blocks = cld(output_height, height_threads)

    width_threads = min(output_width, threads_per_dim)
    width_blocks = cld(output_width, width_threads)

    kernel(output, input, output_size, y_in_indices, x_in_indices, threads=(channel_threads, height_threads, width_threads), blocks=(channel_blocks, height_blocks, width_blocks))

    return output
end

function adaptive_average_pooling2d(input::DenseCuArray{T, 4}, output_size::NTuple{2, Int}) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_height, output_width = output_size

    output = CUDA.zeros(eltype(input), output_width, output_height, channels, current_batch_size)

    return adaptive_average_pooling2d!(output, input, output_size)
end

function adaptive_average_pooling2d_backward!_kernel(input_gradient::CuDeviceArray{T, 4}, output_gradient::CuDeviceArray{T, 4}, y_in_indices, x_in_indices) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input_gradient)
    output_height, output_width = size(output_gradient)

    channel = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y_out = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    x_out = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    if channel <= channels && y_out <= output_height && x_out <= output_width 
        @inbounds for index_batch in 1:current_batch_size
            for y_in in y_in_indices[y_out], x_in in x_in_indices[x_out]
                input_gradient[x_in, y_in, channel, index_batch] += output_gradient[x_out, y_out, channel, index_batch] / (length(y_in_indices[y_out]) * length(x_in_indices[x_out]))
            end
        end
    end

    return nothing
end

function adaptive_average_pooling2d_backward!(input_gradient::DenseCuArray{T, 4}, output_gradient::DenseCuArray{T, 4}) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input_gradient)
    output_width, output_height, channels, current_batch_size = size(output_gradient)

    y_in_indices = CuArray(get_in_indices(input_height, output_height))
    x_in_indices = CuArray(get_in_indices(input_width, output_width))

    kernel = @cuda launch=false adaptive_average_pooling2d_backward!_kernel(input_gradient, output_gradient, y_in_indices, x_in_indices)

    config = CUDA.launch_configuration(kernel.fun)
    
    threads_per_dim = Int(floor(config.threads^(1/3)))
    
    channel_threads = min(channels, threads_per_dim)
    channel_blocks = cld(channels, channel_threads)

    height_threads = min(output_height, threads_per_dim)
    height_blocks = cld(output_height, height_threads)

    width_threads = min(output_width, threads_per_dim)
    width_blocks = cld(output_width, width_threads)

    kernel(input_gradient, output_gradient, y_in_indices, x_in_indices, threads=(channel_threads, height_threads, width_threads), blocks=(channel_blocks, height_blocks, width_blocks))

    return input_gradient
end

function adaptive_average_pooling2d_backward(output_gradient::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_width, output_height, channels, current_batch_size = size(output_gradient)

    input_gradient = CUDA.zeros(eltype(output_gradient), input_width, input_height, channels, current_batch_size)

    return adaptive_average_pooling2d_backward!(input_gradient, output_gradient)
end

function adaptive_maximum_pooling2d!_kernel(output::CuDeviceArray{T, 4}, input::CuDeviceArray{T, 4}, output_size::NTuple{2, Int}, y_in_indices, x_in_indices) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_height, output_width = output_size

    channel = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y_out = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    x_out = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    if channel <= channels && y_out <= output_height && x_out <= output_width 
        @inbounds for index_batch in 1:current_batch_size
            first_y_in_index = y_in_indices[y_out][1]
            first_x_in_index = x_in_indices[x_out][1]
            max_value = input[first_x_in_index, first_y_in_index, channel, index_batch]
            for y_in in y_in_indices[y_out], x_in in x_in_indices[x_out]
                value = input[x_in, y_in, channel, index_batch]
                if value > max_value
                    max_value = value
                end
            end
            output[x_out, y_out, channel, index_batch] = max_value
        end
    end

    return nothing
end

function adaptive_maximum_pooling2d!(output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, output_size::NTuple{2, Int}) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_height, output_width = output_size

    y_in_indices = CuArray(get_in_indices(input_height, output_height))
    x_in_indices = CuArray(get_in_indices(input_width, output_width))

    kernel = @cuda launch=false adaptive_maximum_pooling2d!_kernel(output, input, output_size, y_in_indices, x_in_indices)

    config = CUDA.launch_configuration(kernel.fun)
    
    threads_per_dim = Int(floor(config.threads^(1/3)))
    
    channel_threads = min(channels, threads_per_dim)
    channel_blocks = cld(channels, channel_threads)

    height_threads = min(output_height, threads_per_dim)
    height_blocks = cld(output_height, height_threads)

    width_threads = min(output_width, threads_per_dim)
    width_blocks = cld(output_width, width_threads)

    kernel(output, input, output_size, y_in_indices, x_in_indices, threads=(channel_threads, height_threads, width_threads), blocks=(channel_blocks, height_blocks, width_blocks))

    return output
end

function adaptive_maximum_pooling2d(input::DenseCuArray{T, 4}, output_size::NTuple{2, Int}) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)
    output_height, output_width = output_size

    output = CUDA.zeros(eltype(input), output_width, output_height, channels, current_batch_size)

    return adaptive_maximum_pooling2d!(output, input, output_size)
end

function adaptive_maximum_pooling2d_backward!_kernel(input_gradient::CuDeviceArray{T, 4}, output_gradient::CuDeviceArray{T, 4}, input::CuDeviceArray{T, 4}, y_in_indices, x_in_indices) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input_gradient)
    output_height, output_width = size(output_gradient)

    channel = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y_out = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    x_out = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    if channel <= channels && y_out <= output_height && x_out <= output_width 
        @inbounds for index_batch in 1:current_batch_size

            first_y_in_index = y_in_indices[y_out][1]
            first_x_in_index = x_in_indices[x_out][1]

            max_value = input[first_x_in_index, first_y_in_index, channel, index_batch]
            max_index_y_in = first_y_in_index
            max_index_x_in = first_x_in_index

            for y_in in y_in_indices[y_out], x_in in x_in_indices[x_out]
                value = input[x_in, y_in, channel, index_batch]
                if value > max_value
                    max_value = value
                    max_index_y_in = y_in
                    max_index_x_in = x_in
                end
            end

            input_gradient[max_index_x_in, max_index_y_in, channel, index_batch] += output_gradient[x_out, y_out, channel, index_batch]
        end
    end

    return nothing
end

function adaptive_maximum_pooling2d_backward!(input_gradient::DenseCuArray{T, 4}, output_gradient::DenseCuArray{T, 4}, input::DenseArray{T, 4}) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input_gradient)
    output_width, output_height, channels, current_batch_size = size(output_gradient)

    y_in_indices = CuArray(get_in_indices(input_height, output_height))
    x_in_indices = CuArray(get_in_indices(input_width, output_width))

    kernel = @cuda launch=false adaptive_maximum_pooling2d_backward!_kernel(input_gradient, output_gradient, input, y_in_indices, x_in_indices)

    config = CUDA.launch_configuration(kernel.fun)
    
    threads_per_dim = Int(floor(config.threads^(1/3)))
    
    channel_threads = min(channels, threads_per_dim)
    channel_blocks = cld(channels, channel_threads)

    height_threads = min(output_height, threads_per_dim)
    height_blocks = cld(output_height, height_threads)

    width_threads = min(output_width, threads_per_dim)
    width_blocks = cld(output_width, width_threads)

    kernel(input_gradient, output_gradient, input, y_in_indices, x_in_indices, threads=(channel_threads, height_threads, width_threads), blocks=(channel_blocks, height_blocks, width_blocks))

    return input_gradient
end

function adaptive_maximum_pooling2d_backward(output_gradient::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, channels, current_batch_size = size(input)

    input_gradient = CUDA.zeros(eltype(output_gradient), input_width, input_height, channels, current_batch_size)

    return adaptive_maximum_pooling2d_backward!(input_gradient, output_gradient, input)
end