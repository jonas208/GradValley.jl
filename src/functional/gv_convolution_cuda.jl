using cuDNN: cudnnDataType, cudnnTensorDescriptor
using cuDNN: CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION, CUDNN_DEFAULT_REORDER
using cuDNN: cudnnConvolutionDescriptor, cudnnConvolutionForward!, cudnnConvolutionForward,
             cudnnConvolutionBwdDataAlgoPerf, cudnnConvolutionBackwardData,
             cudnnConvolutionBwdFilterAlgoPerf, cudnnFilterDescriptor, cudnnConvolutionBackwardFilter,
             cudnnConvolutionBackwardBias

function cudnnConvolutionDescriptor(dtype::DataType; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1)
    dtype <: CUDNNFloat || error("GradValley: cudnnConvolutionDescriptor: dtype must be Float16, Float32 or Float64")
    conv_descriptor = cudnnConvolutionDescriptor(collect(Cint, padding), 
                        collect(Cint, stride), 
                        collect(Cint, dilation), 
                        CUDNN_CROSS_CORRELATION, 
                        cudnnDataType(dtype), 
                        cuDNN.math_mode(), 
                        CUDNN_DEFAULT_REORDER, 
                        Cint(groups))

    return conv_descriptor
end

function reshape_bias_for_cudnn(bias::DenseCuVector{T}) where {T <: CUDNNFloat}
    channels = length(bias)
    return reshape(bias, (1, 1, channels, 1))
end

function reshape_bias_for_cudnn_backward(bias::DenseCuArray{T, 4}) where {T <: CUDNNFloat}
    channels = length(bias)
    return reshape(bias, (channels, ))
end

function convolution2d!(output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}, bias::DenseCuVector{T}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    if cuDNN.version() < v"6"
        all(x -> x == 1, dilation) || error("GradValley: convolution2d!: only dilation = (1, 1) is supported in cuDNN version < 6")
    end
    conv_descriptor = cudnnConvolutionDescriptor(T, stride=stride, padding=padding, dilation=dilation, groups=groups)

    return cudnnConvolutionForward!(output, weight, input, conv_descriptor, bias=reshape_bias_for_cudnn(bias))
end

function convolution2d(input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}, bias::DenseCuVector{T}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    if cuDNN.version() < v"6"
        all(x -> x == 1, dilation) || error("GradValley: convolution2d: only dilation = (1, 1) is supported in cuDNN version < 6")
    end
    conv_descriptor = cudnnConvolutionDescriptor(T, stride=stride, padding=padding, dilation=dilation, groups=groups)

    return cudnnConvolutionForward(weight, input, conv_descriptor, bias=reshape_bias_for_cudnn(bias))
end

function convolution2d_data_backward!(input_gradient::DenseCuArray{T, 4}, output_gradient::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    if cuDNN.version() < v"6"
        all(x -> x == 1, dilation(cdims)) || error("GradValley: convolution2d_data_backward!: only dilation = (1, 1) is supported in cuDNN version < 6")
    end
    conv_descriptor = cudnnConvolutionDescriptor(T, stride=stride, padding=padding, dilation=dilation, groups=groups)

    alpha, beta = cuDNN.scalingParameter(T, 1), cuDNN.scalingParameter(T, 0)
    
    input_gradient_descriptor = cudnnTensorDescriptor(input_gradient)
    output_gradient_descriptor = cudnnTensorDescriptor(output_gradient)
    weight_descriptor = cudnnFilterDescriptor(weight)

    p = cudnnConvolutionBwdDataAlgoPerf(weight_descriptor, weight, output_gradient_descriptor, output_gradient, conv_descriptor, input_gradient_descriptor, input_gradient, beta!=0)
    cuDNN.with_workspace(p.memory) do workspace
        cudnnConvolutionBackwardData(cuDNN.handle(), alpha, weight_descriptor, weight, output_gradient_descriptor, output_gradient, conv_descriptor, p.algo, workspace, sizeof(workspace), beta, input_gradient_descriptor, input_gradient)
    end
    
    return input_gradient
end

function convolution2d_data_backward(output_gradient::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)
    input_width, input_height, in_channels, current_batch_size = size(input)
    # allocate the input_gradient with size of input before padding 
    # input_gradient = zeros(T, input_width, input_height, in_channels_weight * groups, current_batch_size)
    input_gradient = CUDA.zeros(T, input_width, input_height, in_channels, current_batch_size)

    return convolution2d_data_backward!(input_gradient, output_gradient, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function convolution2d_filter_backward!(weight_gradient::DenseCuArray{T, 4}, output_gradient::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    if cuDNN.version() < v"6"
        all(x -> x == 1, dilation(cdims)) || error("GradValley: convolution2d_filter_backward!: only dilation = (1, 1) is supported in cuDNN version < 6")
    end
    conv_descriptor = cudnnConvolutionDescriptor(T, stride=stride, padding=padding, dilation=dilation, groups=groups)

    alpha, beta = cuDNN.scalingParameter(T, 1), cuDNN.scalingParameter(T, 0)

    input_descriptor = cudnnTensorDescriptor(input)
    output_gradient_descriptor = cudnnTensorDescriptor(output_gradient)
    weight_gradient_descriptor = cudnnFilterDescriptor(weight_gradient)

    p = cudnnConvolutionBwdFilterAlgoPerf(input_descriptor, input, output_gradient_descriptor, output_gradient, conv_descriptor, weight_gradient_descriptor, weight_gradient, beta!=0);
    cuDNN.with_workspace(p.memory) do workspace
        cudnnConvolutionBackwardFilter(cuDNN.handle(), alpha, input_descriptor, input, output_gradient_descriptor, output_gradient, conv_descriptor, p.algo, workspace, sizeof(workspace), beta, weight_gradient_descriptor, weight_gradient);
    end

    return weight_gradient
end

function convolution2d_filter_backward(output_gradient::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)
    # allocate the input_gradient with size of input before padding 
    weight_gradient = CUDA.zeros(T, weight_width, weight_height, in_channels_weight, out_channels)

    return convolution2d_filter_backward!(weight_gradient, output_gradient, input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function convolution2d_bias_backward!(bias_gradient::DenseCuVector{T}, output_gradient::DenseCuArray{T, 4}) where {T <: CUDNNFloat}
    alpha, beta = cuDNN.scalingParameter(T, 1), cuDNN.scalingParameter(T, 0)
    bias_gradient = reshape_bias_for_cudnn(bias_gradient)
    bias_gradient_descriptor = cudnnTensorDescriptor(bias_gradient)
    output_gradient_descriptor = cudnnTensorDescriptor(output_gradient)
    cudnnConvolutionBackwardBias(cuDNN.handle(), alpha, output_gradient_descriptor, output_gradient, beta, bias_gradient_descriptor, bias_gradient)
    bias_gradient = reshape_bias_for_cudnn_backward(bias_gradient)

    return bias_gradient
end

function convolution2d_bias_backward(output_gradient::DenseCuArray{T, 4}) where {T <: CUDNNFloat}
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    bias_gradient = CUDA.zeros(T, out_channels)

    return convolution2d_bias_backward!(bias_gradient, output_gradient)
end

function deconvolution2d!(output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}, bias::DenseCuVector{T}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    output = convolution2d_data_backward!(output, input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # storing all the necessary shapes
    weight_width, weight_height, out_channels_weight, out_channels = size(weight)
    output_width, output_height, out_channels, current_batch_size = size(output)
    # adding bias if necessary
    output .= output .+ reshape_bias_for_cudnn(bias)

    return output
end

function deconvolution2d(input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}, bias::DenseCuVector{T}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), output_padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, in_channels, current_batch_size = size(input)
    weight_width, weight_height, out_channels_weight, in_channels = size(weight)
    # calculating shape of output
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + dilation[1] * (weight_height - 1) + output_padding[1] + 1
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + dilation[2] * (weight_width - 1) + output_padding[2] + 1

    output = CUDA.zeros(T, output_width, output_height, out_channels_weight * groups, current_batch_size)

    return deconvolution2d!(output, input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function deconvolution2d_data_backward!(input_gradient::DenseCuArray{T, 4}, output_gradient::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, in_channels, current_batch_size = size(input_gradient)
    input_gradient = convolution2d!(input_gradient, output_gradient, weight, CUDA.zeros(T, in_channels), stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    return input_gradient
end

function deconvolution2d_data_backward(output_gradient::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    input_width, input_height, in_channels, current_batch_size = size(input)
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)

    input_gradient = CUDA.zeros(T, input_width, input_height, in_channels, current_batch_size)

    return deconvolution2d_data_backward!(input_gradient, output_gradient, input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function deconvolution2d_filter_backward!(weight_gradient::DenseCuArray{T, 4}, output_gradient::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    weight_gradient = convolution2d_filter_backward!(weight_gradient, input, output_gradient, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)

    return weight_gradient
end

function deconvolution2d_filter_backward(output_gradient::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight::DenseCuArray{T, 4}; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1), groups::Int=1) where {T <: CUDNNFloat}
    # storing all the necessary shapes
    weight_width, weight_height, out_channels_weight, in_channels = size(weight)

    weight_gradient = CUDA.zeros(T, weight_width, weight_height, out_channels_weight, in_channels)

    return deconvolution2d_filter_backward!(weight_gradient, output_gradient, input, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)
end

function deconvolution2d_bias_backward!(bias_gradient::DenseCuVector{T}, output_gradient::DenseCuArray{T, 4}) where {T <: CUDNNFloat}
    return convolution2d_bias_backward!(bias_gradient, output_gradient)
end

function deconvolution2d_bias_backward(output_gradient::DenseCuArray{T, 4}) where {T <: CUDNNFloat}
    return convolution2d_bias_backward(output_gradient)
end