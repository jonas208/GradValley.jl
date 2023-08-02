using cuDNN: CUDNN_BN_MIN_EPSILON, CUDNN_BATCHNORM_SPATIAL, scalingParameter, cudnnNormalizationForward!, cudnnNormalizationForward, cudnnBatchNormalizationBackward

reshape_channel_vector_for_cudnn(channel_vector::DenseCuVector{T}) where {T <: CUDNNFloat} = reshape_bias_for_cudnn(channel_vector)
reshape_channel_vector_for_cudnn_backward(channel_array::DenseCuArray{T, 4}) where {T <: CUDNNFloat} = reshape_bias_for_cudnn_backward(channel_array)

function batch_norm2d_forward!(output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight_gamma::DenseCuVector{T}, weight_beta::DenseCuVector{T}, track_running_stats::Bool, running_mean::DenseCuVector{T}, running_variance::DenseCuVector{T}, test_mode::Bool; momentum::T=T(0.1), epsilon::T=T(1e-5)) where {T <: CUDNNFloat}
    running_mean_new, running_variance_new = copy(running_mean), copy(running_variance)
    if track_running_stats
        return cudnnNormalizationForward!(output, 
                                          input, 
                                          reshape_channel_vector_for_cudnn(running_mean_new), 
                                          reshape_channel_vector_for_cudnn(running_variance_new), 
                                          reshape_channel_vector_for_cudnn(weight_beta), 
                                          reshape_channel_vector_for_cudnn(weight_gamma), 
                                          training=!test_mode, 
                                          exponentialAverageFactor=momentum, 
                                          epsilon=epsilon), running_mean_new, running_variance_new
    else
        return cudnnNormalizationForward!(output, 
                                          input, 
                                          reshape_channel_vector_for_cudnn(running_mean_new), 
                                          reshape_channel_vector_for_cudnn(running_variance_new), 
                                          reshape_channel_vector_for_cudnn(weight_beta), 
                                          reshape_channel_vector_for_cudnn(weight_gamma), 
                                          training=true, 
                                          exponentialAverageFactor=momentum, 
                                          epsilon=epsilon)[1], running_mean, running_variance
    end
end

function batch_norm2d_forward(input::DenseCuArray{T, 4}, weight_gamma::DenseCuVector{T}, weight_beta::DenseCuVector{T}, track_running_stats::Bool, running_mean::DenseCuVector{T}, running_variance::DenseCuVector{T}, test_mode::Bool; momentum::T=T(0.1), epsilon::T=T(1e-5)) where {T <: CUDNNFloat}
    running_mean_new, running_variance_new = copy(running_mean), copy(running_variance)
    if track_running_stats
        return cudnnNormalizationForward(input, 
                                         reshape_channel_vector_for_cudnn(running_mean_new), 
                                         reshape_channel_vector_for_cudnn(running_variance_new), 
                                         reshape_channel_vector_for_cudnn(weight_beta), 
                                         reshape_channel_vector_for_cudnn(weight_gamma), 
                                         training=!test_mode, 
                                         exponentialAverageFactor=momentum, 
                                         epsilon=epsilon), running_mean_new, running_variance_new
    else
        return cudnnNormalizationForward(input, 
                                         reshape_channel_vector_for_cudnn(running_mean_new), 
                                         reshape_channel_vector_for_cudnn(running_variance_new), 
                                         reshape_channel_vector_for_cudnn(weight_beta), 
                                         reshape_channel_vector_for_cudnn(weight_gamma), 
                                         training=true, 
                                         exponentialAverageFactor=momentum, 
                                         epsilon=epsilon), running_mean, running_variance
    end
end

function batch_norm2d_data_backward!(input_gradient::DenseCuArray{T, 4}, output_gradient::DenseCuArray{T, 4}, output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight_gamma::DenseCuVector{T}, weight_beta::DenseCuVector{T}, track_running_stats::Bool, running_mean::Union{DenseCuVector{T}, Nothing}, running_variance::Union{DenseCuVector{T}, Nothing}, test_mode::Bool; epsilon::T=T(1e-5)) where {T <: CUDNNFloat}
    weight_beta = reshape_channel_vector_for_cudnn(weight_beta)
    weight_gamma = reshape_channel_vector_for_cudnn(weight_gamma)

    input_descriptor = cudnnTensorDescriptor(input)
    input_gradient_descriptor = cudnnTensorDescriptor(input_gradient)
    output_gradient_descriptor = cudnnTensorDescriptor(output_gradient)
    weight_gamma_descriptor = cudnnTensorDescriptor(weight_gamma)

    @warn "GradValley: batch_norm2d_data_backward!: batch_norm2d_data_backward! was called, but cuDNN does not distinguish between data backward and weight backward, so all gradients are calculated. However, only input_gradient will be overwritten and returned. Save some computing time and call the combined version batch_norm2d_backward! directly."
    weight_gamma_gradient = CUDA.similar(weight_gamma)
    weight_beta_gradient = CUDA.similar(weight_beta)

    if epsilon < CUDNN_BN_MIN_EPSILON
        error("GradValley: batch_norm2d_data_backward!: epsilon = $epsilon is too small for cuDNN, the minimum allowed value for epsilon is $CUDNN_BN_MIN_EPSILON")
    end

    if running_mean === nothing || running_variance === nothing
        running_mean !== running_variance && error("GradValley: batch_norm2d_data_backward! both or neither of running_mean and running_variance must be nothing")
        running_mean = CU_NULL
        running_variance = CU_NULL
    else
        if test_mode && track_running_stats
            running_mean = reshape_channel_vector_for_cudnn(running_mean)
            running_variance = reshape_channel_vector_for_cudnn(running_variance)
        else
            running_mean = CU_NULL
            running_variance = CU_NULL
        end
    end

    cudnnBatchNormalizationBackward(cuDNN.handle(),
                                    CUDNN_BATCHNORM_SPATIAL,
                                    scalingParameter(T, 1), 
                                    scalingParameter(T, 0), 
                                    scalingParameter(T, 1), 
                                    scalingParameter(T, 0),
                                    input_descriptor, input, 
                                    output_gradient_descriptor, output_gradient, 
                                    input_gradient_descriptor, input_gradient, 
                                    weight_gamma_descriptor, weight_gamma, weight_gamma_gradient, weight_beta_gradient, 
                                    epsilon, 
                                    running_mean, 
                                    running_variance)

    return input_gradient
end

function batch_norm2d_data_backward(output_gradient::DenseCuArray{T, 4}, output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight_gamma::DenseCuVector{T}, weight_beta::DenseCuVector{T}, track_running_stats::Bool, running_mean::Union{DenseCuVector{T}, Nothing}, running_variance::Union{DenseCuVector{T}, Nothing}, test_mode::Bool; epsilon::T=T(1e-5)) where {T <: CUDNNFloat}
    input_gradient = CUDA.similar(output_gradient)
    return batch_norm2d_data_backward!(input_gradient, output_gradient, output, input, weight_gamma, weight_beta, track_running_stats, running_mean, running_variance, test_mode, epsilon=epsilon)
end

function batch_norm2d_weight_backward!(weight_gamma_gradient::DenseCuVector{T}, weight_beta_gradient::DenseCuVector{T}, output_gradient::DenseCuArray{T, 4}, output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight_gamma::DenseCuVector{T}, weight_beta::DenseCuVector{T}, track_running_stats::Bool, running_mean::Union{DenseCuVector{T}, Nothing}, running_variance::Union{DenseCuVector{T}, Nothing}, test_mode::Bool; epsilon::T=T(1e-5)) where {T <: CUDNNFloat}
    weight_beta = reshape_channel_vector_for_cudnn(weight_beta)
    weight_gamma = reshape_channel_vector_for_cudnn(weight_gamma)
    weight_gamma_gradient = reshape_channel_vector_for_cudnn(weight_gamma_gradient)
    weight_beta_gradient = reshape_channel_vector_for_cudnn(weight_beta_gradient)

    input_descriptor = cudnnTensorDescriptor(input)
    input_gradient_descriptor = cudnnTensorDescriptor(input)
    output_gradient_descriptor = cudnnTensorDescriptor(output_gradient)
    weight_gamma_descriptor = cudnnTensorDescriptor(weight_gamma)

    @warn "GradValley: batch_norm2d_weight_backward!: batch_norm2d_weight_backward! was called, but cuDNN does not distinguish between data backward and weight backward, so all gradients are calculated. However, only weight_gamma_gradient and weight_beta_gradient will be overwritten and returned. Save some computing time and call the combined version batch_norm2d_backward! directly."
    input_gradient = CUDA.similar(input)

    if epsilon < CUDNN_BN_MIN_EPSILON
        error("GradValley: batch_norm2d_weight_backward!: epsilon = $epsilon is too small for cuDNN, the minimum allowed value for epsilon is $CUDNN_BN_MIN_EPSILON")
    end

    if running_mean === nothing || running_variance === nothing
        running_mean !== running_variance && error("GradValley: batch_norm2d_weight_backward! both or neither of running_mean and running_variance must be nothing")
        running_mean = CU_NULL
        running_variance = CU_NULL
    else
        if test_mode && track_running_stats
            running_mean = reshape_channel_vector_for_cudnn(running_mean)
            running_variance = reshape_channel_vector_for_cudnn(running_variance)
        else
            running_mean = CU_NULL
            running_variance = CU_NULL
        end
    end

    cudnnBatchNormalizationBackward(cuDNN.handle(),
                                    CUDNN_BATCHNORM_SPATIAL,
                                    scalingParameter(T, 1), 
                                    scalingParameter(T, 0), 
                                    scalingParameter(T, 1), 
                                    scalingParameter(T, 0),
                                    input_descriptor, input, 
                                    output_gradient_descriptor, output_gradient, 
                                    input_gradient_descriptor, input_gradient, 
                                    weight_gamma_descriptor, weight_gamma, weight_gamma_gradient, weight_beta_gradient, 
                                    epsilon, 
                                    running_mean, 
                                    running_variance)

    return reshape_channel_vector_for_cudnn_backward(weight_gamma_gradient), reshape_channel_vector_for_cudnn_backward(weight_beta_gradient)
end

function batch_norm2d_weight_backward(output_gradient::DenseCuArray{T, 4}, output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight_gamma::DenseCuVector{T}, weight_beta::DenseCuVector{T}, track_running_stats::Bool, running_mean::Union{DenseCuVector{T}, Nothing}, running_variance::Union{DenseCuVector{T}, Nothing}, test_mode::Bool; epsilon::T=T(1e-5)) where {T <: CUDNNFloat}
    weight_gamma_gradient = CUDA.similar(weight_gamma)
    weight_beta_gradient = CUDA.similar(weight_beta)
    return batch_norm2d_weight_backward!(weight_gamma_gradient, weight_beta_gradient, output_gradient, output, input, weight_gamma, weight_beta, track_running_stats, running_mean, running_variance, test_mode, epsilon=epsilon)
end

function batch_norm2d_backward!(input_gradient::DenseCuArray{T, 4}, weight_gamma_gradient::DenseCuVector{T}, weight_beta_gradient::DenseCuVector{T}, output_gradient::DenseCuArray{T, 4}, output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight_gamma::DenseCuVector{T}, weight_beta::DenseCuVector{T}, track_running_stats::Bool, running_mean::Union{DenseCuVector{T}, Nothing}, running_variance::Union{DenseCuVector{T}, Nothing}, test_mode::Bool; epsilon::T=T(1e-5)) where {T <: CUDNNFloat}
    weight_beta = reshape_channel_vector_for_cudnn(weight_beta)
    weight_gamma = reshape_channel_vector_for_cudnn(weight_gamma)
    weight_gamma_gradient = reshape_channel_vector_for_cudnn(weight_gamma_gradient)
    weight_beta_gradient = reshape_channel_vector_for_cudnn(weight_beta_gradient)

    input_descriptor = cudnnTensorDescriptor(input)
    input_gradient_descriptor = cudnnTensorDescriptor(input_gradient)
    output_gradient_descriptor = cudnnTensorDescriptor(output_gradient)
    weight_gamma_descriptor = cudnnTensorDescriptor(weight_gamma)

    if epsilon < CUDNN_BN_MIN_EPSILON
        error("GradValley: batch_norm2d_backward!: epsilon = $epsilon is too small for cuDNN, the minimum allowed value for epsilon is $CUDNN_BN_MIN_EPSILON")
    end

    if running_mean === nothing || running_variance === nothing
        running_mean !== running_variance && error("GradValley: batch_norm2d_weight_backward! both or neither of running_mean and running_variance must be nothing")
        running_mean = CU_NULL
        running_variance = CU_NULL
    else
        if test_mode && track_running_stats
            running_mean = reshape_channel_vector_for_cudnn(running_mean)
            running_variance = reshape_channel_vector_for_cudnn(running_variance)
        else
            running_mean = CU_NULL
            running_variance = CU_NULL
        end
    end

    cudnnBatchNormalizationBackward(cuDNN.handle(),
                                    CUDNN_BATCHNORM_SPATIAL,
                                    scalingParameter(T, 1), 
                                    scalingParameter(T, 0), 
                                    scalingParameter(T, 1), 
                                    scalingParameter(T, 0),
                                    input_descriptor, input, 
                                    output_gradient_descriptor, output_gradient, 
                                    input_gradient_descriptor, input_gradient, 
                                    weight_gamma_descriptor, weight_gamma, weight_gamma_gradient, weight_beta_gradient, 
                                    epsilon, 
                                    running_mean, 
                                    running_variance)
    
    return input_gradient, reshape_channel_vector_for_cudnn_backward(weight_gamma_gradient), reshape_channel_vector_for_cudnn_backward(weight_beta_gradient)
end

function batch_norm2d_backward(output_gradient::DenseCuArray{T, 4}, output::DenseCuArray{T, 4}, input::DenseCuArray{T, 4}, weight_gamma::DenseCuVector{T}, weight_beta::DenseCuVector{T}, track_running_stats::Bool, running_mean::Union{DenseCuVector{T}, Nothing}, running_variance::Union{DenseCuVector{T}, Nothing}, test_mode::Bool; epsilon::T=T(1e-5)) where {T <: CUDNNFloat}
    input_gradient = CUDA.similar(output_gradient)
    weight_gamma_gradient = CUDA.similar(weight_gamma)
    weight_beta_gradient = CUDA.similar(weight_beta)
    return batch_norm2d_backward!(input_gradient, weight_gamma_gradient, weight_beta_gradient, output_gradient, output, input, weight_gamma, weight_beta, track_running_stats, running_mean, running_variance, test_mode, epsilon=epsilon)
end