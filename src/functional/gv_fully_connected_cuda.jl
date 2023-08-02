function fc_forward!(output::DenseCuMatrix{T}, input::DenseCuMatrix{T}, weight::DenseCuMatrix{T}, bias::DenseCuVector{T}) where {T <: CUDNNFloat}
    current_batch_size = size(input)[2]
    for index_batch in 1:current_batch_size
        output[:, index_batch] = weight * input[:, index_batch] + bias
    end

    return output
end

function fc_forward(input::DenseCuMatrix{T}, weight::DenseCuMatrix{T}, bias::DenseCuVector{T}) where {T <: CUDNNFloat}
    current_batch_size = size(input)[2]
    # in_features, out_features = size(weight)
    out_features, in_features = size(weight)
    output = CUDA.zeros(T, out_features, current_batch_size)
    return fc_forward!(output, input, weight, bias)
end

function fc_backward!(input_gradient::DenseCuMatrix{T}, weight_gradient::DenseCuMatrix{T}, bias_gradient::DenseCuVector{T}, output_gradient::DenseCuMatrix{T}, input::DenseCuMatrix{T}, weight::DenseCuMatrix{T}) where {T <: CUDNNFloat}
    current_batch_size = size(input_gradient)[2]
    # in_features, out_features = size(weight_gradient)
    out_features, in_features = size(weight_gradient)

    # because in the actual computation section, values are added, it's saver to reset the given weight_gradient first
    weight_gradient .= zero(T)

    weight_transposed = weight'
    for index_batch in 1:current_batch_size
        input_gradient[:, index_batch] = weight_transposed * output_gradient[:, index_batch]

        single_output_gradient_vector = output_gradient[:, index_batch]
        single_output_gradient = reshape(single_output_gradient_vector, out_features, 1)
        weight_gradient += single_output_gradient * input[:, index_batch]'
    end

    bias_gradient .= sum(output_gradient, dims=2)[:, 1]

    return input_gradient, weight_gradient, bias_gradient
end

function fc_backward(output_gradient::DenseCuMatrix{T}, input::DenseCuMatrix{T}, weight::DenseCuMatrix{T}) where {T <: CUDNNFloat}
    # in_features, out_features = size(weight)
    out_features, in_features = size(weight)
    input_gradient = CUDA.similar(input)
    weight_gradient = CUDA.similar(weight)
    bias_gradient = CUDA.zeros(T, out_features)

    return fc_backward!(input_gradient, weight_gradient, bias_gradient, output_gradient, input, weight)
end