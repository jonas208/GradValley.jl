function fc_forward!(output::AbstractMatrix{T}, input::AbstractMatrix{T}, weight::AbstractMatrix{T}, bias::Vector{T}) where {T <: Real}
    #= GOOD VERSION 
    current_batch_size = size(input)[2]
    # in_features, out_features = size(weight)
    out_features, in_features = size(weight)
    typed_zero = zero(T)
    @turbo for index_batch in 1:current_batch_size
        for out_feature in 1:out_features
            output_value = typed_zero
            for in_feature in 1:in_features
                # output_value += input[in_feature, index_batch] * weight[in_feature, out_feature]
                output_value += input[in_feature, index_batch] * weight[out_feature, in_feature]
            end
            output[out_feature, index_batch] = output_value + bias[out_feature]
        end
    end
    =#
    #=
    current_batch_size = size(input)[2]
    for index_batch in 1:current_batch_size
        output[:, index_batch] = weight * input[:, index_batch] + bias
    end
    =#
    # output .= weight * input .+ bias
    output .= T(0) # might not be necessary!
    output = BLAS.gemm!('N', 'N', T(1), weight, input, T(0), output) .+ bias # T(1)

    return output
end

function fc_forward(input::AbstractMatrix{T}, weight::AbstractMatrix{T}, bias::Vector{T}) where {T <: Real}
    current_batch_size = size(input)[2]
    # in_features, out_features = size(weight)
    out_features, in_features = size(weight)
    output = zeros(T, out_features, current_batch_size)
    return fc_forward!(output, input, weight, bias)
end

function fc_backward!(input_gradient::AbstractMatrix{T}, weight_gradient::AbstractMatrix{T}, bias_gradient::AbstractVector{T}, output_gradient::AbstractMatrix{T}, input::AbstractMatrix{T}, weight::AbstractMatrix{T}) where {T <: Real}
    #= GOOD VERSION =#
    current_batch_size = size(input_gradient)[2]
    # in_features, out_features = size(weight_gradient)
    out_features, in_features = size(weight_gradient)

    typed_zero = zero(T)
    # because in the actual computation section, values are added, it's saver to reset the given weight_gradient first
    weight_gradient .= typed_zero
    @turbo for index_batch in 1:current_batch_size
        for in_feature in 1:in_features
            value = typed_zero
            for out_feature in 1:out_features
                # value += weight[in_feature, out_feature] * output_gradient[out_feature, index_batch]
                value += weight[out_feature, in_feature] * output_gradient[out_feature, index_batch]
                # weight_gradient[in_feature, out_feature] += input[in_feature, index_batch] * output_gradient[out_feature, index_batch]
                weight_gradient[out_feature, in_feature] += input[in_feature, index_batch] * output_gradient[out_feature, index_batch]
            end
            input_gradient[in_feature, index_batch] = value
        end
    end
    #=
    input_gradient .= T(0) # might not be necessary!
    weight_gradient .= T(0) # might not be necessary!
    input_gradient = BLAS.gemm!('T', 'N', T(1), weight, output_gradient, T(0), input_gradient)
    weight_gradient = BLAS.gemm!('N', 'T', T(1), output_gradient, input, T(0), weight_gradient)
    =#
    
    bias_gradient .= sum(output_gradient, dims=2)[:, 1]

    return input_gradient, weight_gradient, bias_gradient
end

function fc_backward(output_gradient::AbstractMatrix{T}, input::AbstractMatrix{T}, weight::AbstractMatrix{T}) where {T <: Real}
    # in_features, out_features = size(weight)
    out_features, in_features = size(weight)
    input_gradient = similar(input)
    weight_gradient = similar(weight)
    bias_gradient = zeros(T, out_features)

    return fc_backward!(input_gradient, weight_gradient, bias_gradient, output_gradient, input, weight)
end