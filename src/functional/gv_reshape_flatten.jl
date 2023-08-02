function reshape_forward(input::AbstractArray{T, N}, out_shape::NTuple{N2, Int}) where {T, N, N2}
    current_batch_size = size(input)[end]
    out_shape = tuplejoin(out_shape, (current_batch_size, ))
    output = reshape(input, out_shape)

    return output
end

function reshape_backward(output_gradient::AbstractArray{T, N}, input::AbstractArray{T, N2}) where {T, N, N2}
    in_shape = size(input)
    input_gradient = reshape(output_gradient, in_shape)

    return input_gradient
end