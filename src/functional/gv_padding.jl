#=
Padding-Operations
=#

@doc raw"""
    zero_pad_nd(input::AbstractArray{T, N}, padding::NTuple{2, Integer}) where {T, N}

Perform a padding-operation (nd => number of dimensions doesn't matter) as is usual for neural networks: equal padding one each "end" of an axis/dimension.

# Arguments
- `input::AbstractArray{T, N}`: of shape(d1, d2, ..., dn)
- `padding::NTuple{2, Integer}`: must be always a tuple of length of the number of dimensions of input: (pad-d1, pad-d2, ..., pad-dn)

Shape of returned output: (d1 + padding[1] * 2, d2 + padding[2] * 2, ..., dn + padding[n] * 2)
"""
function zero_pad_nd(input::AbstractArray{T, N}, padding::NTuple{2, Integer}) where {T <: Real, N}
    # calculate shape of output
    input_shape = size(input)
    output_shape = ()
    for (dim, in_dim_size) in enumerate(input_shape)
        out_dim_size = in_dim_size + padding[dim] * 2
        output_shape = tuplejoin(output_shape, (out_dim_size, ))
    end
    
    output = zeros(eltype(input), output_shape)

    # actual computation
    for position in CartesianIndices(input)
        output_positon = () 
        for (dim, index) in enumerate(Tuple(position))
            output_index = index + padding[dim]
            output_positon = tuplejoin(output_positon, (output_index, ))
        end
        output[CartesianIndex(output_positon)] = input[position]
    end

    return output
end

@doc raw"""
    zero_pad_nd(input::AbstractArray{T, 4}, padding::NTuple{2, Integer}) where {T}

Perform a padding-operation (2d => 4 dimensions, where the last 2 dimensions will be padded) as is usual for neural networks: equal padding one each "end" of the last two axis/dimension.

# Arguments
- `input::AbstractArray{T, 4}`: of shape(d1, d2, d3, d4)
- `padding::NTuple{2, Integer}`: must be always a tuple of length of the number of dimensions of input: (pad-d1, pad-d2, ..., pad-dn)

Shape of returned output: (d1, d2, d3 + padding[1] * 2, d4 + padding[2] * 2)
"""
function zero_pad_2d(input::AbstractArray{T, 4}, padding::NTuple{2, Integer}) where {T <: Real}
    current_batch_size, channels, height, width = size(input)
    y_padding, x_padding = padding
    output_height, output_width = height + 2 * y_padding, width + 2 * x_padding
    output = zeros(eltype(input), current_batch_size, channels, output_height, output_width)
    output[:, :, y_padding + 1:output_height - y_padding, x_padding + 1:output_width - x_padding] = input

    return output
end