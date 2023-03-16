#=
Padding-Operations
=#

# Performing a padding-operation (nd, number of dimensions doesn't matter) as is usual with neural networks: equal padding one each "end" of an axis/dimension
# Shape of input: (d1, d2, ..., dn)
# padding must always be a tuple with length of the number of dimensions of input: (d1, d2, ..., dn)
# Shape of output: (d1 + padding[0] * 2, d2 + padding[2] * 2, ..., dn + padding[n] * 2)
function zero_pad_nd(input::Array, padding::Tuple)
    # calculate shape of output
    input_shape = size(input)
    output_shape = ()
    for (dim, in_dim_size) in enumerate(input_shape)
        out_dim_size = in_dim_size + padding[dim] * 2
        output_shape = tuplejoin(output_shape, (out_dim_size, ))
    end
    
    output = zeros(output_shape)

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

function zero_pad_2d(input::AbstractArray, padding::Tuple{Integer, Integer})
    current_batch_size, channels, height, width = size(input)
    y_padding, x_padding = padding
    output_height, output_width = height + 2 * y_padding, width + 2 * x_padding
    output = zeros(eltype(input), current_batch_size, channels, output_height, output_width)
    output[:, :, y_padding + 1:output_height - y_padding, x_padding + 1:output_width - x_padding] = input

    return output
end