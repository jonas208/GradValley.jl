module Functional
using LoopVectorization
using LinearAlgebra
using CUDA
using cuDNN

const CUDNNFloat = Union{Float16, Float32, Float64}

#=
General convention:
in_channels, out_channels: indices for the respective channel types
in_channel, out_channel: numbers for the respective channel types
y_w, x_w are indexes used for weights/kernels (and their gradients)
x_in, y_in are indexes used for inputs
y_out, x_out are indexes used for outputs
m, n are indexes which were calculated by get_input_position()
=#

#=
Internal functions, Internals
=#

# returns the position in an input matrix given by the position in output (e.g. usefull for conv, pool and diffrent	backward-passes)
# output_position and stride must be tuples
function get_input_position(output_position::NTuple{2, Int}, stride::NTuple{2, Int})
    m = output_position[1] + (stride[1] - 1) * (output_position[1] - 1)
    n = output_position[2] + (stride[2] - 1) * (output_position[2] - 1)

    return m, n
end

# returns the shape of an output from a pooling- or convolution-operation
function calculate_output_shape(input_height::Int, input_width::Int, weight_height::Int, weight_width::Int; stride::NTuple{2, Int}=(1, 1), padding::NTuple{2, Int}=(0, 0), dilation::NTuple{2, Int}=(1, 1))
    output_height = (input_height + 2 * padding[1] - dilation[1] * (weight_height - 1) - 1) / stride[1] + 1
    output_width = (input_width + 2 * padding[2] - dilation[2] * (weight_width - 1) - 1) / stride[2] + 1

    output_height = convert(Int, trunc(output_height))
    output_width = convert(Int, trunc(output_width))

    return output_height, output_width
end

# combine many tuples
tuplejoin(t1::Tuple, t2::Tuple, t3...) = tuplejoin((t1..., t2...), t3...)
tuplejoin(t::Tuple) = t

# creates an array of iterators (UnitRanges) for each dimension without the itertor for the given dimension (dim), the given size_tuple is the size of the orginal array, e.g. useful for softmax & backward_softmax
function get_iters_without_at_dim(size_tuple, dim)
    # checks if dim is a valid value
    num_dims = length(size_tuple)
    if dim == 0 || abs(dim) > num_dims
        error("GradValley: dim_sum: the given dim is out of bounce")
    end
    if dim < 0
        dim = num_dims + 1 - dim
    end
    iterators = UnitRange{Int}[]
    for (index, dim_size) in enumerate(size_tuple)
        if index != dim
            push!(iterators, 1:dim_size)
        end
    end

    return iterators
end

# internal function for adaptive pooling, returns a list of ranges containing the indices for reading the input array with the correct pooling kernels/windows
function get_in_indices(in_len, out_len)
    get_start_index(a, b, c) = floor((a * c) / b)
    get_end_index(a, b, c) = ceil(((a + 1) * c) / b)
    # indices = UnitRange{Int}[]
    indices = Vector{UnitRange{Int}}(undef, out_len)
    for index_out in 0:out_len - 1
        start_index = get_start_index(index_out, out_len, in_len)
        end_index = get_end_index(index_out, out_len, in_len)
        difference = end_index - start_index
        range_start = convert(Int, trunc(start_index + 1))
        range_end = convert(Int, trunc(start_index + difference)) # + 1
        indices_range = range_start:range_end
        # push!(indices, indices_range)
        indices[index_out + 1] = indices_range
    end

    return indices
end

# include all necessary files (categorized in different tasks like convolution and pooling)
include("functional/gv_padding.jl")
include("functional/gv_padding_cuda.jl")
include("functional/gv_convolution.jl")
include("functional/gv_convolution_cuda.jl")
include("functional/gv_pooling.jl")
include("functional/gv_pooling_cuda.jl")
include("functional/gv_batch_normalization.jl")
include("functional/gv_batch_normalization_cuda.jl")
include("functional/gv_fully_connected.jl")
include("functional/gv_fully_connected_cuda.jl")
include("functional/gv_reshape_flatten.jl")
include("functional/gv_activation_functions.jl")
include("functional/gv_activation_functions_cuda.jl")
include("functional/gv_weight_initialization.jl")

end # end of module "Functional"