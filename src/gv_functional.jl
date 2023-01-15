module Functional
using LoopVectorization
using LinearAlgebra

#=
General convention:
in_channels, out_channels: indices for the respective channel types
in_channel, out_channel: numbers for the respective channel types
y_w, x_w are indexes used for kernels/weights (and their gradients)
x_in, y_in are indexes used for inputs
y_out, x_out are indexes used for outputs
m, n are indexes which were calculated by get_input_position()
=#

#=
Internal functions, Internals
=#

# returns the position in an input matrix given by the position in output (e.g. usefull for conv, pool and diffrent	backward-passes)
# output_position and stride must be tuples
function get_input_position(output_position::Tuple, stride::Tuple)
    m = output_position[1] + (stride[1] - 1) * (output_position[1] - 1)
    n = output_position[2] + (stride[2] - 1) * (output_position[2] - 1)

    return m, n
end

# returns the shape of an output from a pooling- or convolution-operation
function calculate_output_shape(input_height::Int, input_width::Int, kernel_height::Int, kernel_width::Int; stride::Tuple=(1, 1), padding::Tuple=(0, 0), dilation::Tuple=(1, 1)) # dilation::Tuple{Int, Int}
    output_height = (input_height + 2 * padding[1] - dilation[1] * (kernel_height - 1) - 1) / stride[1] + 1
    output_width = (input_width + 2 * padding[2] - dilation[2] * (kernel_width - 1) - 1) / stride[2] + 1

    output_height = convert(Int, trunc(output_height))
    output_width = convert(Int, trunc(output_width))

    return output_height, output_width
end

# combine many tuples
tuplejoin(t1::Tuple, t2::Tuple, t3...) = tuplejoin((t1..., t2...), t3...)
# @inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)
tuplejoin(t::Tuple) = t

# creates an array of iterators (UnitRanges) for each dimension without the itertor for the given dimension (dim), the given size_tuple is the size of the orginal array, e.g. useful for dim_sum or softmax & backward_softmax
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

# calculates the sum along a specific dim (removes this dimension in the output),
# negative dim starts counting at the end of all dimensions, so dim=-1 for example is the last dimension
function dim_sum(input; dim=1)
    input_size = size(input)
    num_dims = length(input_size)
    # checks if dim is a valid value
    if dim == 0 || abs(dim) > num_dims
        error("GradValley: dim_sum: the given dim is out of bounce")
    end
    if dim < 0
        dim = num_dims + 1 - dim
    end
    dim_size = input_size[dim]
    output_size = Int[]
    for (dim_index, dim_size) in enumerate(input_size)
        if dim_index != dim
            push!(output_size, dim_size)
        end
    end
    output_size = Tuple(output_size)
    # println(output_size)
    output = zeros(eltype(input), output_size)
    indices_array = Union{UnitRange{Int}, Int}[1:dim_size for dim_size in input_size]
    for index_dim in 1:dim_size
        indices_array[dim] = index_dim
        output .+= input[indices_array...]
    end
    
    return output
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
include("functional/gv_convolution.jl")
include("functional/gv_pooling.jl")
include("functional/gv_fully_connected.jl")
include("functional/gv_reshape_flatten.jl")
include("functional/gv_batch_normalization.jl")
include("functional/gv_activation_functions.jl")
include("functional/gv_weight_initialization.jl")

end # end of module "Functional"