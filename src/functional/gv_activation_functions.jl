#=
Activation functions: Forward & Backward (their derivatives)
All functions take an array (nd, number of dimensions doesn't matter) and return a new array with the modified values
(The given array will not be modified)
The prefix "d_" stands for the derivative (of the activation function)
=#

# appplies a element-wise relu activation on a copy of inputs
function relu(inputs::Array)
    outputs = copy(inputs)
    # for (index, value) in enumerate(outputs)
    for index in eachindex(outputs)
        value = outputs[index]
        if value < 0
            outputs[index] = 0
        end
    end
    # outputs = max.(0, inputs)

    return outputs
end

# appplies the element-wise derivative of relu activation on a copy of inputs
function d_relu(inputs::Array)
    outputs = copy(inputs)
    # for (index, value) in enumerate(outputs)
    for index in eachindex(outputs)
        value = outputs[index]
        if value < 0
            outputs[index] = 0
        else
            outputs[index] = 1
        end
    end

    return outputs
end

# appplies a element-wise sigmoid activation on a copy of inputs
function sigmoid(inputs::Array)
    outputs = copy(inputs)
    # for (index, value) in enumerate(outputs)
    # sig(x) = 1 / (1 + exp(-x))
    # outputs = map(sig, outputs)
    for index in eachindex(outputs)
        value = outputs[index]
        outputs[index] = 1 / (1 + exp(-value))
    end

    return outputs
end

# appplies the element-wise derivative of sigmoid activation on a copy of inputs
function d_sigmoid(inputs::Array)
    outputs = copy(inputs)
    sig(x) = 1 / (1 + exp(-x))
    # for (index, value) in enumerate(outputs)
    for index in eachindex(outputs)
        value = outputs[index]
        outputs[index] = sig(value) * (1 - sig(value))
    end

    return outputs
end

# appplies the element-wise derivative of tanh activation on a copy of inputs
function gv_tanh(inputs::Array)
    #=
    outputs = copy(inputs)
    for index in eachindex(outputs)
        value = outputs[index]
        outputs[index] = tanh(value)
    end
    =#
    outputs = tanh.(inputs)

    return outputs
end

# DEPRECTED: there is no further tanh function because Julia already has a built-in tanh function
# appplies the element-wise derivative of tanh activation on a copy of inputs
function d_tanh(inputs::Array)
    outputs = copy(inputs)
    # for (index, value) in enumerate(outputs)
    for index in eachindex(outputs)
        value = outputs[index]
        outputs[index] = 1 - tanh(value)^2
    end

    return outputs
end

#=
Softmax along a specific dimension (dim): Forward & Backward
=#

# computes the softmax along a specific dimension
function softmax(input; dim=1)
    #=
    output = copy(input)
    exps_sum = dim_sum(exp.(input); dim=dim)
    dim_size = size(input)[dim]
    indices_array = Union{UnitRange{Int}, Int}[1:dim_size for dim_size in size(input)]
    for index_dim in 1:dim_size
        indices_array[dim] = index_dim
        output[indices_array...] = exp.(input[indices_array...]) ./ exps_sum
    end
    # output = exp.(input) ./ exps_sum # only works for matrices (exp.(input)) followed by a vector (exps_sum)
    =#

    # truly iterates over all slices (vectors)

    output = zeros(size(input))
    input_size = size(input)
    num_dims = length(input_size)
    # checks if dim is a valid value
    if dim == 0 || abs(dim) > num_dims
        error("GradValley: dim_sum: the given dim is out of bounce")
    end
    if dim < 0
        dim = num_dims + 1 - dim
    end
    iterators = get_iters_without_at_dim(input_size, dim)
    for indices_tuple in Base.product(iterators...)
        indices_array = collect(Union{UnitRange{Int}, Int}, indices_tuple)
        # insert!(indices_array, dim, :)
        indices_array = insert!(indices_array, dim, 1:input_size[dim])
        input_array_slice = input[indices_array...] # input_array_slice is always a vector
        exps_sum = sum(exp.(input_array_slice))
        output[indices_array...] = exp.(input_array_slice) ./ exps_sum
    end


    return output
end

# Functions used for Backpropagation (Softmax)
# The only input each function takes is an instance of a softmax layer struct (Softmax)
# Because a layer is given, these functions directly work on the hole batch

# computes the derivative of softmax activation on the given layer
function softmax_backward(softmax_layer)
    out_losses = softmax_layer.losses
    softmax_output = softmax_layer.outputs
    dim = softmax_layer.dim
    out_losses_size = size(out_losses)
    dim_size = out_losses_size[dim]
    losses = zeros(eltype(out_losses), out_losses_size)
    iterators = get_iters_without_at_dim(out_losses_size, dim)
    for indices_tuple in Base.product(iterators...)
        indices_array = collect(Union{UnitRange{Int}, Int}, indices_tuple)
        insert!(indices_array, dim, 1:dim_size)

        softmax_output_array_slice = softmax_output[indices_array...] # is always a vector
        n = length(softmax_output_array_slice)
        replicated_softmax_output = zeros(n, n)
        for x in 1:n
            replicated_softmax_output[:, x] = softmax_output_array_slice
        end
        identity = Matrix(1.0I, n, n) # Identity matrix of Float64 type
        jacobian_matrix = (replicated_softmax_output .* (identity - transpose(replicated_softmax_output)))

        losses[indices_array...] = jacobian_matrix * out_losses[indices_array...]
    end

    return losses
end