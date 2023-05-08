#=
Activation functions: Forward & Backward (their derivatives)
All functions take an array (nd, number of dimensions doesn't matter) and return a new array with the modified values
(The given input array will not be modified)
The prefix "d_" stands for the derivative (of the activation function)
=#

# calculate the element-wise relu activation 
function relu(inputs::AbstractArray)
    # outputs = max.(0, inputs)
    outputs = similar(inputs)
    typed_zero = zero(eltype(inputs))
    for index in eachindex(outputs)
        value = inputs[index]
        if value < 0
            outputs[index] = typed_zero
        else
            outputs[index] = value
        end
    end

    return outputs
end

# calculate the element-wise relu activation 
function relu(outputs::AbstractArray, inputs::AbstractArray)
    typed_zero = zero(eltype(outputs))
    for index in eachindex(outputs)
        value = inputs[index]
        if value < 0
            outputs[index] = typed_zero
        else
            outputs[index] = value
        end
    end

    return outputs
end

# calculate the element-wise derivative of relu activation 
function d_relu(inputs::AbstractArray)
    outputs = similar(inputs)
    typed_zero = zero(eltype(inputs))
    typed_one = one(eltype(inputs))
    for index in eachindex(outputs)
        value = inputs[index]
        if value < 0
            outputs[index] = typed_zero
        else
            outputs[index] = typed_one
        end
    end

    return outputs
end

# calculate the element-wise derivative of relu activation 
function d_relu(outputs::AbstractArray, inputs::AbstractArray)
    typed_zero = zero(eltype(outputs))
    typed_one = one(eltype(outputs))
    for index in eachindex(outputs)
        value = inputs[index]
        if value < 0
            outputs[index] = typed_zero
        else
            outputs[index] = typed_one
        end
    end

    return outputs
end

# calculate the element-wise sigmoid activation 
function sigmoid(inputs::AbstractArray)
    # sig(x) = 1 / (1 + exp(-x))
    # outputs = map(sig, inputs)
    outputs = similar(inputs)
    for index in eachindex(outputs)
        value = inputs[index]
        outputs[index] = 1 / (1 + exp(-value))
    end

    return outputs
end

# calculate the element-wise sigmoid activation 
function sigmoid(outputs::AbstractArray, inputs::AbstractArray)
    for index in eachindex(outputs)
        value = inputs[index]
        outputs[index] = 1 / (1 + exp(-value))
    end

    return outputs
end

# calculate the element-wise derivative of sigmoid activation 
function d_sigmoid(inputs::AbstractArray)
    outputs = similar(inputs)
    sig(x) = 1 / (1 + exp(-x))
    for index in eachindex(outputs)
        value = inputs[index]
        outputs[index] = sig(value) * (1 - sig(value))
    end

    return outputs
end

# calculate the element-wise derivative of sigmoid activation 
function d_sigmoid(outputs::AbstractArray, inputs::AbstractArray)
    sig(x) = 1 / (1 + exp(-x))
    for index in eachindex(outputs)
        value = inputs[index]
        outputs[index] = sig(value) * (1 - sig(value))
    end

    return outputs
end

# calculate the element-wise derivative of tanh activation 
function gv_tanh(inputs::AbstractArray)
    #=
    outputs = similar(inputs)
    for index in eachindex(outputs)
        value = inputs[index]
        outputs[index] = tanh(value)
    end
    =#
    outputs = tanh.(inputs)

    return outputs
end

# calculate the element-wise derivative of tanh activation 
function gv_tanh(outputs::AbstractArray, inputs::AbstractArray)
    for index in eachindex(outputs)
        value = inputs[index]
        outputs[index] = tanh(value)
    end

    return outputs
end

# calculate the element-wise derivative of tanh activation 
function d_tanh(inputs::AbstractArray)
    outputs = similar(inputs)
    for index in eachindex(outputs)
        value = inputs[index]
        outputs[index] = 1 - tanh(value)^2
    end

    return outputs
end

# calculate the element-wise derivative of tanh activation 
function d_tanh(outputs::AbstractArray, inputs::AbstractArray)
    for index in eachindex(outputs)
        value = inputs[index]
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