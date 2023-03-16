# custom rules for the layers
import Base: +, -, *, /, ^

#=
function forward(sc::SequentialContainer, inputs::TrackedArray)
    tracked_args = (sc, inputs)
    outputs, pullback = rrule(forward, sc, inputs.primal)
    outputs_tracked = IntermediateTracked(outputs, tracked_args, pullback)
    return outputs_tracked
end

function ChainRules.rrule(::typeof(forward), sc::SequentialContainer, inputs::AbstractArray)
    # doing the forward pass of the sc 
    outputs = forward(sc, inputs)
    function forward_pullback(derivative_loss::AbstractArray)
        # doing the backpropagation of the sc
        backward(sc, derivative_loss)
        # defining gradients
        forward_gradient = NoTangent()
        sc_gradient = NoTangent()
        inputs_gradient = sc.previous_losses

        return forward_gradient, sc_gradient, inputs_gradient
    end

    return outputs, forward_pullback
end
=#

function forward(container::Union{SequentialContainer, GraphContainer}, inputs::TrackedArray)
    tracked_args = (container, inputs)
    outputs, pullback = rrule(forward, container, inputs.primal)
    outputs_tracked = IntermediateTracked(outputs, tracked_args, pullback)
    return outputs_tracked
end

function ChainRules.rrule(::typeof(forward), container::Union{SequentialContainer, GraphContainer}, inputs::AbstractArray)
    # doing the forward pass of the container 
    outputs = forward(container, inputs)
    function forward_pullback(derivative_loss::AbstractArray)
        # doing the backpropagation of the container
        backward(container, derivative_loss)
        # defining gradients
        forward_gradient = NoTangent()
        container_gradient = NoTangent()
        inputs_gradient = container.previous_losses

        return forward_gradient, container_gradient, inputs_gradient
    end

    return outputs, forward_pullback
end

# different union types for different classes of layers
no_params_layers = Union{MaxPool, AvgPool, AdaptiveMaxPool, AdaptiveAvgPool, Reshape, Softmax, Identity}
params_layers = Union{Conv, DepthwiseConv, ConvTranspose, Fc, BatchNorm2d}
all_layers = Union{no_params_layers, params_layers}

function forward(layer::all_layers, inputs::TrackedArray)
    tracked_args = (layer, inputs)
    outputs, pullback = rrule(forward, layer, inputs.primal)
    outputs_tracked = IntermediateTracked(outputs, tracked_args, pullback)
    return outputs_tracked
end

function ChainRules.rrule(::typeof(forward), layer::all_layers, inputs::AbstractArray)
    # doing the forward pass of the layer 
    outputs = forward(layer, inputs)
    function forward_pullback(derivative_loss::AbstractArray)
        # doing the backpropagation of the layer 
        layer.losses = derivative_loss
        compute_previous_losses(layer)
        # if the layer has trainable parameters, than also calculate the gradients with respect to the parameters/weights
        if typeof(layer) <: params_layers
            compute_gradients(layer)
        end
        # defining gradients
        forward_gradient = NoTangent()
        layer_gradient = NoTangent()
        inputs_gradient = layer.previous_losses

        return forward_gradient, layer_gradient, inputs_gradient
    end

    return outputs, forward_pullback
end

# custom rules for selected operators/functions

function ChainRules.rrule(::typeof(-), a::AbstractArray, b::AbstractArray)
    # doing the forward pass
    output = a - b # -(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = seed # .* ones(eltype(a), size(a)) 
        b_gradient = -seed # .* ones(eltype(b), size(b)) 

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function get_number_of_extensions_old(gradient_target::Tuple{Vararg{Integer}}, other::Tuple{Vararg{Integer}})
    # check if gradient_target is "longer" (check if the array corresponding to gradient_target has more values)
    # if length(gradient_target) >= length(other)
    if sum(gradient_target) >= sum(other)
        return 0, [], []
    end
    # change names 
    at = gradient_target # shorter
    bt = other # longer
    # extend at
    at_extended = Int[]
    for i in 1:length(bt)
        if i <= length(at) && at[i] != 1 # && bt[i] != 1
            push!(at_extended, at[i])
        else
            push!(at_extended, 0)
        end
    end
    at_extended = Tuple(at_extended)
    # calculate differences
    differences = abs.(at_extended .- bt)
    # calculate the number of extensions during broadcasting
    # number_extensions = sum(differences)
    new_differences = Int[]
    for x in differences  
        if x != 0
            push!(new_differences, x)
        end
    end
    number_extensions = prod(new_differences)

    # println(at)
    # println(bt)
    # println(at_extended)
    # println(number_extensions)

    #=
    # extend at
    at_extended = Int[]
    for i in 1:length(bt)
        if i <= length(at)
            push!(at_extended, at[i])
        else
            push!(at_extended, 0)
        end
    end
    at_extended = Tuple(at_extended)
    # calculate differences
    differences = abs.(at_extended .- bt)
    =#

    dims = Int[]
    dims_drop = Int[]
    for (dim, difference) in enumerate(differences)
        if difference != 0 # && at[dim] != 1 # || at[dim] == 1 # vieleicht ohen das!!
            push!(dims, dim)
            if dim <= length(at)
                if at[dim] != 1
                    push!(dims_drop, dim)
                end
            else
                push!(dims_drop, dim)
            end
        end
    end
    # println("differences: $differences")
    # println("dims: $dims")

    return number_extensions, dims, dims_drop
end

function get_number_of_extensions(gradient_target::Tuple{Vararg{Integer}}, other::Tuple{Vararg{Integer}})
    # check if gradient_target is "longer" (check if the array corresponding to gradient_target has more values)
    if sum(gradient_target) >= sum(other)
        return 0, [], []
    end
    # change names 
    at = gradient_target # shorter
    bt = other # longer
    # extend at
    at_extended = Int[]
    for i in 1:length(bt)
        if i <= length(at) && at[i] != 1 # && bt[i] != 1
            push!(at_extended, at[i])
        else
            push!(at_extended, 0)
        end
    end
    at_extended = Tuple(at_extended)
    # calculate differences
    differences = abs.(at_extended .- bt)
    # calculate the number of extensions during broadcasting
    new_differences = Int[]
    for x in differences  
        if x != 0
            push!(new_differences, x)
        end
    end
    number_extensions = prod(new_differences)

    # calculate the dimensions along the gradient should be summed and the dimensions which should be dropped afterward
    dims = Int[]
    dims_drop = Int[]
    for (dim, difference) in enumerate(differences)
        if difference != 0 # && at[dim] != 1 # || at[dim] == 1 # vieleicht ohen das!!
            push!(dims, dim)
            if dim <= length(at)
                if at[dim] != 1
                    push!(dims_drop, dim)
                end
            else
                push!(dims_drop, dim)
            end
        end
    end

    return number_extensions, dims, dims_drop
end

#=
function ChainRules.rrule(::typeof(.+), a::AbstractArray, b::AbstractArray)
    # doing the forward pass
    output = a .+ b # .+(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        # a_gradient = ones(eltype(a), size(a))
        # b_gradient = ones(eltype(b), size(b)) # * 5
        #=
        a_gradient = ones(eltype(a), size(a)) .+ zeros(eltype(a), size(b))
        println(a_gradient[1])
        b_gradient = zeros(eltype(b), size(a)) .+ ones(eltype(b), size(b))
        println(b_gradient[1])
        exit()
        =#
        a_gradient = ones(eltype(a), size(a))
        b_gradient = ones(eltype(b), size(b))
        a_extensions, a_dims, a_dims_drop = get_number_of_extensions(size(a), size(b))
        b_extensions, b_dims, b_dims_drop = get_number_of_extensions(size(b), size(a))
        if a_extensions != 0
            println(a_extensions)
            a_gradient *= a_extensions
        end
        if b_extensions != 0
            ## println(b_extensions)
            ## b_gradient *= b_extensions # 25
            b_gradient = sum(ones(eltype(b), size(output)), dims=b_dims)
            # b_gradient = dropdims(b_gradient, dims=b_dims[1])
            # for (i, dim) in enumerate(sort(b_dims[2:end]))
            println(size(b_gradient))
            println("start")
            for (i, dim) in enumerate(sort(b_dims_drop))
                println(dim)
                println(i)
                # if dim size(b)[dim] != 1
                    if i == 1
                        b_gradient = dropdims(b_gradient, dims=dim)
                    else
                        b_gradient = dropdims(b_gradient, dims=dim - (i - 1))
                    end
                # end
                println(size(b_gradient))
            end
            # println(size(b_gradient))
        end

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end
=#

function Base.broadcasted(::typeof(+), a::TrackedArray, b::TrackedArray)
    tracked_args = (a, b)
    output, pullback = rrule(.+, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.+), a::AbstractArray, b::AbstractArray)
    # doing the forward pass
    output = a .+ b # .+(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = seed # .* ones(eltype(a), size(a))
        b_gradient = seed # .* ones(eltype(b), size(b))
        a_extensions, a_dims, a_dims_drop = get_number_of_extensions(size(a), size(b))
        b_extensions, b_dims, b_dims_drop = get_number_of_extensions(size(b), size(a))
        if a_extensions != 0
            # a_gradient *= a_extensions
            # a_gradient = sum(ones(eltype(a), size(output)) .* seed, dims=a_dims)
            a_gradient = sum(seed, dims=a_dims)
            for (i, dim) in enumerate(sort(a_dims_drop))
                if i == 1
                    a_gradient = dropdims(a_gradient, dims=dim)
                else
                    a_gradient = dropdims(a_gradient, dims=dim - (i - 1))
                end
            end
        end
        if b_extensions != 0
            # b_gradient *= b_extensions
            # b_gradient = sum(ones(eltype(b), size(output)) .* seed, dims=b_dims)
            b_gradient = sum(seed, dims=b_dims)
            for (i, dim) in enumerate(sort(b_dims_drop))
                if i == 1
                    b_gradient = dropdims(b_gradient, dims=dim)
                else
                    b_gradient = dropdims(b_gradient, dims=dim - (i - 1))
                end
            end
        end

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(-), a::TrackedArray, b::TrackedArray)
    tracked_args = (a, b)
    output, pullback = rrule(.-, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.-), a::AbstractArray, b::AbstractArray)
    # doing the forward pass
    output = a .- b # .-(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = seed # .* ones(eltype(a), size(a))
        b_gradient = -seed # .* -ones(eltype(b), size(b))
        a_extensions, a_dims, a_dims_drop = get_number_of_extensions(size(a), size(b))
        b_extensions, b_dims, b_dims_drop = get_number_of_extensions(size(b), size(a))
        if a_extensions != 0
            # a_gradient *= a_extensions
            # a_gradient = sum(ones(eltype(a), size(output)) .* seed, dims=a_dims)
            a_gradient = sum(seed, dims=a_dims)
            for (i, dim) in enumerate(sort(a_dims_drop))
                if i == 1
                    a_gradient = dropdims(a_gradient, dims=dim)
                else
                    a_gradient = dropdims(a_gradient, dims=dim - (i - 1))
                end
            end
        end
        if b_extensions != 0
            # b_gradient *= b_extensions
            b_gradient = sum(-seed, dims=b_dims)
            for (i, dim) in enumerate(sort(b_dims_drop))
                if i == 1
                    b_gradient = dropdims(b_gradient, dims=dim)
                else
                    b_gradient = dropdims(b_gradient, dims=dim - (i - 1))
                end
            end
        end

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(*), a::TrackedArray, b::TrackedArray)
    tracked_args = (a, b)
    output, pullback = rrule(.*, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.*), a::AbstractArray, b::AbstractArray)
    # doing the forward pass
    output = a .* b # .*(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = b .* seed
        b_gradient = a .* seed
        a_extensions, a_dims, a_dims_drop = get_number_of_extensions(size(a), size(b))
        b_extensions, b_dims, b_dims_drop = get_number_of_extensions(size(b), size(a))
        if a_extensions != 0
            # a_gradient *= a_extensions
            a_gradient = sum(b .* seed, dims=a_dims)
            for (i, dim) in enumerate(sort(a_dims_drop))
                if i == 1
                    a_gradient = dropdims(a_gradient, dims=dim)
                else
                    a_gradient = dropdims(a_gradient, dims=dim - (i - 1))
                end
            end
        end
        if b_extensions != 0
            # b_gradient *= b_extensions
            b_gradient = sum(a .* seed, dims=b_dims)
            for (i, dim) in enumerate(sort(b_dims_drop))
                if i == 1
                    b_gradient = dropdims(b_gradient, dims=dim)
                else
                    b_gradient = dropdims(b_gradient, dims=dim - (i - 1))
                end
            end
        end

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(^), a::TrackedArray, b::TrackedArray)
    tracked_args = (a, b)
    output, pullback = rrule(.^, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.^), a::AbstractArray, b::AbstractArray)
    # doing the forward pass
    output = a .^ b # .^(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = (b .* a .^ (b .- 1)) .* seed
        b_gradient = (a .^ b .* log.(a)) .* seed
        a_extensions, a_dims, a_dims_drop = get_number_of_extensions(size(a), size(b))
        b_extensions, b_dims, b_dims_drop = get_number_of_extensions(size(b), size(a))
        if a_extensions != 0
            # a_gradient *= a_extensions
            a_gradient = sum(a_gradient, dims=a_dims)
            for (i, dim) in enumerate(sort(a_dims_drop))
                if i == 1
                    a_gradient = dropdims(a_gradient, dims=dim)
                else
                    a_gradient = dropdims(a_gradient, dims=dim - (i - 1))
                end
            end
        end
        if b_extensions != 0
            # b_gradient *= b_extensions
            b_gradient = sum(b_gradient, dims=b_dims)
            for (i, dim) in enumerate(sort(b_dims_drop))
                if i == 1
                    b_gradient = dropdims(b_gradient, dims=dim)
                else
                    b_gradient = dropdims(b_gradient, dims=dim - (i - 1))
                end
            end
        end

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(/), a::TrackedArray, b::TrackedArray)
    tracked_args = (a, b)
    output, pullback = rrule(./, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(./), a::AbstractArray, b::AbstractArray)
    # doing the forward pass
    output = a ./ b # ./(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = (1 ./ b) .* seed
        b_gradient = -(a ./ (b .^ 2)) .* seed
        a_extensions, a_dims, a_dims_drop = get_number_of_extensions(size(a), size(b))
        b_extensions, b_dims, b_dims_drop = get_number_of_extensions(size(b), size(a))
        if a_extensions != 0
            # a_gradient *= a_extensions
            a_gradient = sum(a_gradient, dims=a_dims)
            for (i, dim) in enumerate(sort(a_dims_drop))
                if i == 1
                    a_gradient = dropdims(a_gradient, dims=dim)
                else
                    a_gradient = dropdims(a_gradient, dims=dim - (i - 1))
                end
            end
        end
        if b_extensions != 0
            # b_gradient *= b_extensions
            b_gradient = sum(b_gradient, dims=b_dims)
            for (i, dim) in enumerate(sort(b_dims_drop))
                if i == 1
                    b_gradient = dropdims(b_gradient, dims=dim)
                else
                    b_gradient = dropdims(b_gradient, dims=dim - (i - 1))
                end
            end
        end

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(*), a::TrackedArray, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(.*, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function *(a::TrackedArray, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(.*, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.*), a::AbstractArray, b::Real)
    # doing the forward pass
    output = a .* b # .*(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = b .* seed
        b_gradient = sum(a .* seed)

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(*), a::Real, b::TrackedArray)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(.*, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function *(a::Real, b::TrackedArray)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(.*, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.*), a::Real, b::AbstractArray)
    # doing the forward pass
    output = a .* b # .*(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = sum(b .* seed)
        b_gradient = a .* seed

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(/), a::TrackedArray, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(./, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function Base.broadcasted(::typeof(/), a::Real, b::TrackedArray)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(./, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function /(a::TrackedArray, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(./, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(./), a::AbstractArray, b::Real)
    # doing the forward pass
    output = a ./ b # ./(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = (1 ./ b) .* seed
        b_gradient = sum(-(a ./ (b .^ 2)) .* seed)

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function ChainRules.rrule(::typeof(./), a::Real, b::AbstractArray)
    # doing the forward pass
    output = a ./ b # ./(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = sum((1 ./ b) .* seed)
        b_gradient = -(a ./ (b .^ 2)) .* seed

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(^), a::TrackedArray, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(.^, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.^), a::AbstractArray, b::Real)
    # doing the forward pass
    output = a .^ b # .^(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = (b .* a .^ (b .- 1)) .* seed
        b_gradient = sum((a .^ b .* log.(a)) .* seed)

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(^), a::Real, b::TrackedArray)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(.^, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.^), a::Real, b::AbstractArray)
    # doing the forward pass
    output = a .^ b # .^(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = sum((b .* a .^ (b .- 1)) .* seed)
        b_gradient = (a .^ b .* log.(a)) .* seed

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(+), a::TrackedArray, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(.+, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.+), a::AbstractArray, b::Real)
    # doing the forward pass
    output = a .+ b # .+(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = seed
        b_gradient = sum(seed)

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(+), a::Real, b::TrackedArray)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(.+, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.+), a::Real, b::AbstractArray)
    # doing the forward pass
    output = a .+ b # .+(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = sum(seed)
        b_gradient = seed

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(-), a::TrackedArray, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(.-, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.-), a::AbstractArray, b::Real)
    # doing the forward pass
    output = a .- b # .-(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = seed
        b_gradient = sum(-seed)

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function Base.broadcasted(::typeof(-), a::Real, b::TrackedArray)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(.-, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ChainRules.rrule(::typeof(.-), a::Real, b::AbstractArray)
    # doing the forward pass
    output = a .- b # .-(A, B)
    function pullback(seed::AbstractArray)
        # doing the backpropagation/defining gradients
        function_gradient = NoTangent()
        a_gradient = sum(seed)
        b_gradient = -seed

        return function_gradient, a_gradient, b_gradient
    end

    return output, pullback
end

function +(a::TR, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(+, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function +(a::Real, b::TR)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(+, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function -(a::TR, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(-, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function -(a::Real, b::TR)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(-, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function *(a::TR, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(*, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function *(a::Real, b::TR)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(*, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function /(a::TR, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(/, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function /(a::Real, b::TR)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(/, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ^(a::TR, b::Real)
    b = TrackedReal(b, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(^, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

function ^(a::Real, b::TR)
    a = TrackedReal(a, nothing, nothing, nothing)
    tracked_args = (a, b)
    output, pullback = rrule(^, a.primal, b.primal)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

# convert(::Type{TR}, x::R) where R <: Real = TrackedReal(x, nothing, nothing, nothing)
# Base.promote_rule(::Type{TrackedReal}, x::Type{<: Real}) = TR

refresh_rules()