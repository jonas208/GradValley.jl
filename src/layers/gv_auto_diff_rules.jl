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

refresh_rules()