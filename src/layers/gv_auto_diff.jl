# a real number for saving the computaional graph
mutable struct TrackedReal{T} # <: Real
    primal::T
    inputs::Union{Nothing, Any} # Vector{<: TrackedReal}, Vector{<: Union{TrackedArray, TrackedReal}}
    # gradient::Union{Nothing, T} # if nothing, the gradient will not be saved
    gradient::Union{Nothing, Real} # if nothing, the gradient will not be saved
    pullback::Union{Nothing, Function} # if nothing, it is a start value
end
TR = TrackedReal # alias

# an array of real numbers for saving the computaional graph
mutable struct TrackedArray{T, N} <: AbstractArray{T, N}
    primal::AbstractArray{T, N}
    inputs::Union{Nothing, Any} # Vector{<: TrackedArray}, Vector{<: Union{TrackedArray, TrackedReal}}
    # gradient::Union{Nothing, AbstractArray{T, N}} # if nothing, the gradient will not be saved
    gradient::Union{Nothing, AbstractArray{<: Real, N}} # if nothing, the gradient will not be saved
    pullback::Union{Nothing, Function} # if nothing, it is a start values array
end
TA = TrackedArray # alias

# an intermediate tracked real/array in the computaional graph (no gradients will be saved)
function IntermediateTracked(primal::Union{Real, AbstractArray}, tracked_args::Tuple{Vararg}, pullback::Function) # tracked_args::Tuple{Vararg{Union{TR, TA}}}
    if typeof(primal) <: Real
        return TR(primal, collect(tracked_args), nothing, pullback)
    elseif typeof(primal) <: AbstractArray
        return TA(primal, collect(tracked_args), nothing, pullback)
    else
        error("GradValley: AnyIntermediateTracked: primal must be a real or an abstract array")
    end
end

# returns the primal of a tracked type
primal(tracked::Union{TrackedReal, TrackedArray}) = tracked.primal
# if primal is called with any other type, the given variable is just returned
primal(other) = other

# a tracked real/array with gradients saved during backward pass
function TrackedWithGradient(primal::Union{Real, AbstractArray})
    if typeof(primal) <: Real
        return TR(primal, nothing, zero(typeof(primal)), nothing)
    elseif typeof(primal) <: CuArray
        return TA(primal, nothing, CUDA.zeros(eltype(primal), size(primal)...), nothing)
    elseif typeof(primal) <: AbstractArray
        return TA(primal, nothing, zeros(eltype(primal), size(primal)...), nothing)
    else
        error("GradValley: TrackedWithGradient: primal must be a real or an abstract array")
    end
end

# check if a rule should be used, returns true or false (function from Nabla.jl)
function should_use_rrule(sig)
    opT, argTs = Iterators.peel(ExprTools.parameters(sig))
    opT <: Core.Builtin && return false  # can't do operator overloading for builtins

    # isabstracttype(opT) || fieldcount(opT) == 0 || return false  # not handling functors
    # isempty(argTs) && return false  # we are an operator overloading AD, need operands

    try
        isabstracttype(opT) || fieldcount(opT) == 0 || return false  # not handling functors
        isempty(argTs) && return false  # we are an operator overloading AD, need operands
    catch ArgumentError
        return false
    end

    # Don't care about NaNMath
    opT isa DataType && nameof(opT.name.module) == :NaNMath  && return false

    # Ignore functions that have complex ranges. This may change when Nabla supports complex
    # numbers.
    opT ∈ typeof.((
        SpecialFunctions.hankelh1, SpecialFunctions.hankelh2,
        log1p, rem2pi, mod, atan, rem,
    ))  && return false
    opT <: Type{<:Complex} && return false  # skip complex constructor

    # Ignore these functions because they have better Nabla specific versions.
    opT ∈ typeof.((
        isapprox, axes, size, length, isassigned, one, zero,
        Base.Broadcast.combine_styles,  #TODO should i keep this?
    )) && return false

    # Ignore these functions because in practice they are never used and defining them cause
    # a ton of compiler invalidations, making loading slow.
    opT ∈ typeof.((
        string, repr, print, println, write, readlines, eachline, Core.print, Core.println,
        isequal, ==, in, haskey,
        isnothing, ismissing, isfile,
        isbitstype, isbits, isabstracttype, isconcretetype,
        startswith, endswith, join, joinpath, normpath, chomp,
        schedule,  # this one is huge, causes over 2500 invalidations
    )) && return false

    #=
    # Rules currently implemented directly in Nabla, but that could use ChainRules in future
    sig <: Union{ 
        Tuple{typeof(+),AbstractArray,LinearAlgebra.UniformScaling},
        Tuple{typeof(+),LinearAlgebra.UniformScaling,AbstractArray},
        Tuple{typeof(/),Number,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.symm),Char,Char,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.symm),Char,Char,Number,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.symv),Char,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.symv),Char,Number,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.trmm),Char,Char,Char,Char,Number,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.trmv),Char,Char,Char,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.trsm),Char,Char,Char,Char,Number,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.trsv),Char,Char,Char,AbstractArray,AbstractArray},
        Tuple{typeof(Statistics.mean),Function,AbstractArray},
        Tuple{typeof(\),AbstractArray,Number},
        Tuple{typeof(broadcast),Any,Vararg},
        Tuple{typeof(copy),Any},
        Tuple{typeof(float),Any},
        Tuple{typeof(getindex),Ref},
        Tuple{typeof(kron),AbstractArray,AbstractArray},
        Tuple{typeof(map),Function,Vararg},
        Tuple{typeof(mapfoldl),Any,Union{typeof(+), typeof(Base.add_sum)},Union{Number,AbstractArray}},
        Tuple{typeof(mapfoldr),Any,Union{typeof(+), typeof(Base.add_sum)},Union{Number,AbstractArray}},
        Tuple{typeof(mapreduce),Any,Union{typeof(+), typeof(Base.add_sum)},AbstractArray},
        Tuple{typeof(sum),Function,AbstractArray},
        Tuple{typeof(sum),typeof(abs2),AbstractArray},
    } && return false
    =#

    # Functions that cause Nabla to have issues and that we don't use
    sig <: Union{
        Tuple{Type{<:Array}, AbstractArray},  # Nabla support for constructors is limitted

        Tuple{Type{T}, LinearAlgebra.UniformScaling{<:Bool}, Any} where T<:(AbstractMatrix),
        Tuple{Type{T}, LinearAlgebra.UniformScaling, Any} where T<:(AbstractMatrix),
        Tuple{Type{T}, AbstractMatrix} where T<:(SparseArrays.AbstractSparseMatrix),
        Tuple{Type{T}, Number} where T<:LinearAlgebra.UniformScaling,    
        Tuple{Type{AbstractArray{T}}, AbstractArray} where T,
        Tuple{Type{T}, AbstractVector} where T<:(SparseArrays.AbstractSparseVector),   

    } && return false

    return true  # no exclusion applies
end

# hook function for "on_new_rule" in ChainRulesOverloadGeneration
function define_tracked_overload(sig)
    # decide if to use the current rule
    if !should_use_rrule(sig)
        return
    end

    sig = Base.unwrap_unionall(sig)  # not really handling most UnionAll
    opT, argTs = Iterators.peel(sig.parameters)

    N = length(sig.parameters) - 1  # skip the op
    fdef = quote
        # function (op::$opT)(tracked_args::Vararg{Union{TA, TR}, $N}; kwargs...)
        function (op::$opT)(tracked_args::Vararg{Union{TA, TR}}; kwargs...)
            args = (op, primal.(tracked_args)...)
            y, y_pullback = rrule(args...; kwargs...)
            y_tracked = IntermediateTracked(y, tracked_args, y_pullback)
            return y_tracked
        end
    end
    eval(fdef)
end

function tracked_backward(tracked::Union{TrackedReal, TrackedArray}, seed::Union{Real, AbstractArray})
    if isnothing(tracked.pullback) # start value
        if !isnothing(tracked.gradient) # has gradient value, than gradient will be saved
            tracked.gradient += seed
        end
    else # expression in the tree (computaional graph)
        if !isnothing(tracked.gradient) # has gradient value, than gradient will be saved
            tracked.gradient += seed
        end
        gradients = tracked.pullback(seed)
        for (index, gradient) in enumerate(gradients)
            gradient = unthunk(gradient)
            if typeof(gradient) <: AbstractArray || typeof(gradient) <: Real
                tracked_backward(tracked.inputs[index - 1], gradient) # - 1 offset because first argument of rrule is always the forward function
            end
        end
    end
end

# attach the define function to the "on_new_rule" hook from ChainRulesOverloadGeneration
on_new_rule(define_tracked_overload, rrule)