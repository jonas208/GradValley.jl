# struct SequentialContainer
@doc raw"""
    SequentialContainer(layer_stack::Vector{<: Any})

A sequential container (recommended method for building models). A SequtialContainer can take a vector of layers or other containers (submodules).
While forward-pass, the given inputs are *sequentially* propagated through every layer (or submodule) and the output will be returned.
The execution order during forward pass is of course the same as the order in the vector containing the layers or submodules.

!!! note
    You can use a SequentialContainer in a GraphContainer (and vice versa).
    You can also use a SequentialContainer in a SequentialContainer (nesting allowed).

# Arguments
- `layer_stack::Vector{<: Any}`: the vector containing the layers (or submodules, so other containers), the order of the modules in the vector corresponds to the execution order

# Indexing and Iteration 
The sequential container is indexable and iterable. Indexing one element/iterating behaves like indexing one element of/iterating over 
the `sequential_container.layer_stack` passed to the container at initialization. However, if the index is a range (UnitRange{<: Integer}), 
a new SequentialContainer containing all the requested submodules/layers is initialized and returned. 
`length(sequential_container)` and `size(sequential_container)` both just return the number of modules in the layers vector (equivalent to `length(sequential_container.layer_stack)`).

# Examples
```julia-repl
# a simple chain of fully connected layers
julia> m = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# computing the output of the module (with random inputs)
julia> input = rand(Float32, 1000, 32)
julia> output = forward(m, input)

# a more complicated example with with nested submodules
julia> feature_extractor_part_1 = SequentialContainer([Conv(1, 6, (5, 5), activation_function="relu"), AvgPool((2, 2))])
julia> feature_extractor_part_2 = SequentialContainer([Conv(6, 16, (5, 5), activation_function="relu"), AvgPool((2, 2))])
julia> feature_extractor = SequentialContainer([feature_extractor_part_1, feature_extractor_part_2])
julia> classifier = SequentialContainer([Fc(256, 120, activation_function="relu"), Fc(120, 84, activation_function="relu"), Fc(84, 10)])
julia> m = SequentialContainer([feature_extractor, Reshape((256, )), classifier, Softmax(dims=1)])
# computing the output of the module (with random inputs)
julia> input = rand(Float32, 28, 28, 1, 32)
julia> output = forward(m, input)

# indexing 
julia> m[begin] # returns the feature_extractor submodule (SequentialContainer)
julia> m[end] # returns the softmax layer (Softmax)
julia> m[begin:end-1] # returns the entire model except the softmax layer (a new SequentialContainer is initialized and returned) 

# if a SequentialContainer contains BatchNorm layers (regardless of whether they are nested somewhere in a submodule or not), 
# the mode of all these layers at once can be switched as follows
julia> trainmode!(m)
julia> testmode!(m)

# if a SequentialContainer contains layers with trainable parameters/weights (what is hopefully in nearly all situations the case),
# regardless of whether they are nested somewhere in a submodule or not, the gradients of all these layers at once can be reset as follows
julia> zero_gradients(m)
```
"""
mutable struct SequentialContainer <: AbstractContainer
    layer_stack::Vector{<: Any}
    num_layers::Int
    tracked_input::Union{TrackedReal, TrackedArray} # saved in the sc just because to acces the gradient to the input after backward pass easily 
    tracked_output::Union{TrackedReal, TrackedArray} # contains the computational graph 
    # custom constructor
    function SequentialContainer(layer_stack::Vector{<: Any})
        num_layers = length(layer_stack)
        if num_layers < 2
            error("GradValley: SequentialContainer: the number of layers in layer_stack muste be at least 2")
        end
        # create new instance/object
        new(layer_stack, num_layers)
    end
end

# making the SequentialContainer iterable
Base.iterate(SC::SequentialContainer, state=1) = state > SC.num_layers ? nothing : (SC.layer_stack[state], state+1)
# making the length/size (=num_batches) of the SequentialContainer available
Base.length(SC::SequentialContainer) = SC.num_layers
Base.size(SC::SequentialContainer) = SC.num_layers
# making the SequentialContainer indexable
function Base.getindex(SC::SequentialContainer, index::Integer)
    1 <= index <= SC.num_layers || throw(BoundsError(SC, index))
    return SC.layer_stack[index]
end
Base.firstindex(SC::SequentialContainer) = 1
Base.lastindex(SC::SequentialContainer) = SC.num_layers
function Base.getindex(SC::SequentialContainer, index_range::UnitRange{<: Integer})
    1 <= index_range[1] <= SC.num_layers || throw(BoundsError(SC, index))
    1 <= index_range[end] <= SC.num_layers || throw(BoundsError(SC, index))
    layer_stack = SC.layer_stack[index_range]
    sc = SequentialContainer(layer_stack)
    return sc
end

# function forward(sc::SequentialContainer, input::Union{AbstractArray{T, N}, Real}) where {T, N}
function forward(sc::SequentialContainer, input::Union{AbstractArray, Real})
    tracked_input = TrackedWithGradient(input)
    sc.tracked_input = tracked_input
    
    layer_stack = sc.layer_stack
    tracked_output = tracked_input
    for layer in layer_stack
        # tracked_output = forward(layer, tracked_output)
        tracked_output = layer(tracked_output)
    end

    sc.tracked_output = tracked_output
    primal_output = primal(tracked_output)

    return primal_output
end

# function (sc::SequentialContainer)(input::Union{AbstractArray{T, N}, Real}) where {T, N}
function (sc::SequentialContainer)(input::Union{AbstractArray, Real})
    return forward(sc, input)
end

# function forward(sc::SequentialContainer, input::Union{TrackedArray{T, N}, TrackedReal{T}}) where {T, N}
function forward(sc::SequentialContainer, input::Union{TrackedArray, TrackedReal})
    tracked_args = (sc, input)
    output, pullback = rrule(forward, sc, primal(input))
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

# function ChainRules.rrule(::typeof(forward), container::SequentialContainer, input::Union{AbstractArray{T, N}, Real}) where {T, N}
function ChainRules.rrule(::typeof(forward), container::SequentialContainer, input::Union{AbstractArray, Real})
    # doing the forward pass of the container 
    output = forward(container, input)
    # function forward_pullback(derivative_loss::Union{AbstractArray{T, N}, Real}) where {T, N}
    function forward_pullback(derivative_loss::Union{AbstractArray, Real})
        # doing the backpropagation of the container, SequentialContainer always returns just the gradient w.r.t the one input
        input_gradient = backward(container, unthunk(derivative_loss))
        # defining gradients
        forward_gradient = NoTangent()
        container_gradient = NoTangent()

        return forward_gradient, container_gradient, input_gradient
    end

    return output, forward_pullback
end

"""
    backward(sc::SequentialContainer, derivative_loss::Union{AbstractArray{T, N}, Real}) where {T, N}

The backward function for computing the gradients for a SequentialContainer (highly recommend for model building). The function takes the container (so mostly the whole model)
as the first argument and the derivative of the loss as the second argument. No gradients are returned, they are just saved in the layers the container contains.

!!! warning
    Calling `backward` multiple times can have serious consequences. Gradients are added (accumulated) by convention, so calling `backward` multiple times after the corresponding `forward` call,
    the gradients of the weights AND the returned gradient w.r.t. the input are added up (accumulated)! 
    So even the gradient returned by the `backward` call doesn't stay the same when calling `backward` multiple times after the `forward` call.
    Note that the gradients of the weights can be reset by [`zero_gradients`](@ref) but the gradient w.r.t. to the input of a container cannot be reset (except of course by another `forward` call).

# Examples
```julia-repl
# define a model
julia> m = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# compute the output of the model (with random inputs)
julia> output = forward(m, rand(Float32, 1000, 32))
# use a loss function (with random data as target values) and save the derivative of the loss
julia> loss, derivative_loss = mse_loss(output, rand(Float32, 125, 32)) # note that GradValley.Optimization.mse_loss must be imported
# before the gradients are recalculated, the old gradients should always be reset first
julia> zero_gradients(m)
# backpropagation 
julia> backward(m, derivative_loss)
```
"""
function backward(sc::SequentialContainer, derivative_loss::Union{AbstractArray, Real})
# function backward(sc::SequentialContainer, derivative_loss::Union{AbstractArray{T, N}, Real}) where {T, N}
    tracked_backward(sc.tracked_output, derivative_loss)
    # save the gradient to the input (argument)
    input_gradient = sc.tracked_input.gradient

    return input_gradient
end

# GraphConainer behaves similar to SequentialContainer
# a GraphContainer can be used in itself, a SequentialContainer can be also used in itself
# a SequentialContainer can be used in a GraphContainer and a GraphConainer can be used in a SequentialContainer
# it containes all the layers and their parameters, so saving the model is easy like saving a SequentialContainer
@doc raw"""
    GraphContainer(forward_pass::Function, layer_stack::Vector{<: Any})

A computational graph container (recommended method for building models). A GraphContainer can take a function representing the forward pass of a model
and a vector of layers or other containers (submodules).
While forward-pass, a tracked version of the given inputs are passed through the given forward pass function and the output will be returned.
During forward pass, the computational graph is build by a function overload based automatic differentiation system (AD). During backward pass, this computational graph 
is used to compute the gradients.

!!! note
    You can use a GraphContainer in a SequentialContainer (and vice versa).
    You can also use a GraphContainer in a GraphContainer (nesting allowed).

!!! warning
    Note that the GraphContainer is an experimental feature. The behavior of this module could change dramatically in the future.
    Using this module can may cause problems.

# Arguments
- `forward_pass::Function`: the function representing the forward pass of a model
- `layer_stack::Vector{<: Any}`: the vector containing the layers (or submodules, so other Containers), the order doesn't matter

# Guidelines
GradValley has its own little, rudimentary function overload based automatic differentiation system based on [ChainRules.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).
It was designed to allow simple modifications of a normal sequential signal flow, which is the basis of most neural networks. 
For example, to be able to implement ResNet's residual connections. So it represents an alternative to data flow layers known from other Deep Learning packages.
In a way, it is similar to the forward function known from every [PyTorch model](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#define-the-class).
Since the AD does not offer that much functionality at this point in time, the following guidelines must be observed:
- The forward pass function must take at least two arguments. The first is the vector containing the layers (which was passed to GraphContainer at initialization). The following arguments (the last could also be a Vararg argument) are the data inputs.
- The forward pass function must be written generically enough to accept arrays of type T<:AbstractArray or numbers of type T<:Real as input (starting with the second argument).
- Array inputs that are being differentiated cannot be mutated.
- The initialization of new arrays (for example with `zeros` or `rand`) and their use in mix with the inputs passed to the forward function is not allowed.
- Avoid dot syntax in most cases, there only exist a few differentiation rules for the most basic vectorized operators (.+, .-, .*, ./, .^).

# Examples
```julia-repl
# a simple chain of fully connected layers (equivalent to the first example of SequentialContainer)
julia> layers = [Fc(1000, 500), Fc(500, 250), Fc(250, 125)]
julia> function forward_pass(layers::Vector, input::AbstractArray)
           fc_1, fc_2, fc_3 = layers
           output = forward(fc_1, input)
           output = forward(fc_2, output)
           output = forward(fc_3, output)
           return output
       end
julia> m = GraphContainer(forward_pass, layers)
# computing the output of the module (with random inputs)
julia> input = rand(Float32, 1000, 32)
julia> output = forward(m, input)

# a more complicated example: implementation of an inverted residual block
julia> layers = [Conv(16, 64, (1, 1), activation_function="relu"), 
                 Conv(64, 64, (3, 3), padding=(1, 1), groups=64, activation_function="relu"), # depthwise-conv layer because groups==in_channels
                 Conv(64, 16, (1, 1), activation_function="relu")]
julia> function forward_pass(layers::Vector, input::AbstractArray)
           conv_1, depthwise_conv, conv_2 = layers
           output = forward(conv_1, input)
           output = forward(depthwise_conv, output)
           output = forward(conv_2, output)
           output = output + input # residual/skipped connection
           return output
       end
julia> m = GraphContainer(forward_pass, layers)
# computing the output of the module (with random inputs)
julia> input = rand(Float32, 50, 50, 16, 32)
julia> output = forward(m, input)

# a simple example with a polynomial, just to show that it is possible to use the GraphContainer like an automatic differentiation (AD) tool 
julia> f(layers, x) = 0.5x^3 - 2x^2 + 10
julia> df(x) = 1.5x^2 - 4x # checking the result of the AD with this manually written derivation 
julia> m = GraphContainer(f, [])
julia> y = forward(m, 3)
julia> dydx = backward(m, 1) # in this case, no loss function was used, so we have no gradient information, therefore we use 1 as the so-called seed
1-element Vector{Float64}:
 1.5
julia> manual_dydx = df(3)
1.5
julia> isapprox(dydx[1], manual_dydx)
true

# if a GraphContainer contains BatchNorm layers (regardless of whether they are nested somewhere in a submodule or not), 
# the mode of all these layers at once can be switched as follows
julia> trainmode!(m)
julia> testmode!(m)

# if a GraphContainer contains layers with trainable parameters/weights (what is hopefully in nearly all situations the case),
# regardless of whether they are nested somewhere in a submodule or not, the gradients of all these layers at once can be reset as follows
julia> zero_gradients(m)
```
"""
mutable struct GraphContainer <: AbstractContainer
    layer_stack::Vector{<: Any}
    num_layers::Int
    # the function which defines the forward pass, 
    # any variables are used in the function should either be passed directly to the function or reside in the layers vector
    forward_pass::Function 
    tracked_inputs::Vector{Union{TrackedReal, TrackedArray}} # saved in the grc just because to acces the gradients to the inputs after backward pass easily 
    tracked_output::Union{TrackedReal, TrackedArray} # contains the computational graph 
    # custom constructor
    function GraphContainer(forward_pass::Function, layer_stack::Vector{<: Any})
        num_layers = length(layer_stack)
        # create new instance/object
        new(layer_stack, num_layers, forward_pass)
    end
end

# function forward(grc::GraphContainer, inputs::Vararg{Union{AbstractArray{T, N}, Real}}) where {T, N}
function forward(grc::GraphContainer, inputs::Vararg{Union{AbstractArray, Real}})
    tracked_inputs = [TrackedWithGradient(input) for input in inputs]
    grc.tracked_inputs = tracked_inputs
    tracked_output = grc.forward_pass(grc.layer_stack, tracked_inputs...)
    grc.tracked_output = tracked_output
    primal_output = primal(tracked_output)

    return primal_output
end

# function (grc::GraphContainer)(inputs::Vararg{Union{AbstractArray{T, N}, Real}}) where {T, N}
function (grc::GraphContainer)(inputs::Vararg{Union{AbstractArray, Real}})
    return forward(grc, inputs...)
end

# function forward(layer::GraphContainer, input::Vararg{Union{TrackedArray{T, N}, TrackedReal{T}}}) where {T, N}
function forward(layer::GraphContainer, input::Vararg{Union{TrackedArray, TrackedReal}})
    tracked_args = (layer, input...)
    output, pullback = rrule(forward, layer, primal.(input)...)
    output_tracked = IntermediateTracked(output, tracked_args, pullback)
    return output_tracked
end

# function ChainRules.rrule(::typeof(forward), container::GraphContainer, inputs::Vararg{Union{AbstractArray{T, N}, Real}}) where {T, N}
function ChainRules.rrule(::typeof(forward), container::GraphContainer, inputs::Vararg{Union{AbstractArray, Real}})
    # doing the forward pass of the container 
    output = forward(container, inputs...)
    # function forward_pullback(derivative_loss::Union{AbstractArray{T, N}, Real}) where {T, N}
    function forward_pullback(derivative_loss::Union{AbstractArray, Real})
        # doing the backpropagation of the container, GraphContainer always returns a vector containing the gradients w.r.t each input argument
        input_gradients = backward(container, unthunk(derivative_loss))
        # defining gradients
        forward_gradient = NoTangent()
        container_gradient = NoTangent()

        return forward_gradient, container_gradient, input_gradients...
    end

    return output, forward_pullback
end

"""
    backward(grc::GraphContainer, derivative_loss::Union{AbstractArray{T, N}, Real}) where {T, N}

The backward function for computing the gradients for a GraphContainer (recommend for model building). The function takes the container (so mostly the whole model)
as the first argument and the derivative of the loss as the second argument. The gradients to the input arguments are returned (in a vector, in the same order as the inputs were passed to the `forward` function).

!!! warning
    Calling `backward` multiple times can have serious consequences. Gradients are added (accumulated) by convention, so calling `backward` multiple times after the corresponding `forward` call,
    the gradients of the weights AND the returned gradients w.r.t. the inputs are added up (accumulated)! 
    So even the gradients returned by the `backward` call doesn't stay the same when calling `backward` multiple times after the `forward` call. 
    Note that the gradients of the weights can be reset by [`zero_gradients`](@ref) but the gradients w.r.t. to the inputs of a container cannot be reset (except of course by another `forward` call).

# Examples
```julia-repl
# define a model
julia> layers = [Fc(1000, 500), Fc(500, 250), Fc(250, 125)]
julia> function forward_pass(layers::Vector, input::AbstractArray)
           fc_1, fc_2, fc_3 = layers
           output = forward(fc_1, input)
           output = forward(fc_2, output)
           output = forward(fc_3, output)
           return output
       end
julia> m = GraphContainer(forward_pass, layers)
# compute the output of the model (with random inputs)
julia> input = rand(Float32, 1000, 32)
julia> output = forward(m, input)
# use a loss function (with random data as target values) and save the derivative of the loss
julia> loss, derivative_loss = mse_loss(output, rand(Float32, 125, 32)) # note that GradValley.Optimization.mse_loss must be imported
# before the gradients are (re)calculated, the old gradients should always be reset first
julia> zero_gradients(m)
# backpropagation 
julia> input_gradients = backward(m, derivative_loss) # input_gradients is a vector of length 1 because we only passed one input to the forward function
julia> input_gradient = input_gradients[1] # input_gradient is the gradient w.r.t to the single input
```
"""
function backward(grc::GraphContainer, derivative_loss::Union{AbstractArray, Real})
# function backward(grc::GraphContainer, derivative_loss::Union{AbstractArray{T, N}, Real}) where {T, N}
    tracked_backward(grc.tracked_output, derivative_loss)
    # save the gradients to the inputs (arguments) of the forward pass function (exept the first "layers" argument)
    gradients = [tracked_input.gradient for tracked_input in grc.tracked_inputs]

    return gradients
end

# resets the the gradients of all Conv/ConvTranspose/Fc/BatchNorm layers in the given container
function zero_gradients(container::AbstractContainer)
    for layer in container.layer_stack
        if typeof(layer) <: AbstractParamLayer || typeof(layer) <: AbstractContainer
            zero_gradients(layer)
        end
    end
end

# if the given Container contains BatchNorm layers, there mode will be set to trainmode
function trainmode!(container::AbstractContainer)
    for layer in container.layer_stack
        if typeof(layer) == BatchNorm || typeof(layer) <: AbstractContainer
            trainmode!(layer)
        end
    end
end

# if the given Container contains BatchNorm layers, there mode will be set to testmode
function testmode!(container::AbstractContainer)
    for layer in container.layer_stack
        if typeof(layer) == BatchNorm || typeof(layer) <: AbstractContainer
            testmode!(layer)
        end
    end
end