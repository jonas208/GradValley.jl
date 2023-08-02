#=Arguments
Optimization algorithems (SGD, MSGD, Nesterov)
=#

# implementation of stochastic gradient descent optimization algorithm (including weight decay and dampening)
@doc raw"""
    SGD(layer_stack::Union{Vector, AbstractContainer}, learning_rate::Real; weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)

Implementation of stochastic gradient descent optimization algorithm (including optional weight decay and dampening).

# Arguments
- `layer_stack::Union{Vector, AbstractContainer}`: the vector OR the container (SequentialContainer/GraphContainer, often simply the whole model) containing the layers with the parameters to be optimized (can also contain layers without parameters)
- `learning_rate::Real`: the learning rate (shouldn't be 0)
- `weight_decay::Real=0.00`: the weight decay (L2 penalty)
- `dampening::Real=0.00`: the dampening (normally just for optimizers with momentum, however, can be theoretically used without, in this case acts like: ``(1 - dampening) \cdot learning\_rate``)
- `maximize::Bool=false`: maximize the parameters, instead of minimizing 

# Definition
For example, a definition of this algorithm in pseudocode can be found [here](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD).
(Note that in this case of a simple SGD with no momentum, the momentum ``μ`` is zero in the sense of the mentioned documentation.)

# Examples
```julia-repl
# define a model
julia> model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# initialize a SGD optimizer with learning-rate equal 0.1 and weight decay equal to 0.5 (otherwise default values)
julia> optimizer = SGD(model, 0.1, weight_decay=0.5)
# create some random input data
julia> input = rand(1000, 32)
# compute the output of the model
julia> output = forward(model, input)
# generate some random target values 
julia> target = rand(size(output)...)
# compute the loss and it's derivative 
julia> loss, loss_derivative = mse_loss(output, target)
# computet the gradients 
julia> backward(model, loss_derivative)
# perform a single optimization step (parameter update)
julia> step!(optimizer)
```
"""
mutable struct SGD
    layer_stack::Vector
    learning_rate::Real
    weight_decay::Real
    dampening::Real
    maximize::Bool
    modified_weight_gradients::Vector
    modified_bias_gradients::Vector
    iteration_counter::Int
    # custom constructor
    function SGD(layer_stack::Union{Vector, AbstractContainer}, learning_rate::Real; weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)
        if typeof(layer_stack) <: AbstractContainer
            layer_stack = extract_layers(layer_stack, [])
        end

        modified_weight_gradients = Any[false for layer_index in 1:length(layer_stack)]
        modified_bias_gradients = Any[false for layer_index in 1:length(layer_stack)]

        # create new instance/object
        new(layer_stack,
            learning_rate,
            weight_decay,
            dampening,
            maximize,
            modified_weight_gradients,
            modified_bias_gradients,
            0)
    end
end

# implementation of stochastic gradient descent with momentum optimization algorithm (including weight decay and dampening)
@doc raw"""
    MSGD(layer_stack::Union{Vector, AbstractContainer}, learning_rate::Real; momentum::Real=0.90, weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)

Implementation of stochastic gradient descent with momentum optimization algorithm (including optional weight decay and dampening).

# Arguments
- `layer_stack::Union{Vector, AbstractContainer}`: the vector OR the container (SequentialContainer/GraphContainer, often simply the whole model) containing the layers with the parameters to be optimized (can also contain layers without any parameters)
- `learning_rate::Real`: the learning rate (shouldn't be 0)
- `momentum::Real=0.90`: the momentum factor (shouldn't be 0)
- `weight_decay::Real=0.00`: the weight decay (L2 penalty)
- `dampening::Real=0.00`: the dampening for the momentum 
- `maximize::Bool=false`: maximize the parameters, instead of minimizing 

# Definition
For example, a definition of this algorithm in pseudocode can be found [here](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD).
(Note that in this case of SGD with default momentum, in the sense of the mentioned documentation, the momentum ``\mu`` isn't zero (``\mu \neq 0``) and ``nesterov`` is ``false``.)

# Examples
```julia-repl
# define a model
julia> model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# initialize a MSGD optimizer with learning-rate equal 0.1 and momentum equal to 0.75 (otherwise default values)
julia> optimizer = Nesterov(model, 0.1, momentum=0.75)
# create some random input data
julia> input = rand(32, 1000)
# compute the output of the model
julia> output = forward(model, input)
# generate some random target values 
julia> target = rand(size(output)...)
# compute the loss and it's derivative 
julia> loss, loss_derivative = mse_loss(output, target)
# computet the gradients 
julia> backward(model, loss_derivative)
# perform a single optimization step (parameter update)
julia> step!(optimizer)
```
"""
mutable struct MSGD
    layer_stack::Vector
    learning_rate::Real
    momentum::Real
    weight_decay::Real
    dampening::Real
    maximize::Bool
    modified_weight_gradients::Vector
    modified_bias_gradients::Vector
    iteration_counter::Int
    # custom constructor
    function MSGD(layer_stack::Union{Vector, AbstractContainer}, learning_rate::Real; momentum::Real=0.90, weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)
        if typeof(layer_stack) <: AbstractContainer
            layer_stack = extract_layers(layer_stack, [])
        end

        modified_weight_gradients = Any[false for layer_index in 1:length(layer_stack)]
        modified_bias_gradients = Any[false for layer_index in 1:length(layer_stack)]

        # create new instance/object
        new(layer_stack,
            learning_rate,
            momentum,
            weight_decay,
            dampening,
            maximize,
            modified_weight_gradients,
            modified_bias_gradients,
            0)
    end
end

# implementation of nesterov optimization algorithm (including weight decay and dampening)
@doc raw"""
    Nesterov(layer_stack::Union{Vector, AbstractContainer}, learning_rate::Real; momentum::Real=0.90, weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)

Implementation of stochastic gradient descent with nesterov momentum optimization algorithm (including optional weight decay and dampening).

# Arguments
- `layer_stack::Union{Vector, AbstractContainer}`: the vector OR the container (SequentialContainer/GraphContainer, often simply the whole model) containing the layers with the parameters to be optimized (can also contain layers without any parameters)
- `learning_rate::Real`: the learning rate (shouldn't be 0)
- `momentum::Real=0.90`: the momentum factor (shouldn't be 0)
- `weight_decay::Real=0.00`: the weight decay (L2 penalty)
- `dampening::Real=0.00`: the dampening for the momentum (for true nesterov momentum, dampening must be 0)
- `maximize::Bool=false`: maximize the parameters, instead of minimizing 

# Definition
For example, a definition of this algorithm in pseudocode can be found [here](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD).
(Note that in this case of SGD with nesterov momentum, ``nesterov`` is ``true`` in the sense of the mentioned documentation.)

# Examples
```julia-repl
# define a model
julia> model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# initialize a Nesterov optimizer with learning-rate equal 0.1 and nesterov momentum equal to 0.8 (otherwise default values)
julia> optimizer = Nesterov(model, 0.1, momentum=0.8)
# create some random input data
julia> input = rand(1000, 32)
# compute the output of the model
julia> output = forward(model, input)
# generate some random target values 
julia> target = rand(size(output)...)
# compute the loss and it's derivative 
julia> loss, loss_derivative = mse_loss(output, target)
# computet the gradients 
julia> backward(model, loss_derivative)
# perform a single optimization step (parameter update)
julia> step!(optimizer)
```
"""
mutable struct Nesterov
    layer_stack::Vector
    learning_rate::Real
    momentum::Real
    weight_decay::Real
    dampening::Real
    maximize::Bool
    modified_weight_gradients::Vector
    modified_bias_gradients::Vector
    iteration_counter::Int
    # custom constructor
    function Nesterov(layer_stack::Union{Vector, AbstractContainer}, learning_rate::Real; momentum::Real=0.90, weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)
        if typeof(layer_stack) <: AbstractContainer
            layer_stack = extract_layers(layer_stack, [])
        end
        # layer_stack = layer_stack.layer_stack

        modified_weight_gradients = Any[false for layer_index in 1:length(layer_stack)]
        modified_bias_gradients = Any[false for layer_index in 1:length(layer_stack)]

        # create new instance/object
        new(layer_stack,
            learning_rate,
            momentum,
            weight_decay,
            dampening,
            maximize,
            modified_weight_gradients,
            modified_bias_gradients,
            0)
    end
end

@doc raw"""
    step!(optimizer::Union{SGD, MSGD, Nesterov})

Perform a single optimization step (parameter update) for the given optimizer.

# Examples
```julia-repl
# define a model
julia> model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# initialize an optimizer (which optimizer specifically dosen't matter)
julia> optimizer = SGD(model, 0.1)
# create some random input data
julia> input = rand(1000, 32)
# compute the output of the model
julia> output = forward(model, input)
# generate some random target values 
julia> target = rand(size(output)...)
# compute the loss and it's derivative 
julia> loss, loss_derivative = mse_loss(output, target)
# computet the gradients 
julia> backward(model, loss_derivative)
# perform a single optimization step (parameter update)
julia> step!(optimizer)
```
"""
function step! end

# updates the weights with the hyperparameters defined in the given optimizer of type Union{SGD, MSGD, Nesterov}
function step!(optimizer::Union{SGD, MSGD, Nesterov})
    optimizer.iteration_counter += 1
    if typeof(optimizer) == SGD
        momentum = 0
    else
        momentum = optimizer.momentum
    end
    for (layer_index, layer) in enumerate(optimizer.layer_stack)
        if typeof(layer) <: AbstractParamLayer
            weight = layer.weight
            bias = layer.bias
            weight_gradient = layer.weight_gradient
            bias_gradient = layer.bias_gradient
        else
            continue
        end

        WT = eltype(weight) # weight eltype
        BT = eltype(bias) # bias eltype

        if optimizer.weight_decay != 0.00
            weight_gradient = weight_gradient .+ WT(optimizer.weight_decay) * weight
            bias_gradient = bias_gradient .+ BT(optimizer.weight_decay) * bias
        end

        if optimizer.iteration_counter > 1
            modified_weight_gradient = WT(momentum) * optimizer.modified_weight_gradients[layer_index] + WT((1 - optimizer.dampening)) * weight_gradient 
            modified_bias_gradient = BT(momentum) * optimizer.modified_bias_gradients[layer_index] + BT((1 - optimizer.dampening)) * bias_gradient 
        else
            modified_weight_gradient = weight_gradient
            modified_bias_gradient = bias_gradient
        end

        if typeof(optimizer) == Nesterov
            weight_gradient = weight_gradient + WT(momentum) * modified_weight_gradient
            bias_gradient = bias_gradient + BT(momentum) * modified_bias_gradient
        else
            weight_gradient = modified_weight_gradient
            bias_gradient = modified_bias_gradient
        end

        if optimizer.maximize
            layer.weight += WT(optimizer.learning_rate) * weight_gradient # .+
            layer.bias += BT(optimizer.learning_rate) * bias_gradient # .+
        else
            layer.weight -= WT(optimizer.learning_rate) * weight_gradient # .-
            layer.bias -= BT(optimizer.learning_rate) * bias_gradient # .-
        end

        optimizer.modified_weight_gradients[layer_index] = modified_weight_gradient 
        optimizer.modified_bias_gradients[layer_index] = modified_bias_gradient 
    end
end

# pretty-printing for the SGD, MSGD and Nesterov structs
Base.show(io::IO, sgd::SGD) = print(io, "SGD(learning_rate=$(sgd.learning_rate), weight_decay=$(sgd.weight_decay), dampening=$(sgd.dampening), maximize=$(sgd.maximize), iteration_counter=$(sgd.iteration_counter))") # , layer_stack=$(sgd.layer_stack)
Base.show(io::IO, msgd::MSGD) = print(io, "MSGD(learning_rate=$(msgd.learning_rate), momentum=$(msgd.momentum), weight_decay=$(msgd.weight_decay), dampening=$(msgd.dampening), maximize=$(msgd.maximize), iteration_counter=$(msgd.iteration_counter))") # , layer_stack=$(msgd.layer_stack)
Base.show(io::IO, nesterov::Nesterov) = print(io, "Nesterov(learning_rate=$(nesterov.learning_rate), momentum=$(nesterov.momentum), weight_decay=$(nesterov.weight_decay), dampening=$(nesterov.dampening), maximize=$(nesterov.maximize), iteration_counter=$(nesterov.iteration_counter))") # , layer_stack=$(nesterov.layer_stack)

@doc raw"""
    Adam(layer_stack::Union{Vector, AbstractContainer}; learning_rate::Real=0.001, beta1::Real=0.9, beta2::Real=0.999, epsilon::Real=1e-08, weight_decay::Real=0, amsgrad::Bool=false, maximize::Bool=false)

Implementation of [Adam](https://arxiv.org/abs/1412.6980) optimization algorithm (including the optional [AMSgrad](https://openreview.net/forum?id=ryQu7f-RZ) version of this algorithm and optional weight decay).

# Arguments
- `layer_stack::Union{Vector, SequentialContainer, GraphContainer}`: the vector OR the container (SequentialContainer/GraphContainer, often simply the whole model) containing the layers with the parameters to be optimized (can also contain layers without any parameters)
- `learning_rate::Real=0.001`: the learning rate (shouldn't be 0)
- `beta1::Real=0.9`, `beta2::Real=0.999`: the two coefficients used for computing running averages of gradient and its square
- `epsilon::Real=1e-08`: value for numerical stability 
- `weight_decay::Real=0.00`: the weight decay (L2 penalty)
- `amsgrad::Bool=false`: use the [AMSgrad](https://openreview.net/forum?id=ryQu7f-RZ) version of this algorithm
- `maximize::Bool=false`: maximize the parameters, instead of minimizing 

# Definition
For example, a definition of this algorithm in pseudocode can be found [here](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).

# Examples
```julia-repl
# define a model
julia> model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# initialize a Adam optimizer with default arguments
julia> optimizer = Adam(model)
# create some random input data
julia> input = rand(32, 1000)
# compute the output of the model
julia> output = forward(model, input)
# generate some random target values 
julia> target = rand(size(output)...)
# compute the loss and it's derivative 
julia> loss, loss_derivative = mse_loss(output, target)
# computet the gradients 
julia> backward(model, loss_derivative)
# perform a single optimization step (parameter update)
julia> step!(optimizer)
```
"""
mutable struct Adam
    layer_stack::Vector
    learning_rate::Real
    beta1::Real
    beta2::Real
    weight_decay::Real
    epsilon::Real
    amsgrad::Bool
    maximize::Bool
    m_weight::Vector; m_bias::Vector
    v_weight::Vector; v_bias::Vector
    v_head_max_weight::Vector; v_head_max_bias::Vector
    iteration_counter::Int
    # custom constructor
    function Adam(layer_stack::Union{Vector, AbstractContainer}; learning_rate::Real=0.001, beta1::Real=0.9, beta2::Real=0.999, epsilon::Real=1e-08, weight_decay::Real=0, amsgrad::Bool=false, maximize::Bool=false)
        if typeof(layer_stack) <: AbstractContainer
            layer_stack = extract_layers(layer_stack, [])
        end

        m_weight = Union{AbstractArray, Real}[0 for layer_index in 1:length(layer_stack)]; m_bias = Union{AbstractArray, Real}[0 for layer_index in 1:length(layer_stack)]
        v_weight = Union{AbstractArray, Real}[0 for layer_index in 1:length(layer_stack)]; v_bias = Union{AbstractArray, Real}[0 for layer_index in 1:length(layer_stack)]
        v_head_max_weight = Union{AbstractArray, Real}[0 for layer_index in 1:length(layer_stack)]; v_head_max_bias = Union{AbstractArray, Real}[0 for layer_index in 1:length(layer_stack)]

        # create new instance/object
        new(layer_stack,
            learning_rate,
            beta1,
            beta2,
            weight_decay,
            epsilon,
            amsgrad,
            maximize,
            m_weight, m_bias,
            v_weight, v_bias,
            v_head_max_weight, v_head_max_bias,
            0)
    end
end

# updates the weights with the hyperparameters defined in the given optimizer of type Adam
function step!(optimizer::Adam)
    optimizer.iteration_counter += 1
    for (layer_index, layer) in enumerate(optimizer.layer_stack)
        if typeof(layer) <: AbstractParamLayer
            weight = layer.weight
            bias = layer.bias
            weight_gradient = layer.weight_gradient
            bias_gradient = layer.bias_gradient
        else
            continue
        end

        WT = eltype(weight) # weight eltype
        BT = eltype(bias) # bias eltype

        if optimizer.maximize
            weight_gradient = -weight_gradient
            bias_gradient = -bias_gradient
        end

        if optimizer.weight_decay != 0
            weight_gradient = weight_gradient + WT(optimizer.weight_decay) * weight
            bias_gradient = bias_gradient + BT(optimizer.weight_decay) * bias
        end

        m_weight = WT(optimizer.beta1) * optimizer.m_weight[layer_index] .+ WT((1 - optimizer.beta1)) * weight_gradient
        m_bias = BT(optimizer.beta1) * optimizer.m_bias[layer_index] .+ BT((1 - optimizer.beta1)) * bias_gradient

        v_weight = WT(optimizer.beta2) * optimizer.v_weight[layer_index] .+ WT((1 - optimizer.beta2)) * weight_gradient .^ 2
        v_bias = BT(optimizer.beta2) * optimizer.v_bias[layer_index] .+ BT((1 - optimizer.beta2)) * bias_gradient .^ 2

        m_head_weight = m_weight / WT((1 - optimizer.beta1 ^ optimizer.iteration_counter))
        m_head_bias = m_bias / BT((1 - optimizer.beta1 ^ optimizer.iteration_counter))

        v_head_weight = v_weight / WT((1 - optimizer.beta2 ^ optimizer.iteration_counter))
        v_head_bias = v_bias / BT((1 - optimizer.beta2 ^ optimizer.iteration_counter))

        if optimizer.amsgrad
            v_head_max_weight = max.(optimizer.v_head_max_weight[layer_index], v_head_weight)
            v_head_max_bias = max.(optimizer.v_head_max_bias[layer_index], v_head_bias)
            # v_head_max_weight = max.(0, v_head_weight) # equal to pytorch, however, pytorch may be wrong here 
            # v_head_max_bias = max.(0, v_head_bias) # equal to pytorch, however, pytorch may be wrong here 

            new_weight = weight - WT(optimizer.learning_rate) * m_head_weight ./ (sqrt.(v_head_max_weight) .+ WT(optimizer.epsilon))
            new_bias = bias - BT(optimizer.learning_rate) * m_head_bias ./ (sqrt.(v_head_max_bias) .+ BT(optimizer.epsilon))
        else    
            new_weight = weight - WT(optimizer.learning_rate) * m_head_weight ./ (sqrt.(v_head_weight) .+ WT(optimizer.epsilon))
            new_bias = bias - BT(optimizer.learning_rate) * m_head_bias ./ (sqrt.(v_head_bias) .+ BT(optimizer.epsilon))
        end

        layer.weight = new_weight
        layer.bias = new_bias

        optimizer.m_weight[layer_index] = m_weight; optimizer.m_bias[layer_index] = m_bias
        optimizer.v_weight[layer_index] = v_weight; optimizer.v_bias[layer_index] = v_bias
        if optimizer.amsgrad
            optimizer.v_head_max_weight[layer_index] = v_head_max_weight; optimizer.v_head_max_bias[layer_index] = v_head_max_bias
        end
    end
end

# pretty-printing for the Adam struct
Base.show(io::IO, adam::Adam) = print(io, "Adam(learning_rate=$(adam.learning_rate), beta1=$(adam.beta1), beta2=$(adam.beta2), weight_decay=$(adam.weight_decay), epsilon=$(adam.epsilon), amsgrad=$(adam.amsgrad), maximize=$(adam.maximize), iteration_counter=$(adam.iteration_counter))") # , layer_stack=$(adam.layer_stack)