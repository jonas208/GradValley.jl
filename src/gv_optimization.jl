module Optimization

using ..Layers: SequentialContainer, GraphContainer, Conv, DepthwiseConv, ConvTranspose, BatchNorm2d, Fc, extract_layers
using ..Functional
# make Functional accessible via gv functional
gv_functional = Functional

# export all optimizers and their step! methods
export SGD, MSGD, Nesterov, Adam, step!
# export all loss functions
export mse_loss, mae_loss, bce_loss

#=
Internal functions, Internals
=#

# computes the mean of an arbitrary sized input array
function mean(x)
    return sum(x) / length(x)
end

#=
Optimization algorithems (SGD, MSGD, Nesterov)
=#

# implementation of stochastic gradient descent optimization algorithm (including weight decay and dampening)
@doc raw"""
    SGD(layer_stack::Union{Vector, SequentialContainer, GraphContainer}, learning_rate::Real; weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)

Implementation of stochastic gradient descent optimization algorithm (including optional weight decay and dampening).

# Arguments
- `layer_stack::Union{Vector, SequentialContainer, GraphContainer}`: the vector OR the container (SequentialContainer/GraphContainer, often simply the whole model) containing the layers with the parameters to be optimized (can also contain layers without parameters)
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
    function SGD(layer_stack::Union{Vector, SequentialContainer, GraphContainer}, learning_rate::Real; weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)
        if typeof(layer_stack) == SequentialContainer || typeof(layer_stack) == GraphContainer
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
    MSGD(layer_stack::Union{Vector, SequentialContainer, GraphContainer}, learning_rate::Real; momentum::Real=0.90, weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)

Implementation of stochastic gradient descent with momentum optimization algorithm (including optional weight decay and dampening).

# Arguments
- `layer_stack::Union{Vector, SequentialContainer, GraphContainer}`: the vector OR the container (SequentialContainer/GraphContainer, often simply the whole model) containing the layers with the parameters to be optimized (can also contain layers without any parameters)
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
    function MSGD(layer_stack::Union{Vector, SequentialContainer, GraphContainer}, learning_rate::Real; momentum::Real=0.90, weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)
        if typeof(layer_stack) == SequentialContainer || typeof(layer_stack) == GraphContainer
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
    Nesterov(layer_stack::Union{Vector, SequentialContainer, GraphContainer}, learning_rate::Real; momentum::Real=0.90, weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)

Implementation of stochastic gradient descent with nesterov momentum optimization algorithm (including optional weight decay and dampening).

# Arguments
- `layer_stack::Union{Vector, SequentialContainer, GraphContainer}`: the vector OR the container (SequentialContainer/GraphContainer, often simply the whole model) containing the layers with the parameters to be optimized (can also contain layers without any parameters)
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
    function Nesterov(layer_stack::Union{Vector, SequentialContainer, GraphContainer}, learning_rate::Real; momentum::Real=0.90, weight_decay::Real=0.00, dampening::Real=0.00, maximize::Bool=false)
        if typeof(layer_stack) == SequentialContainer || typeof(layer_stack) == GraphContainer
            # println("something")
            # layer_stack = layer_stack.layer_stack
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
        if typeof(layer) == Conv || typeof(layer) == DepthwiseConv || typeof(layer) == ConvTranspose
            weight = layer.kernels
            bias = layer.bias
            weight_gradient = layer.gradients
            bias_gradient = layer.bias_gradients
        elseif typeof(layer) == Fc
            weight = layer.weights
            bias = layer.bias
            weight_gradient = layer.gradients
            bias_gradient = layer.bias_gradients
        elseif typeof(layer) == BatchNorm2d
            weight = layer.weight_gamma
            bias = layer.weight_beta
            weight_gradient = layer.gradient_gamma
            bias_gradient = layer.gradient_beta
        else
            continue
        end

        if optimizer.weight_decay != 0.00
            weight_gradient = weight_gradient .+ optimizer.weight_decay * weight
            bias_gradient = bias_gradient .+ optimizer.weight_decay * bias
        end

        if optimizer.iteration_counter > 1
            modified_weight_gradient = momentum * optimizer.modified_weight_gradients[layer_index] + (1 - optimizer.dampening) * weight_gradient 
            modified_bias_gradient = momentum * optimizer.modified_bias_gradients[layer_index] + (1 - optimizer.dampening) * bias_gradient 
        else
            modified_weight_gradient = weight_gradient
            modified_bias_gradient = bias_gradient
        end

        if typeof(optimizer) == Nesterov
            weight_gradient = weight_gradient + momentum * modified_weight_gradient
            bias_gradient = bias_gradient + momentum * modified_bias_gradient
        else
            weight_gradient = modified_weight_gradient
            bias_gradient = modified_bias_gradient
        end

        if typeof(layer) == Conv || typeof(layer) == DepthwiseConv || typeof(layer) == ConvTranspose
            if optimizer.maximize
                layer.kernels += optimizer.learning_rate * weight_gradient # .+
                layer.bias += optimizer.learning_rate * bias_gradient # .+
            else
                layer.kernels -= optimizer.learning_rate * weight_gradient # .-
                layer.bias -= optimizer.learning_rate * bias_gradient # .-
            end
        elseif typeof(layer) == Fc
            if optimizer.maximize
                layer.weights += optimizer.learning_rate * weight_gradient # .+
                layer.bias += optimizer.learning_rate * bias_gradient # .+
            else
                layer.weights -= optimizer.learning_rate * weight_gradient # .-
                layer.bias -= optimizer.learning_rate * bias_gradient # .-
            end
        elseif typeof(layer) == BatchNorm2d
            if optimizer.maximize
                layer.weight_gamma += optimizer.learning_rate * weight_gradient # .+
                layer.weight_beta += optimizer.learning_rate * bias_gradient # .+
            else
                layer.weight_gamma -= optimizer.learning_rate * weight_gradient # .-
                layer.weight_beta -= optimizer.learning_rate * bias_gradient # .-
            end
        end

        optimizer.modified_weight_gradients[layer_index] = modified_weight_gradient 
        optimizer.modified_bias_gradients[layer_index] = modified_bias_gradient 
    end
end

# pretty-printing for the SGD, MSGD and Nesterov structs
Base.show(io::IO, sgd::SGD) = print(io, "SGD(learning_rate=$(sgd.learning_rate), weight_decay=$(sgd.weight_decay), dampening=$(sgd.dampening), maximize=$(sgd.maximize), iteration_counter=$(sgd.iteration_counter), layer_stack=$(sgd.layer_stack))")
Base.show(io::IO, msgd::MSGD) = print(io, "MSGD(learning_rate=$(msgd.learning_rate), momentum=$(msgd.momentum), weight_decay=$(msgd.weight_decay), dampening=$(msgd.dampening), maximize=$(msgd.maximize), iteration_counter=$(msgd.iteration_counter), layer_stack=$(msgd.layer_stack))")
Base.show(io::IO, nesterov::Nesterov) = print(io, "Nesterov(learning_rate=$(nesterov.learning_rate), momentum=$(nesterov.momentum), weight_decay=$(nesterov.weight_decay), dampening=$(nesterov.dampening), maximize=$(nesterov.maximize), iteration_counter=$(nesterov.iteration_counter), layer_stack=$(nesterov.layer_stack))")

@doc raw"""
    Adam(layer_stack::Union{Vector, SequentialContainer, GraphContainer}; learning_rate::Real=0.001, beta1::Real=0.9, beta2::Real=0.999, epsilon::Real=1e-08, weight_decay::Real=0, amsgrad::Bool=false, maximize::Bool=false)

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
    function Adam(layer_stack::Union{Vector, SequentialContainer, GraphContainer}; learning_rate::Real=0.001, beta1::Real=0.9, beta2::Real=0.999, epsilon::Real=1e-08, weight_decay::Real=0, amsgrad::Bool=false, maximize::Bool=false)
        if typeof(layer_stack) == SequentialContainer || typeof(layer_stack) == GraphContainer
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
        if typeof(layer) == Conv || typeof(layer) == DepthwiseConv || typeof(layer) == ConvTranspose
            weight = layer.kernels
            bias = layer.bias
            weight_gradient = layer.gradients
            bias_gradient = layer.bias_gradients
        elseif typeof(layer) == Fc
            weight = layer.weights
            bias = layer.bias
            weight_gradient = layer.gradients
            bias_gradient = layer.bias_gradients
        elseif typeof(layer) == BatchNorm2d
            weight = layer.weight_gamma
            bias = layer.weight_beta
            weight_gradient = layer.gradient_gamma
            bias_gradient = layer.gradient_beta
        else
            continue
        end

        if optimizer.maximize
            weight_gradient = -weight_gradient
            bias_gradient = -bias_gradient
        end

        if optimizer.weight_decay != 0
            weight_gradient = weight_gradient + optimizer.weight_decay * weight
            bias_gradient = bias_gradient + optimizer.weight_decay * bias
        end

        m_weight = optimizer.beta1 * optimizer.m_weight[layer_index] .+ (1 - optimizer.beta1) * weight_gradient
        m_bias = optimizer.beta1 * optimizer.m_bias[layer_index] .+ (1 - optimizer.beta1) * bias_gradient

        v_weight = optimizer.beta2 * optimizer.v_weight[layer_index] .+ (1 - optimizer.beta2) * weight_gradient .^ 2
        v_bias = optimizer.beta2 * optimizer.v_bias[layer_index] .+ (1 - optimizer.beta2) * bias_gradient .^ 2

        m_head_weight = m_weight / (1 - optimizer.beta1 ^ optimizer.iteration_counter)
        m_head_bias = m_bias / (1 - optimizer.beta1 ^ optimizer.iteration_counter)

        v_head_weight = v_weight / (1 - optimizer.beta2 ^ optimizer.iteration_counter)
        v_head_bias = v_bias / (1 - optimizer.beta2 ^ optimizer.iteration_counter)

        if optimizer.amsgrad
            v_head_max_weight = max.(optimizer.v_head_max_weight[layer_index], v_head_weight)
            v_head_max_bias = max.(optimizer.v_head_max_bias[layer_index], v_head_bias)
            # v_head_max_weight = max.(0, v_head_weight) # equal to pytorch, however, pytorch may be wrong here 
            # v_head_max_bias = max.(0, v_head_bias) # equal to pytorch, however, pytorch may be wrong here 

            new_weight = weight - optimizer.learning_rate * m_head_weight ./ (sqrt.(v_head_max_weight) .+ optimizer.epsilon)
            new_bias = bias - optimizer.learning_rate * m_head_bias ./ (sqrt.(v_head_max_bias) .+ optimizer.epsilon)
        else    
            new_weight = weight - optimizer.learning_rate * m_head_weight ./ (sqrt.(v_head_weight) .+ optimizer.epsilon)
            new_bias = bias - optimizer.learning_rate * m_head_bias ./ (sqrt.(v_head_bias) .+ optimizer.epsilon)
        end

        if typeof(layer) == Conv || typeof(layer) == DepthwiseConv || typeof(layer) == ConvTranspose
                layer.kernels = new_weight
                layer.bias = new_bias
        elseif typeof(layer) == Fc
                layer.weights = new_weight
                layer.bias = new_bias
        elseif typeof(layer) == BatchNorm2d
                layer.weight_gamma = new_weight
                layer.weight_beta = new_bias
        end

        optimizer.m_weight[layer_index] = m_weight; optimizer.m_bias[layer_index] = m_bias
        optimizer.v_weight[layer_index] = v_weight; optimizer.v_bias[layer_index] = v_bias
        if optimizer.amsgrad
            optimizer.v_head_max_weight[layer_index] = v_head_max_weight; optimizer.v_head_max_bias[layer_index] = v_head_max_bias
        end
    end
end

# pretty-printing for the Adam struct
Base.show(io::IO, adam::Adam) = print(io, "Adam(learning_rate=$(adam.learning_rate), beta1=$(adam.beta1), beta2=$(adam.beta2), weight_decay=$(adam.weight_decay), epsilon=$(adam.epsilon), amsgrad=$(adam.amsgrad), maximize=$(adam.maximize), iteration_counter=$(adam.iteration_counter), layer_stack=$(adam.layer_stack))")

#=
Loss functions (and their derivatives with respect to their prediction inputs)
=#

# mean absolute loss (l1loss), size/shape of prediction and target must be equal
# size/shape of prediction/target: (*) -> where * means any number of dimensions
@doc raw"""
    mae_loss(prediction::AbstractArray{<: Real, N}, target::AbstractArray{<: Real, N}; reduction_method::Union{AbstractString, Nothing}="mean", return_derivative::Bool=true) where N

Calculate the (mean) absolute error (L1 norm, with optional reduction to a single loss value (mean or sum)) and it's derivative with respect to the prediction input.

# Arguments
- `prediction::AbstractArray{<: Real, N}`: the prediction of the model of shape (*), where * means any number of dimensions 
- `target::AbstractArray{<: Real, N}`: the corresponding target values of shape (*), must have the same shape as the prediction input 
- `reduction_method::Union{AbstractString, Nothing}="mean"`: can be `"mean"`, `"sum"` or `nothing`, specifies the reduction method which reduces the element-wise computed loss to a single value
- `return_derivative::Bool=true`: it true, the loss and it's derivative with respect to the prediction input is returned, if false, just the loss will be returned

# Definition
``L`` is the loss value which will be returned. If `return_derivative` is true, then an array with the same shape as prediction/target is returned as well, it contains the partial derivatives of ``L`` w.r.t. to each prediction value: ``\frac{\partial L}{\partial p_i}``, where ``p_i`` in one prediction value.
If `reduction_method` is `nothing`, the element-wise computed losses are returned. Note that for `reduction_method=nothing`, the derivative is just the same as when `reduction_method="sum"`.
The element-wise calculation can be defined as (where ``t_i`` is one target value and ``l_i`` is one loss value): 
```math
\begin{align*}
l_i = |p_i - t_i|
\end{align*}
```
Then, ``L`` and ``\frac{\partial L}{\partial p_i}`` differ a little bit from case to case (``n`` is the number of values in `prediction`/`target`):
```math
\begin{align*}
L;\frac{\partial L}{\partial p_i} = \begin{cases}\frac{1}{n}\sum_{i=1}^{n} l_i; \frac{p_i - t_i}{l_i \cdot n} &\text{for reduction\_method="mean"}\\\sum_{i=1}^{n} l_i; \frac{p_i - t_i}{l_i} &\text{for reduction\_method="sum"}\end{cases}
\end{align*}
```

# Examples
```julia-repl
# define a model
julia> model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# create some random input data
julia> input = rand(32, 1000)
# compute the output of the model
julia> output = forward(model, input)
# generate some random target values 
julia> target = rand(size(output)...)
# compute the loss and it's derivative (with default reduction method "mean")
julia> loss, loss_derivative = mae_loss(output, target)
# computet the gradients 
julia> backward(model, loss_derivative)
```
"""
function mae_loss(prediction::AbstractArray{<: Real, N}, target::AbstractArray{<: Real, N}; reduction_method::Union{AbstractString, Nothing}="mean", return_derivative::Bool=true) where N
    prediction_shape = size(prediction)
    target_shape = size(target)
    if prediction_shape != target_shape
        error("GradValley: mae_loss: size/shape of prediction must be equal to the size/shape of target")
    end
    losses = abs.(prediction - target)
    if isnothing(reduction_method)
        loss = losses
        loss_derivative = (prediction - target) ./ losses
    elseif reduction_method == "mean"
        loss = mean(losses)
        loss_derivative = (1 / length(prediction)) .* ((prediction - target) ./ losses)
    elseif reduction_method == "sum"
        loss = sum(losses)
        loss_derivative = (prediction - target) ./ losses
    else
        error("""GradValley: mae_loss: given reduction_method is invalid
            reduction_method must be either nothing, "mean" or "sum" """)
    end
    if return_derivative
        return loss, loss_derivative
    else
        return loss
    end
end

# mean squared loss (l2loss), size/shape of prediction and target must be equal
# size/shape of prediction/target: (*) -> where * means any number of dimensions
@doc raw"""
    mse_loss(prediction::AbstractArray{<: Real, N}, target::AbstractArray{<: Real, N}; reduction_method::Union{AbstractString, Nothing}="mean", return_derivative::Bool=true) where N

Calculate the (mean) squared error (squared L2 norm, with optional reduction to a single loss value (mean or sum)) and it's derivative with respect to the prediction input.

# Arguments
- `prediction::AbstractArray{<: Real, N}`: the prediction of the model of shape (*), where * means any number of dimensions 
- `target::AbstractArray{<: Real, N}`: the corresponding target values of shape (*), must have the same shape as the prediction input 
- `reduction_method::Union{AbstractString, Nothing}="mean"`: can be `"mean"`, `"sum"` or `nothing`, specifies the reduction method which reduces the element-wise computed loss to a single value
- `return_derivative::Bool=true`: it true, the loss and it's derivative with respect to the prediction input is returned, if false, just the loss will be returned

# Definition
``L`` is the loss value which will be returned. If `return_derivative` is true, then an array with the same shape as prediction/target is returned as well, it contains the partial derivatives of ``L`` w.r.t. to each prediction value: ``\frac{\partial L}{\partial p_i}``, where ``p_i`` in one prediction value.
If `reduction_method` is `nothing`, the element-wise computed losses are returned. Note that for `reduction_method=nothing`, the derivative is just the same as when `reduction_method="sum"`.
The element-wise calculation can be defined as (where ``t_i`` is one target value and ``l_i`` is one loss value): 
```math
\begin{align*}
l_i = (p_i - t_i)^2
\end{align*}
```
Then, ``L`` and ``\frac{\partial L}{\partial p_i}`` differ a little bit from case to case (``n`` is the number of values in `prediction`/`target`):
```math
\begin{align*}
L;\frac{\partial L}{\partial p_i} = \begin{cases}\frac{1}{n}\sum_{i=1}^{n} l_i; \frac{2}{n}(p_i - t_i)  &\text{for reduction\_method="mean"}\\\sum_{i=1}^{n} l_i; 2(p_i - t_i) &\text{for reduction\_method="sum"}\end{cases}
\end{align*}
```

# Examples
```julia-repl
# define a model
julia> model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# create some random input data
julia> input = rand(32, 1000)
# compute the output of the model
julia> output = forward(model, input)
# generate some random target values 
julia> target = rand(size(output)...)
# compute the loss and it's derivative (with default reduction method "mean")
julia> loss, loss_derivative = mse_loss(output, target)
# computet the gradients 
julia> backward(model, loss_derivative)
```
"""
function mse_loss(prediction::AbstractArray{<: Real, N}, target::AbstractArray{<: Real, N}; reduction_method::Union{AbstractString, Nothing}="mean", return_derivative::Bool=true) where N
    prediction_shape = size(prediction)
    target_shape = size(target)
    if prediction_shape != target_shape
        error("GradValley: mse_loss: size/shape of prediction must be equal to the size/shape of target")
    end
    losses = (prediction - target).^2
    if isnothing(reduction_method)
        loss = losses
        loss_derivative = 2 * (prediction - target)
    elseif reduction_method == "mean"
        loss = mean(losses)
        loss_derivative = ((1 / length(prediction)) * 2) .* (prediction - target)
    elseif reduction_method == "sum"
        loss = sum(losses)
        loss_derivative = 2 * (prediction - target)
    else
        error("""GradValley: mse_loss: given reduction_method is invalid
            reduction_method must be either nothing, "mean" or "sum" """)
    end
    if return_derivative
        return loss, loss_derivative
    else
        return loss
    end
end

@doc raw"""
    bce_loss(prediction::AbstractArray{<: Real, N}, target::AbstractArray{<: Real, N}; weight::AbstractArray{<: Real}=ones(size(prediction)[2:end]), reduction_method::Union{AbstractString, Nothing}="mean", return_derivative::Bool=true) where N

Calculate the binary cross entropy loss (with optional rescaling weight and reduction to a single loss value (mean or sum)) and it's derivative with respect to the prediction input.

# Arguments
- `prediction::AbstractArray{<: Real, N}`: the prediction of the model of shape (*), where * means any number of dimensions 
- `target::AbstractArray{<: Real, N}`: the corresponding target values (should be between 0 and 1) of shape (*), must have the same shape as the prediction input 
- `weight::AbstractArray{<: Real}=ones(size(prediction)[2:end])`: a manual rescaling weight given to the loss, it doesn't need a batch dimension (first dimension) because normally, these rescaling weights shouldn't differ within a batch, so the shape of the weight is either equal to the shape of prediction/target or equal to the shape of prediction/target without the first (batch) dimension
- `reduction_method::Union{AbstractString, Nothing}="mean"`: can be `"mean"`, `"sum"` or `nothing`, specifies the reduction method which reduces the element-wise computed loss to a single value
- `return_derivative::Bool=true`: it true, the loss and it's derivative with respect to the prediction input is returned, if false, just the loss will be returned

# Definition
``L`` is the loss value which will be returned. If `return_derivative` is true, then an array with the same shape as prediction/target is returned as well, it contains the partial derivatives of ``L`` w.r.t. to each prediction value: ``\frac{\partial L}{\partial p_i}``, where ``p_i`` in one prediction value.
If `reduction_method` is `nothing`, the element-wise computed losses are returned. Note that for `reduction_method=nothing`, the derivative is just the same as when `reduction_method="sum"`. ``w_i`` is one rescaling weight value.
The element-wise calculation can be defined as (where ``t_i`` is one target value and ``l_i`` is one loss value): 
```math
\begin{align*}
l_i = -w_n(t_i \cdot \log(p_i) + (1 - t_i) \cdot \log(1 - p_i))
\end{align*}
```
Then, ``L`` and ``\frac{\partial L}{\partial p_i}`` differ a little bit from case to case (``n`` is the number of values in `prediction`/`target`):
```math
\begin{align*}
L;\frac{\partial L}{\partial p_i} = \begin{cases}\frac{1}{n}\sum_{i=1}^{n} l_i; \frac{1}{n}(\frac{-w_i t_i}{p_i} - \frac{-w_i + w_i t_i}{1 - p_i}) &\text{for reduction\_method="mean"}\\\sum_{i=1}^{n} l_i; \frac{-w_i t_i}{p_i} - \frac{-w_i + t_i w_i}{1 - p_i} &\text{for reduction\_method="sum"}\end{cases}
\end{align*}
```

# Examples
```julia-repl
# define a model
julia> model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# create some random input data
julia> input = rand(32, 1000)
# compute the output of the model
julia> output = forward(model, input)
# generate some random target values 
julia> target = rand(size(output)...)
# generate some random rescaling weight values (without a batch dimension)
julia> weight = rand(size(output)[2:end]...)
# compute the loss and it's derivative (with default reduction method "mean")
julia> loss, loss_derivative = bce_loss(output, target, weight=weight)
# computet the gradients 
julia> backward(model, loss_derivative)
```
"""
function bce_loss(prediction::AbstractArray{<: Real, N}, target::AbstractArray{<: Real, N}; weight::AbstractArray{<: Real}=ones(size(prediction)[2:end]), reduction_method::Union{AbstractString, Nothing}="mean", return_derivative::Bool=true) where N
    # weight::Vector{Real}=ones(eltype(prediction), size(prediction)[1])
    prediction_shape = size(prediction)
    target_shape = size(target)
    if prediction_shape != target_shape
        error("GradValley: bce_loss: size/shape of prediction must be equal to the size/shape of target")
    end
    if size(weight) != prediction_shape && size(weight) != prediction_shape[2:end]
        error("GradValley: bce_loss: at least the sizes of dimension 2 to last dimension of weight must be equal to the sizes of dimension 2 to last dimension of prediction/target, otherwise, the size of weight must be equal to the size of prediction/target")
    end
    losses = zeros(prediction_shape)
    losses_derivative = zeros(prediction_shape)
    last_index = ndims(prediction)
    for cartesian_index in CartesianIndices(losses)
        prediction_value = prediction[cartesian_index]
        target_value = target[cartesian_index]
        if size(weight) == prediction_shape[2:last_index]
            weight_value = weight[CartesianIndex(Tuple(cartesian_index)[2:last_index])]
        else # size(weight) == prediction_shape
            weight_value = weight[cartesian_index]
        end
        # weight_value = weight[cartesian_index[1]] # cartesian_index[1] gives the batch index
        losses[cartesian_index] = -weight_value * (target_value * log(prediction_value) + (1 - target_value) * log(1 - prediction_value)) 
        losses_derivative[cartesian_index] = ( (-weight_value * target_value) / prediction_value ) - ( (-weight_value + target_value * weight_value) / (1 - prediction_value) )
    end
    if isnothing(reduction_method)
        loss = losses
        loss_derivative = losses_derivative
    elseif reduction_method == "mean"
        loss = mean(losses)
        loss_derivative = (1 / length(prediction)) .* losses_derivative
    elseif reduction_method == "sum"
        loss = sum(losses)
        loss_derivative = losses_derivative
    else
        error("""GradValley: bce_loss: given reduction_method is invalid
            reduction_method must be either nothing, "mean" or "sum" """)
    end
    if return_derivative
        return loss, loss_derivative
    else
        return loss
    end
end

end # end of module "Optimization"