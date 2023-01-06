module Optimization

using ..Layers: SequentialContainer, Conv, DepthwiseConv, BatchNorm2d, Fc
using ..Functional
# make Functional accessible via gv functional
gv_functional = Functional

# export all optimizers and their step! methods
export SGD, MSGD, Nesterov, step!
# export all loss functions
export mse_loss, mae_loss, cross_entropy_loss, cross_entropy_softmax_included_loss

#=
Internal functions, Internals
=#

# extracts all layers in a stack of (nested) Sequential Containers and layers recursively, returns a vector only containing all the pure layers
function extract_layers(sc::SequentialContainer, layer_stack)
    # println("called extract_layers")
    for layer in sc.layer_stack
        if typeof(layer) == SequentialContainer
            extract_layers(layer, layer_stack)
        else
            push!(layer_stack, layer)
        end
    end

    return layer_stack
end

# computes the mean of an arbitrary sized input array
function mean(x)
    return sum(x) / length(x)
end

# implementation of stochastic gradient descent optimization algorithm (including weight decay and dampening)
mutable struct SGD
    layer_stack::Vector
    learning_rate::Real
    weight_decay::Union{Nothing, Real}
    dampening::Real
    maximize::Bool
    modified_weight_gradients::Vector
    modified_bias_gradients::Vector
    iteration_counter::Int
    # custom constructor
    # function SGD(layer_stack::Union{Vector, SequentialContainer}, learning_rate::AbstractFloat; momentum::AbstractFloat=0.90, weight_decay::Union{Nothing, AbstractFloat}=nothing, dampening::AbstractFloat=0.00, maximize::Bool=false) # ::Real
    function SGD(layer_stack, learning_rate::AbstractFloat; weight_decay=nothing, dampening=0.00, maximize=false) # ::Real
        if typeof(layer_stack) == SequentialContainer
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
            weight_decay,
            dampening,
            maximize,
            modified_weight_gradients,
            modified_bias_gradients,
            0)
    end
end

# implementation of stochastic gradient descent with momentum optimization algorithm (including weight decay and dampening)
mutable struct MSGD
    layer_stack::Vector
    learning_rate::Real
    momentum::Real
    weight_decay::Union{Nothing, Real}
    dampening::Real
    maximize::Bool
    modified_weight_gradients::Vector
    modified_bias_gradients::Vector
    iteration_counter::Int
    # custom constructor
    # function MSGD(layer_stack::Union{Vector, SequentialContainer}, learning_rate::AbstractFloat; momentum::AbstractFloat=0.90, weight_decay::Union{Nothing, AbstractFloat}=nothing, dampening::AbstractFloat=0.00, maximize::Bool=false) # ::Real
    function MSGD(layer_stack, learning_rate::AbstractFloat; momentum=0.90, weight_decay=nothing, dampening=0.00, maximize=false) # ::Real
        if typeof(layer_stack) == SequentialContainer
            # println("something")
            # layer_stack = layer_stack.layer_stack
            layer_stack = extract_layers(layer_stack, [])
            # println(layer_stack)
            # println(length(layer_stack))
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

# implementation of nesterov optimization algorithm (including weight decay and dampening)
mutable struct Nesterov
    layer_stack::Vector
    learning_rate::Real
    momentum::Real
    weight_decay::Union{Nothing, Real}
    dampening::Real
    maximize::Bool
    modified_weight_gradients::Vector
    modified_bias_gradients::Vector
    iteration_counter::Int
    # custom constructor
    # function Nesterov(layer_stack::Union{Vector, SequentialContainer}, learning_rate::AbstractFloat; momentum::AbstractFloat=0.90, weight_decay::Union{Nothing, AbstractFloat}=nothing, dampening::AbstractFloat=0.00, maximize::Bool=false) # ::Real
    function Nesterov(layer_stack, learning_rate::AbstractFloat; momentum=0.90, weight_decay=nothing, dampening=0.00, maximize=false) # ::Real
        if typeof(layer_stack) == SequentialContainer
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

# updates the weights with the hyperparameters defined in the given optimizer of type MSGD
function step!(optimizer::Union{SGD, MSGD, Nesterov})
    if typeof(optimizer) == SGD
        momentum = 0
    else
        momentum = optimizer.momentum
    end
    for (layer_index, layer) in enumerate(optimizer.layer_stack)
        if typeof(layer) == Conv || typeof(layer) == DepthwiseConv
            # println("Conv")
            weight = layer.kernels
            bias = layer.bias
            weight_gradient = layer.gradients
            bias_gradient = layer.bias_gradients
        elseif typeof(layer) == Fc
            # println("Fc")
            weight = layer.weights
            bias = layer.bias
            weight_gradient = layer.gradients
            bias_gradient = layer.bias_gradients
        elseif typeof(layer) == BatchNorm2d
            # println("BatchNorm2d")
            weight = layer.weight_gamma
            bias = layer.weight_beta
            weight_gradient = layer.gradient_gamma
            bias_gradient = layer.gradient_beta
        else
            continue
        end

        if !isnothing(optimizer.weight_decay)
            weight_gradient = weight_gradient .+ optimizer.weight_decay * weight
            bias_gradient = bias_gradient .+ optimizer.weight_decay * bias
        end

        if optimizer.iteration_counter > 1
            # println("something")
            # println(size(optimizer.modified_weight_gradients[layer_index]))
            # println(sum(optimizer.modified_weight_gradients[layer_index]))
            # println(sum(weight_gradient))
            #=
            weight_gradient1 = optimizer.momentum * optimizer.modified_weight_gradients[layer_index] .+ (1 - optimizer.dampening) * weight_gradient
            bias_gradient1 = optimizer.momentum * optimizer.modified_bias_gradients[layer_index] .+ (1 - optimizer.dampening) * bias_gradient
            weight_gradient2 = optimizer.momentum * optimizer.modified_weight_gradients[layer_index] + (1 - optimizer.dampening) * weight_gradient
            bias_gradient2 = optimizer.momentum * optimizer.modified_bias_gradients[layer_index] + (1 - optimizer.dampening) * bias_gradient
            println(isapprox(weight_gradient1, weight_gradient2))
            println(isapprox(bias_gradient1, bias_gradient2))
            =#
            modified_weight_gradient = momentum * optimizer.modified_weight_gradients[layer_index] + (1 - optimizer.dampening) * weight_gradient # optimizer.momentum
            modified_bias_gradient = momentum * optimizer.modified_bias_gradients[layer_index] + (1 - optimizer.dampening) * bias_gradient # optimizer.momentum
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

        if typeof(layer) == Conv || typeof(layer) == DepthwiseConv
            # println(size(layer.kernels))
            # println(size(layer.gradients))
            if optimizer.maximize
                layer.kernels += optimizer.learning_rate * weight_gradient # .+
                layer.bias += optimizer.learning_rate * bias_gradient # .+
            else
                # o_lk = copy(layer.kernels)
                layer.kernels -= optimizer.learning_rate * weight_gradient # .-
                layer.bias -= optimizer.learning_rate * bias_gradient # .-
                # k2 = o_lk - (optimizer.learning_rate * weight_gradient) # .-
                # println("ultitest: $(isapprox(layer.kernels, k2))")
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

        optimizer.modified_weight_gradients[layer_index] = modified_weight_gradient # * optimizer.learning_rate -> WRONG
        optimizer.modified_bias_gradients[layer_index] = modified_bias_gradient # * optimizer.learning_rate -> WRONG
    end
    optimizer.iteration_counter += 1
end

# mean absolute loss (l1loss), size/shape of prediction and target must be equal
# size/shape of prediction/target: (*) -> where * means any number of dimensions
function mae_loss(prediction, target; reduction_method::Union{String, Nothing}="mean", return_derivative::Bool=true)
    prediction_shape = size(prediction)
    target_shape = size(target)
    if prediction_shape != target_shape
        error("GradValley: mse_loss: size/shape of prediction must be equal to the size/shape of target")
    end
    losses = abs.(prediction - target)
    if isnothing(reduction_method)
        loss = losses
        loss_derivative = (prediction - target) ./ losses # losses = abs.(prediction - target)
    elseif reduction_method == "mean"
        loss = mean(losses)
        # current_batch_size = prediction_shape[1]
        # loss_derivative = (1 / (length(prediction) / current_batch_size)) .* ((prediction - target) / losses)
        loss_derivative = (1 / length(prediction)) .* ((prediction - target) ./ losses) # losses = abs.(prediction - target)
    elseif reduction_method == "sum"
        loss = sum(losses)
        loss_derivative = (prediction - target) ./ losses # losses = abs.(prediction - target)
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
function mse_loss(prediction, target; reduction_method::Union{String, Nothing}="mean", return_derivative::Bool=true)
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
        # loss_derivative = ((1 / (length(prediction) / prediction_shape[1])) * 2) .* (prediction - target)
        ## current_batch_size = prediction_shape[1]
        ## loss_derivative = ((1 / (length(prediction) / current_batch_size)) * 2) .* (prediction - target)
        loss_derivative = ((1 / length(prediction)) * 2) .* (prediction - target)
    elseif reduction_method == "sum"
        loss = sum(losses)
        # current_batch_size = prediction_shape[1]
        # loss_derivative = current_batch_size * 2 * (prediction - target)
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

end # end of module "Optimization"