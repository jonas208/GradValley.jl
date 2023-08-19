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
julia> input = rand(1000, 32)
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
    PT = eltype(prediction)
    if prediction_shape != target_shape
        error("GradValley: mae_loss: size/shape of prediction must be equal to the size/shape of target")
    end
    losses = abs.(prediction - target)
    if isnothing(reduction_method)
        loss = losses
        loss_derivative = (prediction - target) ./ losses
    elseif reduction_method == "mean"
        loss = mean(losses)
        loss_derivative = PT((1 / length(prediction))) .* ((prediction - target) ./ losses)
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
julia> input = rand(1000, 32)
# compute the output of the model
julia> output = forward(model, input)
# generate some random target values 
julia> target = rand(size(output)...)
# compute the loss and it's derivative (with default reduction method "mean")
julia> loss, loss_derivative = mse_loss(output, target)
# compute the gradients 
julia> backward(model, loss_derivative)
```
"""
function mse_loss(prediction::AbstractArray{<: Real, N}, target::AbstractArray{<: Real, N}; reduction_method::Union{AbstractString, Nothing}="mean", return_derivative::Bool=true) where N
    prediction_shape = size(prediction)
    target_shape = size(target)
    PT = eltype(prediction)
    if prediction_shape != target_shape
        error("GradValley: mse_loss: size/shape of prediction must be equal to the size/shape of target")
    end
    losses = (prediction - target).^2
    if isnothing(reduction_method)
        loss = losses
        loss_derivative = 2 * (prediction - target)
    elseif reduction_method == "mean"
        loss = mean(losses)
        loss_derivative = PT(((1 / length(prediction))) * 2) .* (prediction - target)
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
    bce_loss(prediction::AbstractArray{<: Real, N}, target::AbstractArray{<: Real, N}; epsilon::Real=eps(eltype(prediction)), reduction_method::Union{AbstractString, Nothing}="mean", return_derivative::Bool=true) where N

Calculate the binary cross entropy loss (with optional reduction to a single loss value (mean or sum)) and it's derivative with respect to the prediction input.

# Arguments
- `prediction::AbstractArray{<: Real, N}`: the prediction of the model of shape (*), where * means any number of dimensions 
- `target::AbstractArray{<: Real, N}`: the corresponding target values (should be between 0 and 1) of shape (*), must have the same shape as the prediction input 
- `epsilon::Real=eps(eltype(prediction))`: term to avoid infinity
- `reduction_method::Union{AbstractString, Nothing}="mean"`: can be `"mean"`, `"sum"` or `nothing`, specifies the reduction method which reduces the element-wise computed loss to a single value
- `return_derivative::Bool=true`: it true, the loss and it's derivative with respect to the prediction input is returned, if false, just the loss will be returned

# Definition
``L`` is the loss value which will be returned. If `return_derivative` is true, then an array with the same shape as prediction/target is returned as well, it contains the partial derivatives of ``L`` w.r.t. to each prediction value: ``\frac{\partial L}{\partial p_i}``, where ``p_i`` in one prediction value.
If `reduction_method` is `nothing`, the element-wise computed losses are returned. Note that for `reduction_method=nothing`, the derivative is just the same as when `reduction_method="sum"`. ``w_i`` is one rescaling weight value.
The element-wise calculation can be defined as (where ``t_i`` is one target value and ``l_i`` is one loss value): 
```math
\begin{align*}
l_i = -t_i \cdot \log(p_i + \epsilon) - (1 - t_i) \cdot \log(1 - p_i + \epsilon)
\end{align*}
```
Then, ``L`` and ``\frac{\partial L}{\partial p_i}`` differ a little bit from case to case (``n`` is the number of values in `prediction`/`target`):
```math
\begin{align*}
L;\frac{\partial L}{\partial p_i} = \begin{cases}\frac{1}{n}\sum_{i=1}^{n} l_i; \frac{1}{n}(\frac{-t_i}{p_i + \epsilon} - \frac{t_i - 1}{1 - p_i + \epsilon}) &\text{for reduction\_method="mean"}\\\sum_{i=1}^{n} l_i;  &\text{for reduction\_method="sum"}\end{cases}
\end{align*}
```

# Examples
```julia-repl
# define a model
julia> model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# create some random input data
julia> input = rand(1000, 32)
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
function bce_loss(prediction::AbstractArray{<: Real, N}, target::AbstractArray{<: Real, N}; epsilon::Real=eps(eltype(prediction)), reduction_method::Union{AbstractString, Nothing}="mean", return_derivative::Bool=true) where N
    # weight::Vector{Real}=ones(eltype(prediction), size(prediction)[1])
    prediction_shape = size(prediction)
    target_shape = size(target)
    PT = eltype(prediction)
    if prediction_shape != target_shape
        error("GradValley: bce_loss: size/shape of prediction must be equal to the size/shape of target")
    end

    losses = @.(-target * log(prediction + epsilon) - (1 - target) * log(1 - prediction + epsilon))
    # losses_derivative = @.( ( -target / prediction ) - ( (target - 1) / (1 - prediction) ) )
    losses_derivative = @.( ( -target / (prediction + epsilon) ) - ( (target - 1) / (1 - prediction + epsilon) ) )

    if isnothing(reduction_method)
        loss = losses
        loss_derivative = losses_derivative
    elseif reduction_method == "mean"
        loss = mean(losses)
        loss_derivative = PT((1 / length(prediction))) .* losses_derivative
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