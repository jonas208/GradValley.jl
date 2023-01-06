#=
Fully connected: Forward & Backward
=#

# Forward propagation for a fully connected layer (Fc)
function fc_forward(inputs::Array{Float64, 2}, weights::Array{Float64, 2}, bias::Vector{Float64}, use_bias::Bool)
    # return rand(size(inputs)[1], size(weights)[1])
    current_batch_size = size(inputs)[1]
    out_features, in_features = size(weights)
    outputs = Array{Float64}(undef, current_batch_size, out_features)
    #=
    # for index_batch in 1:current_batch_size
    @inbounds @views Threads.@threads for index_batch in 1:current_batch_size
        # outputs[index_batch, :] = weights * inputs[index_batch, :]
        # println(size(outputs[index_batch, :]), typeof(outputs[index_batch, :]))
        # println(size(weights * inputs[index_batch, :]), typeof(weights * inputs[index_batch, :]))
        # println(size(outputs), typeof(outputs))
        # exit()
        if use_bias
            outputs[index_batch, :] = weights * inputs[index_batch, :] + bias # .+
        else
            outputs[index_batch, :] = weights * inputs[index_batch, :]
        end
    end
    =#
    # NOCH NICHT GEGEN GETESTET
    if !use_bias
        bias = zeros(out_features)
    end
    @turbo for index_batch in 1:current_batch_size
        for out_feature in 1:out_features
            output_value = 0.00
            for in_feature in 1:in_features
                output_value += inputs[index_batch, in_feature] * weights[out_feature, in_feature]
            end
            outputs[index_batch, out_feature] = output_value + bias[out_feature]
        end
    end

    return outputs
end

# Functions used for Backpropagation (Fc)
# The only input each function takes is an instance of a fc layer struct (Fc)
# Because a layer is given, these functions directly work on the hole batch

# Backpropagation of a fully connected layer, the resuls are the losses for the previous layer
function fc_losses(fc_layer)
    #= OLD VERSION WITH MATRIX MULTIPLICATION
    # return rand(size(fc_layer.inputs)[1], size(fc_layer.inputs)[2])
    current_batch_size = size(fc_layer.inputs)[1]
    out_features, in_features = size(fc_layer.weights)
    losses = Array{Float64}(undef, current_batch_size, in_features)
    # fc_layer.losses .*= fc_layer.activation_function(fc_layer.outputs_no_activation) # maybe . (dot) is not the right way of doing an element-wise multiplication)
    if fc_layer.df != 1
        out_losses = fc_layer.losses .* fc_layer.df(fc_layer.outputs_no_activation)
    else
        out_losses = fc_layer.losses
    end
    weights_transposed = fc_layer.weights'
    # println("Size transposed weights: ", size(weights_transposed))
    # println("Size original weights: ", size(fc_layer.weights))
    # for index_batch in 1:current_batch_size
    @inbounds @views Threads.@threads for index_batch in 1:current_batch_size
        # losses[index_batch, :] = weights_transposed * fc_losses[index_batch, :]
        losses[index_batch, :] = weights_transposed * out_losses[index_batch, :]
    end
    =#
    out_losses = fc_layer.losses
    weights = fc_layer.weights
    inputs = fc_layer.inputs
    if fc_layer.df != 1
        out_losses = out_losses .* fc_layer.df(fc_layer.outputs_no_activation)
    end
    current_batch_size = size(inputs)[1]
    out_features::Int, in_features::Int = size(weights)
    losses = zeros(current_batch_size, in_features)

    @turbo for index_batch in 1:current_batch_size # @tturbo
        for in_feature in 1:in_features
            losses_value = 0.00
            for out_feature in 1:out_features
                losses_value += weights[out_feature, in_feature] * out_losses[index_batch, out_feature]
            end
            losses[index_batch, in_feature] = losses_value
        end
    end

    return losses
end

# Computes the derivative of the weights/bias on the given layer, the results are used to optimize the weights/bias
function fc_gradients(fc_layer)
    #= OLD VERSION WITH MATRIX MULTIPLICATION
    # return rand(size(fc_layer.weights)...)
    current_batch_size = size(fc_layer.inputs)[1]
    out_features, in_features = size(fc_layer.weights)
    # gradients = Array{Float64}(undef, out_features, in_features)
    gradients = zeros(out_features, in_features)
    ## gradients = fc_layer.gradients
    bias_gradients = zeros(out_features)
    # bias_gradients = fc_layer.bias_gradients
    if fc_layer.df != 1
        # df = fc_layer.losses .* fc_layer.df(fc_layer.outputs_no_activation)
        df = fc_layer.df(fc_layer.outputs_no_activation)
    else
        df = ones(current_batch_size, out_features)
    end
    out_losses = fc_layer.losses
    inputs = fc_layer.inputs
    # for index_batch in 1:current_batch_size
    @inbounds @views Threads.@threads for index_batch in 1:current_batch_size
        single_out_losses_vector = out_losses[index_batch, :]
        single_out_losses = reshape(single_out_losses_vector, out_features, 1)
        single_inputs = reshape(inputs[index_batch, :], in_features, 1)
        gradients += (single_out_losses .* df[index_batch, :]) * inputs[index_batch, :]'
        bias_gradients += single_out_losses_vector .* df[index_batch, :]
    end
    =#
    # NOCH NICHT GESTESTET
    out_losses = fc_layer.losses
    weights = fc_layer.weights
    inputs = fc_layer.inputs
    if fc_layer.df != 1
        out_losses = out_losses .* fc_layer.df(fc_layer.outputs_no_activation)
    end
    current_batch_size = size(inputs)[1]
    out_features::Int, in_features::Int = size(weights)
    gradients = zeros(out_features, in_features)

    @turbo for index_batch in 1:current_batch_size # @tturbo
        for in_feature in 1:in_features
            for out_feature in 1:out_features
                gradients[out_feature, in_feature] += inputs[index_batch, in_feature] * out_losses[index_batch, out_feature]
            end
        end
    end
    bias_gradients = sum(out_losses, dims=1)[1, :]

    return gradients, bias_gradients
end

function fc_backward(fc_layer)
    out_losses = fc_layer.losses
    inputs = fc_layer.inputs
    weights = fc_layer.weights
    if fc_layer.df != 1
        out_losses = out_losses .* fc_layer.df(fc_layer.outputs_no_activation)
    end
    # weights_transposed = weights'
    current_batch_size = size(inputs)[1]
    out_features::Int, in_features::Int = size(weights)
    gradients = zeros(out_features, in_features)
    # bias_gradients = zeros(out_features)
    losses = zeros(current_batch_size, in_features)

    @turbo for index_batch in 1:current_batch_size # @tturbo
        for in_feature in 1:in_features
            losses_value = 0.00
            for out_feature in 1:out_features
                losses_value += weights[out_feature, in_feature] * out_losses[index_batch, out_feature]
                gradients[out_feature, in_feature] += inputs[index_batch, in_feature] * out_losses[index_batch, out_feature]
                # gradients[out_feature, in_feature] += inputs[index_batch, out_feature] * out_losses[index_batch, out_feature]
            end
            losses[index_batch, in_feature] = losses_value
        end
    end
    bias_gradients = sum(out_losses, dims=1)[1, :]
    # println(size(bias_gradients))

    return gradients, bias_gradients, losses
end