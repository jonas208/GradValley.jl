#=
Batch normalization for 4d-inputs (2d batch normalization): Forward & Backward
=#

# calculates the variance of a 3-dimensional array, the mean must be given (necessary for batch normalization)
# function batchNorm2d_calculate_variance(channels_over_batch::Array{Float64, 3}, mean::Float64)
function batchNorm2d_calculate_variance(channels_over_batch::AbstractArray{Float64, 3}, mean::Float64)
    variance = 0.00
    for value in channels_over_batch
        variance += (value - mean)^2
    end
    variance /= length(channels_over_batch) # (1 / length(channels_over_batch)) * 

    return variance
end

# Performing a multichannel batch normalization (on a hole batch)
# Shape of input: (batch_size, channels, height, width)
# Shape of weight_gamma/weight_beta/running_mean/running_variance: (channels, )
function batchNorm2d_forward(inputs::Array{Float64, 4}, weight_gamma::Vector{Float64}, weight_beta::Vector{Float64}, track_running_stats::Bool, momentum::Float64, running_mean::Vector{Float64}, running_variance::Vector{Float64}, test_mode::Bool; epsilon::Float64=1e-5) # ::Array{Float64, 1}
    current_batch_size::Int, channels::Int, height::Int, width::Int = size(inputs)
    outputs = Array{Float64, 4}(undef, current_batch_size, channels, height, width)
    outputs_no_weights_applied = Array{Float64, 4}(undef, current_batch_size, channels, height, width)
    
    if !test_mode # executed only in training mode
        current_mean = Vector{Float64}(undef, channels)
        current_variance = Vector{Float64}(undef, channels)
        #=
        # for channel in 1:channels
        @inbounds @views Threads.@threads for channel in 1:channels
            channels_over_batch = inputs[:, channel, :, :]
            mean = sum(channels_over_batch) / length(channels_over_batch)
            variance = batchNorm2d_calculate_variance(channels_over_batch, mean)
            output_channels_over_batch = (channels_over_batch .- mean) / sqrt(variance + epsilon) # - epsilon
            outputs_no_weights_applied[:, channel, :, :] = output_channels_over_batch
            output_channels_over_batch = weight_gamma[channel] * output_channels_over_batch .+ weight_beta[channel]
            outputs[:, channel, :, :] = output_channels_over_batch
            current_mean[channel] = mean
            current_variance[channel] = variance
        end
        =#
        n = length(inputs) / channels
        @inbounds Threads.@threads for channel in 1:channels
            s = 0.00
            @turbo for index_batch in 1:current_batch_size, y in 1:height, x in 1:width
                s += inputs[index_batch, channel, y, x]
            end
            mean = s / n
            variance = 0.00
            @turbo for index_batch in 1:current_batch_size, y in 1:height, x in 1:width
                variance += (inputs[index_batch, channel, y, x] - mean)^2
            end
            variance /= n
            current_mean[channel] = mean 
            current_variance[channel] = variance
            root = sqrt(variance + epsilon)
            @turbo for index_batch in 1:current_batch_size, y in 1:height, x in 1:width
                out_value = (inputs[index_batch, channel, y, x] - mean) / root
                outputs[index_batch, channel, y, x] = weight_gamma[channel] * out_value + weight_beta[channel]
                outputs_no_weights_applied[index_batch, channel, y, x] = out_value
            end
        end
        if track_running_stats
            running_mean = (1 - momentum) * running_mean .+ momentum * current_mean
            running_variance = (1 - momentum) * running_variance .+ momentum * current_variance
        end
    else # executed only in test mode
        # @inbounds @views @batch per=core for channel in 1:channels
        for channel in 1:channels
             channels_over_batch = inputs[:, channel, :, :]
            if !track_running_stats # uses batch statistics when no running estimates were tracked during training
                mean = sum(channels_over_batch) / length(channels_over_batch)
                variance = batchNorm2d_calculate_variance(channels_over_batch, mean)
            else # using the during training computed running estimates
                mean = running_mean[channel]
                variance = running_variance[channel]
            end
            output_channels_over_batch = (channels_over_batch .- mean) / sqrt(variance + epsilon) # - epsilon
            outputs_no_weights_applied[:, channel, :, :] = output_channels_over_batch
            output_channels_over_batch = weight_gamma[channel] * output_channels_over_batch .+ weight_beta[channel]
            outputs[:, channel, :, :] = output_channels_over_batch
        end
    end

    return outputs, running_mean, running_variance, outputs_no_weights_applied
end

# Functions used for Backpropagation (BatchNorm2d)
# The only input each function takes is an instance of a batch normalization layer struct (BatchNorm2d)
# Because a layer is given, these functions directly work on the hole batch

# Computes the derivative of the inputs on the given layer, the results are used as the losses for the previous layer
function batchNorm2d_losses(batchnorm_layer)
    current_batch_size, channels, height, width = size(batchnorm_layer.outputs)
    losses = Array{Float64, 4}(undef, current_batch_size, channels, height, width)

    #=
    if batchnorm_layer.df != 1
        out_losses = batchnorm_layer.losses .* batchnorm_layer.df(batchnorm_layer.outputs_no_activation)
    else
        out_losses = batchnorm_layer.losses
    end
    =#
    out_losses::Array{Float64, 4} = batchnorm_layer.losses
    if batchnorm_layer.df != 1
        # activation_function_gradients = batchnorm_layer.df(batchnorm_layer.outputs, batchnorm_layer.outputs_no_activation) # outputs becomes activation_function_gradients (for allocating as little as possible)
        # out_losses = activation_function_gradients .= (*).(out_losses, activation_function_gradients)
        out_losses = out_losses .* batchnorm_layer.df(batchnorm_layer.outputs_no_activation)
    end

    #=
    # for channel in 1:channels
    @inbounds @views Threads.@threads for channel in 1:channels
        channels_over_batch = batchnorm_layer.inputs[:, channel, :, :]
        num_values = length(channels_over_batch)
        mean = sum(channels_over_batch) / length(channels_over_batch)
        variance = batchNorm2d_calculate_variance(channels_over_batch, mean)
        # mean = batchnorm_layer.running_mean[channel]
        # variance = batchnorm_layer.running_variance[channel]
        denominator = sqrt(variance + batchnorm_layer.epsilon)
        xhat_batch = batchnorm_layer.outputs_no_weights_applied[:, channel, :, :] # xhat are the outputs without weights applied
        dxhat_batch = out_losses[:, channel, :, :] * batchnorm_layer.weight_gamma[channel] # derivative of out losses with respect to outputs without weights applied
        losses[:, channel, :, :] = ((num_values * dxhat_batch) .- sum(dxhat_batch) .- (xhat_batch * sum(dxhat_batch .* xhat_batch))) / (num_values * denominator)
    end
    =#
    # for channel in 1:channels
    @inbounds @views Threads.@threads for channel in 1:channels
        channels_over_batch = batchnorm_layer.inputs[:, channel, :, :]
        num_values = length(channels_over_batch)
        if batchnorm_layer.test_mode && batchnorm_layer.track_running_stats
            mean = batchnorm_layer.running_mean[channel]
            variance = batchnorm_layer.running_variance[channel]
        else
            mean = sum(channels_over_batch) / length(channels_over_batch)
            variance = batchNorm2d_calculate_variance(channels_over_batch, mean)
        end
        # mean = sum(channels_over_batch) / length(channels_over_batch)
        # variance = batchNorm2d_calculate_variance(channels_over_batch, mean)
        denominator = sqrt(variance + batchnorm_layer.epsilon)
        xhat_batch = batchnorm_layer.outputs_no_weights_applied[:, channel, :, :] # xhat are the outputs without weights applied
        dxhat_batch = out_losses[:, channel, :, :] * batchnorm_layer.weight_gamma[channel] # derivative of out losses with respect to outputs without weights applied
        losses[:, channel, :, :] = ((num_values * dxhat_batch) .- sum(dxhat_batch) .- (xhat_batch * sum(dxhat_batch .* xhat_batch))) / (num_values * denominator)
    end

    return losses
end

# Computes the derivative of the weights on the given layer, the results are used to optimize the weights
function batchNorm2d_gradients(batchnorm_layer)
    gradient_gamma = zeros(size(batchnorm_layer.weight_gamma))
    gradient_beta = zeros(size(batchnorm_layer.weight_beta)) # weight_gamma
    if !batchnorm_layer.affine
        return gradient_gamma, gradient_beta
    else
        channels = length(gradient_gamma)

        #=
        if batchnorm_layer.df != 1
            out_losses = batchnorm_layer.losses .* batchnorm_layer.df(batchnorm_layer.outputs_no_activation)
        else
            out_losses = batchnorm_layer.losses
        end
        =#
        out_losses::Array{Float64, 4} = batchnorm_layer.losses
        if batchnorm_layer.df != 1
            # activation_function_gradients = batchnorm_layer.df(batchnorm_layer.outputs, batchnorm_layer.outputs_no_activation) # outputs becomes activation_function_gradients (for allocating as little as possible)
            # out_losses = activation_function_gradients .= (*).(out_losses, activation_function_gradients)
            out_losses = out_losses .* batchnorm_layer.df(batchnorm_layer.outputs_no_activation)
        end
        
        # for channel in 1:channels
        @inbounds @views Threads.@threads for channel in 1:channels
            gradient_beta[channel] = sum(out_losses[:, channel, :, :])
            # channel_outputs_no_weights = (channel_outputs .- batchnorm_layer.weight_beta[channel]) / batchnorm_layer.weight_gamma[channel]
            channel_outputs_no_weights = batchnorm_layer.outputs_no_weights_applied[:, channel, :, :]
            gradient_gamma[channel] = sum(out_losses[:, channel, :, :] .* channel_outputs_no_weights)
        end
    end
    
    return gradient_gamma, gradient_beta
end