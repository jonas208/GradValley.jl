function batch_norm2d_calculate_variance(channels_over_batch::AbstractArray{T, 3}, mean::T) where {T <: Real}
    variance = zero(T)
    for value in channels_over_batch
        variance += (value - mean)^2
    end
    variance::T /= T(length(channels_over_batch))

    return variance
end

function batch_norm2d_forward!(output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight_gamma::AbstractVector{T}, weight_beta::AbstractVector{T}, track_running_stats::Bool, running_mean::AbstractVector{T}, running_variance::AbstractVector{T}, test_mode::Bool; momentum::T=T(0.1), epsilon::T=T(1e-5)) where {T <: Real}
    width, height, channels, current_batch_size = size(input)
    output_no_weights_applied = similar(output)
    
    if !test_mode # executed only in training mode
        current_mean = zeros(T, channels)
        current_variance = zeros(T, channels)
        #=
        @inbounds @views Threads.@threads for channel in 1:channels
            channels_over_batch = input[:, :, channel, :]
            mean = sum(channels_over_batch) / length(channels_over_batch)
            variance = batch_norm2d_calculate_variance(channels_over_batch, mean)
            output_channels_over_batch = (channels_over_batch .- mean) / sqrt(variance + epsilon) # - epsilon
            output_no_weights_applied[:, :, channel, :] = output_channels_over_batch
            output_channels_over_batch = weight_gamma[channel] * output_channels_over_batch .+ weight_beta[channel]
            output[:, :, channel, :] = output_channels_over_batch
            current_mean[channel] = mean
            current_variance[channel] = variance
        end
        =#
        n = length(input) / channels
        @inbounds Threads.@threads for channel in 1:channels
            s = zero(T) # 0.00
            @turbo for index_batch in 1:current_batch_size, y in 1:height, x in 1:width
                s += input[x, y, channel, index_batch]
            end
            mean = s / n
            variance = zero(T) # 0.00
            @turbo for index_batch in 1:current_batch_size, y in 1:height, x in 1:width
                variance += (input[x, y, channel, index_batch] - mean)^2
            end
            variance /= n
            current_mean[channel] = mean 
            current_variance[channel] = variance
            root = sqrt(variance + epsilon)
            @turbo for index_batch in 1:current_batch_size, y in 1:height, x in 1:width
                out_value = (input[x, y, channel, index_batch] - mean) / root
                output_no_weights_applied[x, y, channel, index_batch] = out_value
                output[x, y, channel, index_batch] = weight_gamma[channel] * out_value + weight_beta[channel]
            end
        end
        if track_running_stats
            running_mean = (1 - momentum) * running_mean .+ momentum * current_mean
            running_variance = (1 - momentum) * running_variance .+ momentum * current_variance
        end
    else # executed only in test mode
        @inbounds @views Threads.@threads for channel in 1:channels
            channels_over_batch = input[:, :, channel, :]
            if !track_running_stats # uses batch statistics when no running estimates were tracked during training
                mean = sum(channels_over_batch) / length(channels_over_batch)
                variance = batch_norm2d_calculate_variance(channels_over_batch, mean)
            else # using the during training computed running estimates
                mean = running_mean[channel]
                variance = running_variance[channel]
            end
            output_channels_over_batch = (channels_over_batch .- mean) / sqrt(variance + epsilon) # - epsilon
            output_no_weights_applied[:, :, channel, :] = output_channels_over_batch
            output_channels_over_batch = weight_gamma[channel] * output_channels_over_batch .+ weight_beta[channel]
            output[:, :, channel, :] = output_channels_over_batch
        end
    end

    return output, running_mean, running_variance
end

function batch_norm2d_forward(input::AbstractArray{T, 4}, weight_gamma::AbstractVector{T}, weight_beta::AbstractVector{T}, track_running_stats::Bool, running_mean::AbstractVector{T}, running_variance::AbstractVector{T}, test_mode::Bool; momentum::T=T(0.1), epsilon::T=T(1e-5)) where {T <: Real}
    width, height, channels, current_batch_size = size(input)
    output = zeros(T, width, height, channels, current_batch_size)
    return batch_norm2d_forward!(output, input, weight_gamma, weight_beta, track_running_stats, running_mean, running_variance, test_mode, momentum=momentum, epsilon=epsilon)
end

function batch_norm2d_data_backward!(input_gradient::AbstractArray{T, 4}, output_gradient::AbstractArray{T, 4}, output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight_gamma::AbstractVector{T}, weight_beta::AbstractVector{T}, track_running_stats::Bool, running_mean::AbstractVector{T}, running_variance::AbstractVector{T}, test_mode::Bool; epsilon::T=T(1e-5)) where {T <: Real}
    width, height, channels, current_batch_size = size(output_gradient)

    output_no_weights_applied = similar(output)
    for index_batch in 1:current_batch_size, channel in 1:channels, y in 1:height, x in 1:width # @turbo 
        out_value = output[x, y, channel, index_batch]
        output_no_weights_applied[x, y, channel, index_batch] = (out_value - weight_beta[channel]) / weight_gamma[channel]
    end
    # output_no_weights_applied = output_no_weight_applied

    #=
    @inbounds @views Threads.@threads for channel in 1:channels
        channels_over_batch = input[:, :, channel, :]
        num_values = length(channels_over_batch)
        mean = sum(channels_over_batch) / length(channels_over_batch)
        variance = batch_norm2d_calculate_variance(channels_over_batch, mean)
        # mean = running_mean[channel]
        # variance = running_variance[channel]
        denominator = sqrt(variance + epsilon)
        xhat_batch = output_no_weights_applied[:, :, channel, :] # xhat are the output without weights applied
        dxhat_batch = output_gradient[:, :, channel, :] * weight_gamma[channel] # derivative of output_gradient with respect to output without weights applied
        input_gradient[:, :, channel, :] = ((num_values * dxhat_batch) .- sum(dxhat_batch) .- (xhat_batch * sum(dxhat_batch .* xhat_batch))) / (num_values * denominator)
    end
    =#
    for channel in 1:channels # @inbounds @views Threads.@threads 
        channels_over_batch = input[:, :, channel, :]
        num_values = length(channels_over_batch)
        if test_mode && track_running_stats
            mean = running_mean[channel]
            variance = running_variance[channel]
        else
            mean = sum(channels_over_batch) / length(channels_over_batch)
            variance = batch_norm2d_calculate_variance(channels_over_batch, mean)
        end
        # mean = sum(channels_over_batch) / length(channels_over_batch)
        # variance = batch_norm2d_calculate_variance(channels_over_batch, mean)
        denominator = sqrt(variance + epsilon)
        xhat_batch = output_no_weights_applied[:, :, channel, :] # xhat are the output without weights applied
        dxhat_batch = output_gradient[:, :, channel, :] * weight_gamma[channel] # derivative of output_gradient with respect to output without weights applied
        input_gradient[:, :, channel, :] = ((num_values * dxhat_batch) .- sum(dxhat_batch) .- (xhat_batch * sum(dxhat_batch .* xhat_batch))) / (num_values * denominator)
    end

    return input_gradient
end

function batch_norm2d_data_backward(output_gradient::AbstractArray{T, 4}, output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight_gamma::AbstractVector{T}, weight_beta::AbstractVector{T}, track_running_stats::Bool, running_mean::AbstractVector{T}, running_variance::AbstractVector{T}, test_mode::Bool; epsilon::T=T(1e-5)) where {T <: Real}
    input_gradient = zeros(T, size(input)...) # similar(output_gradient)
    return batch_norm2d_data_backward!(input_gradient, output_gradient, output, input, weight_gamma, weight_beta, track_running_stats, running_mean, running_variance, test_mode, epsilon=epsilon)
end

function batch_norm2d_weight_backward!(weight_gamma_gradient::AbstractVector{T}, weight_beta_gradient::AbstractVector{T}, output_gradient::AbstractArray{T, 4}, output::AbstractArray{T, 4}, weight_gamma::AbstractVector{T}, weight_beta::AbstractVector{T}) where {T <: Real}
    channels = length(weight_gamma_gradient)

    @inbounds @views Threads.@threads for channel in 1:channels
        weight_beta_gradient[channel] = sum(output_gradient[:, :, channel, :])
        channel_output = output[:, :, channel, :]
        channel_output_no_weights_applied = (channel_output .- weight_beta[channel]) / weight_gamma[channel]
        weight_gamma_gradient[channel] = sum(output_gradient[:, :, channel, :] .* channel_output_no_weights_applied)
    end
    
    return weight_gamma_gradient, weight_beta_gradient
end

function batch_norm2d_weight_backward(output_gradient::AbstractArray{T, 4}, output::AbstractArray{T, 4}, weight_gamma::AbstractVector{T}, weight_beta::AbstractVector{T}) where {T <: Real}
    weight_gamma_gradient = similar(weight_gamma)
    weight_beta_gradient = similar(weight_beta)
    
    return batch_norm2d_weight_backward!(weight_gamma_gradient, weight_beta_gradient, output_gradient, output, weight_gamma, weight_beta)
end

function batch_norm2d_backward!(input_gradient::AbstractArray{T, 4}, weight_gamma_gradient::AbstractVector{T}, weight_beta_gradient::AbstractVector{T}, output_gradient::AbstractArray{T, 4}, output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight_gamma::AbstractVector{T}, weight_beta::AbstractVector{T}, track_running_stats::Bool, running_mean::AbstractVector{T}, running_variance::AbstractVector{T}, test_mode::Bool; epsilon::T=T(1e-5)) where {T <: Real}
    input_gradient = batch_norm2d_data_backward!(input_gradient, output_gradient, output, input, weight_gamma, weight_beta, track_running_stats, running_mean, running_variance, test_mode, epsilon=epsilon)
    weight_gamma_gradient, weight_beta_gradient = batch_norm2d_weight_backward!(weight_gamma_gradient, weight_beta_gradient, output_gradient, output, weight_gamma, weight_beta)
    
    return input_gradient, weight_gamma_gradient, weight_beta_gradient
end

function batch_norm2d_backward(output_gradient::AbstractArray{T, 4}, output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, weight_gamma::AbstractVector{T}, weight_beta::AbstractVector{T}, track_running_stats::Bool, running_mean::AbstractVector{T}, running_variance::AbstractVector{T}, test_mode::Bool; epsilon::T=T(1e-5)) where {T <: Real}
    input_gradient = batch_norm2d_data_backward(output_gradient, output, input, weight_gamma, weight_beta, track_running_stats, running_mean, running_variance, test_mode, epsilon=epsilon)
    weight_gamma_gradient, weight_beta_gradient = batch_norm2d_weight_backward(output_gradient, output, weight_gamma, weight_beta)
    
    return input_gradient, weight_gamma_gradient, weight_beta_gradient
end