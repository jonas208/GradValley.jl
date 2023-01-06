#=
Weight initialization:
All functions take a tuple which is the shape of the returned weight, gain is also a necessary paramter
(choosen by the activation function, usually given by a layer struct)
=#

# calculates fan_mode for all types of weight initializations
function calculate_fan_mode(weight_shape::Tuple)
    if length(weight_shape) == 4 # Convolution layer (Conv)
        in_channels = weight_shape[2]
        out_channels = weight_shape[1]
        size_kernel = weight_shape[3] * weight_shape[4]
        fan_in = in_channels * size_kernel
        fan_out = out_channels * size_kernel
    elseif length(weight_shape) == 2 # Fully connected layer (Fc)
        fan_in = weight_shape[2]
        fan_out = weight_shape[1]
    else
        error("GradValley: calculate_fan_mode: invalid weight_shape")
    end

    return fan_in, fan_out
end

# default initialization (normal distribution)
function default_init(weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        std = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        std = gain / sqrt(fan_out)
    else
        error("GradValley: default_init: invalid fan_mode")
    end
    weight = randn(weight_shape) * std

    return weight
end

# default initialization (uniform distribution)
function default_uniform_init(weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        bound = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        bound = gain / sqrt(fan_out)
    else
        error("GradValley: default_uniform_init: invalid fan_mode")
    end
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    weight = rand(weight_shape...) * 2 * bound .- bound

    return weight
end

# kaiming (he) initialization (normal distribution)
function kaiming_init(weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        std = gain * sqrt(1 / fan_in)
    elseif fan_mode == "fan_out"
        std = gain * sqrt(1 / fan_out)
    else
        error("GradValley: kaiming_init: invalid fan_mode")
    end
    weight = randn(weight_shape) * std

    return weight
end

# xavier (glorot) initialization (normal distribution)
function xavier_init(weight_shape::Tuple, gain::Real)
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    std = gain * sqrt(2 / (fan_in + fan_out))
    weight = randn(weight_shape) * std

    return weight
end

# kaiming (he) initialization (uniform distribution)
function kaiming_uniform_init(weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        bound = gain * sqrt(3 / fan_in)
    elseif fan_mode == "fan_out"
        bound = gain * sqrt(3 / fan_out)
    else
        error("GradValley: kaiming_init: invalid fan_mode")
    end
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    weight = rand(weight_shape...) * 2 * bound .- bound

    return weight
end

# xavier (glorot) initialization (uniform distribution)
function xavier_uniform_init(weight_shape::Tuple, gain::Real)
    # weight = rand(weight_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    bound = gain * sqrt(6 / (fan_in + fan_out))
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    weight = rand(weight_shape...) * 2 * bound .- bound

    return weight
end

# bias initialization (normal distribution)
function bias_init(bias_shape::Tuple, weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # bias = rand(bias_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        std = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        std = gain / sqrt(fan_out)
    else
        error("GradValley: bias_normal_init: invalid fan_mode")
    end
    bias = randn(bias_shape) * std

    return bias
end

# bias initialization (uniform distribution)
function bias_uniform_init(bias_shape::Tuple, weight_shape::Tuple, gain::Real; fan_mode="fan_in")
    # bias = rand(bias_shape...) * gain
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        bound = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        bound = gain / sqrt(fan_out)
    else
        error("GradValley: bias_uniform_init: invalid fan_mode")
    end
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    bias = rand(bias_shape...) * 2 * bound .- bound

    return bias
end