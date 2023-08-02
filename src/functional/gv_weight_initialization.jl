# calculates fan_mode for all types of weight initializations
function calculate_fan_mode(weight_shape::NTuple{N, Int}) where N
    if length(weight_shape) == 4 # Convolution layer (Conv or ConvTranspose)
        in_channels = weight_shape[3]
        out_channels = weight_shape[4]
        size_kernel = weight_shape[1] * weight_shape[2]
        fan_in = in_channels * size_kernel
        fan_out = out_channels * size_kernel
    elseif length(weight_shape) == 2 # Fully connected layer (Fc)
        fan_in = weight_shape[2]
        fan_out = weight_shape[1]
    else
        @warn "GradValley: calculate_fan_mode: unknown weight_shape, couldn't identify a fully connected weight nor a convolution weight/kernel, 
               uses length of weight for both fan_in and fan_out instead"
        fan = prod(weight_shape)
        fan_in, fan_out = fan, fan
    end

    return fan_in, fan_out
end

# default initialization (normal distribution)
function default_init(weight_shape::NTuple{N, Int}, gain::Real; fan_mode::AbstractString="fan_in") where N
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        std = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        std = gain / sqrt(fan_out)
    else
        error("""GradValley: default_init: init: invalid fan_mode, fan_mode must be either "fan_in" or "fan_out" """)
    end
    weight = randn(weight_shape) * std

    return weight
end

# default initialization (uniform distribution)
function default_uniform_init(weight_shape::NTuple{N, Int}, gain::Real; fan_mode::AbstractString="fan_in") where N
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        bound = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        bound = gain / sqrt(fan_out)
    else
        error("""GradValley: default_uniform_init: init: invalid fan_mode, fan_mode must be either "fan_in" or "fan_out" """)
    end
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    weight = rand(weight_shape...) * 2 * bound .- bound

    return weight
end

# kaiming (he) initialization (normal distribution)
function kaiming_init(weight_shape::NTuple{N, Int}, gain::Real; fan_mode::AbstractString="fan_in") where N
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        std = gain * sqrt(1 / fan_in)
    elseif fan_mode == "fan_out"
        std = gain * sqrt(1 / fan_out)
    else
        error("""GradValley: kaiming_init: invalid fan_mode, fan_mode must be either "fan_in" or "fan_out" """)
    end
    weight = randn(weight_shape) * std

    return weight
end

# xavier (glorot) initialization (normal distribution)
function xavier_init(weight_shape::NTuple{N, Int}, gain::Real) where N
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    std = gain * sqrt(2 / (fan_in + fan_out))
    weight = randn(weight_shape) * std

    return weight
end

# kaiming (he) initialization (uniform distribution)
function kaiming_uniform_init(weight_shape::NTuple{N, Int}, gain::Real; fan_mode::AbstractString="fan_in") where N
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        bound = gain * sqrt(3 / fan_in)
    elseif fan_mode == "fan_out"
        bound = gain * sqrt(3 / fan_out)
    else
        error("""GradValley: kaiming_uniform_init: invalid fan_mode, fan_mode must be either "fan_in" or "fan_out" """)
    end
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    weight = rand(weight_shape...) * 2 * bound .- bound

    return weight
end

# xavier (glorot) initialization (uniform distribution)
function xavier_uniform_init(weight_shape::NTuple{N, Int}, gain::Real) where N
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
function bias_init(bias_shape::NTuple{N, Int}, weight_shape::NTuple{N2, Int}, gain::Real; fan_mode::AbstractString="fan_in") where {N, N2}
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        std = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        std = gain / sqrt(fan_out)
    else
        error("""GradValley: bias_init: invalid fan_mode, fan_mode must be either "fan_in" or "fan_out" """)
    end
    bias = randn(bias_shape) * std

    return bias
end

# bias initialization (uniform distribution)
function bias_uniform_init(bias_shape::Tuple, weight_shape::Tuple, gain::Real; fan_mode::AbstractString="fan_in") where N
    fan_in, fan_out = calculate_fan_mode(weight_shape)
    if fan_mode == "fan_in"
        bound = gain / sqrt(fan_in)
    elseif fan_mode == "fan_out"
        bound = gain / sqrt(fan_out)
    else
        error("""GradValley: bias_uniform_init: invalid fan_mode, fan_mode must be either "fan_in" or "fan_out" """)
    end
    # uniform distribution in general: rand() * (b - a) + a
    # (https://stackoverflow.com/questions/39083344/how-to-create-a-uniformly-random-matrix-in-julia)
    # uniform(-bound, bound)
    # weight = rand(weight_shape...) * 2 * bound - bound
    bias = rand(bias_shape...) * 2 * bound .- bound

    return bias
end