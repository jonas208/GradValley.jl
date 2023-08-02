#=
Internal functions, Internals
=#

function general_activation_function_init(activation_function::Union{Nothing, AbstractString})
    if isnothing(activation_function)
        new_activation_function = nothing
        df = nothing
        gain = 1
    elseif activation_function == "relu"
        new_activation_function = Functional.relu
        df = Functional.d_relu
        gain = sqrt(2)
    elseif activation_function == "sigmoid"
        new_activation_function = Functional.sigmoid
        df = Functional.d_sigmoid
        gain = 1
    elseif activation_function == "tanh"
        new_activation_function = Functional.gv_tanh
        df = Functional.d_tanh
        gain = 5 / 3
    elseif activation_function == "leaky_relu" # leaky_relu with default negative_slope
        new_activation_function = Functional.leaky_relu
        df = Functional.d_leaky_relu
        gain = sqrt(2/(1 + 0.01^2))
    elseif split(activation_function, ":")[1] == "leaky_relu" && length(split(activation_function, ":")) > 1 # leaky_relu with specified negative_slope
        negative_slope = parse(Float64, split(activation_function, ":")[2])
        _leaky_relu(x::AbstractArray{T, N}) where {T, N} = Functional.leaky_relu(x, negative_slope=T(negative_slope))
        _d_leaky_relu(x::AbstractArray{T, N}) where {T, N} = Functional.d_leaky_relu(x, negative_slope=T(negative_slope))
        new_activation_function = _leaky_relu
        df = _d_leaky_relu
        gain = sqrt(2/(1 + negative_slope^2))
    else
        error("""GradValley: general_activation_function_init: activation_function must be one of the following:\n
            "relu",
            "sigmoid",
            "tanh",
            "leaky_relu"/"leaky_relu:negative_slope",
            use the stand alone Softmax layer for softmax activation
        """)
    end

    return new_activation_function, df, gain
end

function general_weight_and_bias_init(weight_shape::NTuple{N1, Int}, bias_shape::NTuple{N2, Int}, init_mode::AbstractString, gain::Real) where {N1, N2}
    if init_mode == "default"
        weight = Functional.default_init(weight_shape, gain)
        bias = Functional.bias_init(bias_shape, weight_shape, gain)
    elseif init_mode == "default_uniform"
        weight = Functional.default_uniform_init(weight_shape, gain)
        bias = Functional.bias_uniform_init(bias_shape, weight_shape, gain)
    elseif init_mode == "kaiming"
        weight = Functional.kaiming_init(weight_shape, gain)
        bias = Functional.bias_init(bias_shape, weight_shape, gain)
    elseif init_mode == "xavier"
        weight = Functional.xavier_init(weight_shape, gain)
        bias = Functional.bias_init(bias_shape, weight_shape, gain)
    elseif init_mode == "kaiming_uniform"
        weight = Functional.kaiming_uniform_init(weight_shape, gain)
        bias = Functional.bias_uniform_init(bias_shape, weight_shape, gain)
    elseif init_mode == "xavier_uniform"
        weight = Functional.xavier_uniform_init(weight_shape, gain)
        bias = Functional.bias_uniform_init(bias_shape, weight_shape, gain)
    else
        error("""GradValley: general_weight_and_bias_init: init_mode must be one of the following:\n
            "default",
            "default_uniform",
            "kaiming",
            "xavier",
            "kaiming_uniform",
            "xavier_uniform"
        """)
    end

    return weight, bias
end
