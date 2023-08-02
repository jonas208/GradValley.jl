using cuDNN: cudnnSoftmaxForward

# this version is the version of leaky_relu that works with CUDA
function leaky_relu(x::CuArray{T, N}; negative_slope::T=T(0.01)) where {T <: Real, N}
    f(x) = if x < 0 return x * negative_slope else return x end
    return f.(x)
end

# this version is the version of d_leaky_relu that works with CUDA
function d_leaky_relu(x::CuArray{T, N}; negative_slope::T=T(0.01)) where {T <: Real, N}
    f(x) = if x < 0 return negative_slope else return one(T) end
    return f.(x)
end

sigmoid(x::CuArray{T, N}) where {T <: Real, N} = sigmoid.(x)

d_sigmoid(x::CuArray{T, N}) where {T <: Real, N} = d_sigmoid.(x)

gv_tanh(x::CuArray{T, N}) where {T <: Real, N} = tanh.(x)

d_tanh(x::CuArray{T, N}) where {T <: Real, N} = d_tanh.(x)