import Base: tanh

# relu(x::T) where {T <: Real} = if x < 0 return zero(T) else return x end
relu(x::T) where {T <: Real} = max(0, x)
relu(x::AbstractArray{T, N}) where {T <: Real, N} = relu.(x)

d_relu(x::T) where {T <: Real} = if x < 0 return zero(T) else return one(T) end
d_relu(x::AbstractArray{T, N}) where {T <: Real, N} = d_relu.(x)

leaky_relu(x::T; negative_slope::T=T(0.01)) where {T <: Real} = if x < 0 return x * negative_slope else return x end
# this version doesn't work with CUDA
leaky_relu(x::AbstractArray{T, N}; negative_slope::T=T(0.01)) where {T <: Real, N} = leaky_relu.(x, negative_slope=negative_slope)

d_leaky_relu(x::T; negative_slope::T=T(0.01)) where {T <: Real} = if x < 0 return negative_slope else return one(T) end
# this version doesn't work with CUDA
d_leaky_relu(x::AbstractArray{T, N}; negative_slope::T=T(0.01)) where {T <: Real, N} = d_leaky_relu.(x, negative_slope=negative_slope)

sigmoid(x::T) where {T <: Real} = 1 / (1 + exp(-x))
sigmoid(x::AbstractArray{T, N}) where {T <: Real, N} = @turbo sigmoid.(x)

# d_sigmoid(x::T) where {T <: Real} = sigmoid(x) * sigmoid(1 - sigmoid(x))
d_sigmoid(x::T) where {T <: Real} = sigmoid(x) * (1 - sigmoid(x))
d_sigmoid(x::AbstractArray{T, N}) where {T <: Real, N} = @turbo d_sigmoid.(x)

gv_tanh(x::AbstractArray{T, N}) where {T <: Real, N} = @turbo tanh.(x)

d_tanh(x::T) where {T <: Real} = 1 - tanh(x)^2
d_tanh(x::AbstractArray{T, N}) where {T <: Real, N} = @turbo d_tanh.(x)

function softmax_forward(input::AbstractArray{T, N}; dims=1) where {T <: Real, N}
    output = similar(input)
    output .= exp.(input .- maximum(input, dims=dims))
    output ./= sum(output; dims=dims)

    return output
end

function softmax_backward(output_gradient::AbstractArray{T, N}, output::AbstractArray{T, N}; dims=1) where {T <: Real, N}
    tmp = output_gradient .* output
    input_gradient = tmp .- output .* sum(tmp, dims=dims)

    return input_gradient
end