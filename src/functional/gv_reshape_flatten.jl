#=
Reshape: Forward & Backward
=#

function reshape_forward(inputs::Array, out_shape::Tuple)
    current_batch_size = size(inputs)[1]
    output_shape = tuplejoin((current_batch_size, ), out_shape)
    # outputs = Array{Float64}(undef, output_shape...)
    outputs = reshape(inputs, output_shape)

    return outputs
end

# Functions used for Backpropagation (Reshape)
# The only input each function takes is an instance of a reshape layer struct (Reshape)
# Because a layer is given, these functions directly work on the hole batch

function reshape_backward(reshape_layer)
    if reshape_layer.df != 1
        out_losses = reshape_layer.losses .* reshape_layer.df(reshape_layer.outputs_no_activation)
    else
        out_losses = reshape_layer.losses
    end
    in_shape = size(reshape_layer.inputs)
    losses = reshape(out_losses, in_shape)

    return losses
end
