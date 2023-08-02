# returns a string containing useful information about the given layer
function get_layer_summary(layer)
    summary = "$(typeof(layer)): "
    num_params = 0
    if typeof(layer) == Conv
        summary *= "in_channels=$(layer.in_channels), out_channels=$(layer.out_channels), kernel_size=$(layer.kernel_size), stride=$(layer.stride), padding=$(layer.padding), dilation=$(layer.dilation), groups=$(layer.groups), activation_function=$(layer.activation_function), use_bias=$(layer.use_bias)"
        if layer.use_bias
            num_params += length(layer.weight) + length(layer.bias)
        else
            num_params += length(layer.weight)
        end
    elseif typeof(layer) == ConvTranspose
        summary *= "in_channels=$(layer.in_channels), out_channels=$(layer.out_channels), kernel_size=$(layer.kernel_size), stride=$(layer.stride), padding=$(layer.padding), output_padding=$(layer.output_padding), dilation=$(layer.dilation), groups=$(layer.groups), activation_function=$(layer.activation_function), use_bias=$(layer.use_bias)"
        if layer.use_bias
            num_params += length(layer.weight) + length(layer.bias)
        else
            num_params += length(layer.weight)
        end
    elseif typeof(layer) == MaxPool || typeof(layer) == AvgPool
        summary *= "kernel_size=$(layer.kernel_size), stride=$(layer.stride), padding=$(layer.padding), dilation=$(layer.dilation), activation_function=$(layer.activation_function)"
    elseif typeof(layer) == AdaptiveMaxPool || typeof(layer) == AdaptiveAvgPool
        summary *= "output_size=$(layer.output_size), activation_function=$(layer.activation_function)"
    elseif typeof(layer) == Identity
        summary *= "activation_function=$(layer.activation_function)"
    elseif typeof(layer) == Fc
        summary *= "in_features=$(layer.in_features), out_features=$(layer.out_features), activation_function=$(layer.activation_function), use_bias=$(layer.use_bias)"
        if layer.use_bias
            num_params += length(layer.weight) + length(layer.bias)
        else
            num_params += length(layer.weight)
        end
    elseif typeof(layer) == Reshape
        summary *= "out_shape=$(layer.out_shape), activation_function=$(layer.activation_function)"
    elseif typeof(layer) == Softmax
        summary *= "dims=$(layer.dims)"
    elseif typeof(layer) == BatchNorm
        summary *= "num_features=$(layer.num_features), epsilon=$(layer.epsilon), momentum=$(layer.momentum), affine=$(layer.affine), track_running_stats=$(layer.track_running_stats), test_mode=$(layer.test_mode), activation_function=$(layer.activation_function)"
        num_params += length(layer.weight) + length(layer.bias)
    end
    
    return summary, num_params
end

# returns a string (and the total number of parameters) with an overview of the container (currently doesn't show an visualization of the computational graph)
function summarize_container(container::AbstractContainer, sub_counter::String)
    if typeof(container) == SequentialContainer
        summary = "SequentialContainer\n(\n"
    else # GraphContainer
        summary = "GraphContainer\n(\n"
    end
    num_params = 0
    for (i, layer) in enumerate(container.layer_stack)
        if sub_counter == ""
            sub_counter_new = string(i)
        else
            sub_counter_new = sub_counter * "." * string(i)
        end
        if typeof(layer) <: AbstractContainer
            layer_summary, layer_num_params = summarize_container(layer, sub_counter_new) # works recursively
            lines = split(layer_summary, "\n")
            summary *= "($sub_counter_new) " * lines[1] * "\n"
            for line in lines[2:end]
                summary *= "     " * line * "\n"
            end
        else
            layer_summary, layer_num_params = get_layer_summary(layer)
            summary *= "($sub_counter_new) $layer_summary\n"
        end
        num_params += layer_num_params
    end
    summary *= ")" # ")\n"

    return summary, num_params
end

# returns a string (and the total number of parameters) intended for printing with an overview of the model (currently doesn't show an visualization of the computational graph) and its number of parameters
"""
    summarize_model(container::AbstractContainer)

Return a string (and the total number of parameters) intended for printing with an overview of the model 
(currently doesn't show an visualization of the computational graph) and its number of parameters.
"""
function summarize_model(container::AbstractContainer)
    summary, num_params = summarize_container(container, "")
    summary *= "\n"
    summary *= "Total Layers: $(length(extract_layers(container, [])))\n"
    summary *= "Total Parameters: $num_params"

    return summary, num_params
end

# pretty-printing for all layers and containers 
Base.show(io::IO, layer::AbstractLayer) = print(io, get_layer_summary(layer)[1])
Base.show(io::IO, container::AbstractContainer) = print(io, summarize_model(container)[1])