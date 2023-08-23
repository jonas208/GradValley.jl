# extracts all layers in a stack of (nested) containers and layers recursively, returns a vector only containing all the pure layers
function extract_layers(container::AbstractContainer, layer_stack)
    for layer in container.layer_stack
        if typeof(layer) <: AbstractContainer
            extract_layers(layer, layer_stack)
        else
            push!(layer_stack, layer)
        end
    end

    return layer_stack
end

# extracts all containers in a stack of (nested) containers and layers recursively, returns a vector containing all containers
function extract_containers(container::AbstractContainer, container_stack)
    # make sure the first given container is also part of container_stack
    if length(container_stack) == 0
        push!(container_stack, container)
    end

    for layer in container.layer_stack
        if typeof(layer) <: AbstractContainer
            push!(container_stack, layer)
            extract_containers(layer, container_stack)
        end
    end

    return container_stack
end

"""
    clean_model_from_backward_information!(model::AbstractContainer)

Clean a container from backward pass information (e.g. computational graph).
It is recommended to run this function on a model which should be saved to a file.
"""
function clean_model_from_backward_information! end

function clean_model_from_backward_information!(model::AbstractContainer)
    for container in extract_containers(model, [])
        if typeof(container) == SequentialContainer
            container.tracked_input = TrackedReal(0, nothing, nothing, nothing)
            container.tracked_output = TrackedReal(0, nothing, nothing, nothing)
        else # typeof(container) == GraphContainer
            container.tracked_inputs = TrackedReal[]
            container.tracked_output = TrackedReal(0, nothing, nothing, nothing)
        end
    end
end

"""
    module_to_eltype_device!(layer::AbstractLayer; element_type::DataType=Float32, device::AbstractString="cpu")

Convert the parameters of a container or layer to a different element type and move the parameters to the specified device.

# Arguments
- `layer::AbstractLayer`: the layer or container (often just the entire model) holding the parameters to be changed
- `element_type::DataType=Float32`: the element type into which the parameters will be converted to
- `device::AbstractString="cpu"`: the device to which the parameters will be moved, can be "cpu" or "gpu" (only CUDA is supported)
"""
function module_to_eltype_device! end

function module_to_eltype_device!(layer::AbstractLayer; element_type::DataType=Float32, device::AbstractString="cpu")
    if device == "gpu"
        for property_name in propertynames(layer)
            if property_name in [:weight, :bias, :weight_gradient, :bias_gradient, :running_mean, :running_variance]
                weight = getfield(layer, property_name)
                if typeof(weight) <: CuArray
                    converted_weight = convert(CuArray{element_type}, weight)
                else
                    converted_weight = convert(Array{element_type}, weight)
                    converted_weight = CuArray(converted_weight)
                end
                setfield!(layer, property_name, converted_weight)
            end
        end
    elseif device == "cpu"
        for property_name in propertynames(layer)
            if property_name in [:weight, :bias, :weight_gradient, :bias_gradient, :running_mean, :running_variance]
                weight = getfield(layer, property_name)
                converted_weight = convert(Array{element_type}, weight)
                setfield!(layer, property_name, converted_weight)
            end
        end
    else
        error("""GradValley: module_to_eltype_device: device must be "cpu" or "gpu" """)
    end
end

function module_to_eltype_device!(model::AbstractContainer; element_type::DataType=Float32, device::AbstractString="cpu")
    clean_model_from_backward_information!(model)
    module_to_eltype_device!.(extract_layers(model, []), element_type=element_type, device=device)
end