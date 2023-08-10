using Images, ImageTransformations

# the image is expected to have (W, H, ...) shape, where ... means an arbitrary number of suffixed dimensions
function crop(image::AbstractArray{T, N}, top::Integer, left::Integer, height::Integer, width::Integer) where {T, N}
    suffixed_dims_indices = [Colon() for _ in 1:ndims(image)-2]
    return image[left:left+width-1, top:top+height-1, suffixed_dims_indices...]
end

# the image is expected to have (W, H, ...) shape, where ... means an arbitrary number of suffixed dimensions
function center_crop(image::AbstractArray{T, N}, crop_height::Integer, crop_width::Integer) where {T, N}
    image_width, image_height = size(image)[1:2]
    crop_top = convert(Int, (round((image_height - crop_height) / 2.0))) + 1
    crop_left = convert(Int, (round((image_width - crop_width) / 2.0))) + 1
    return crop(image, crop_top, crop_left, crop_height, crop_width)
end

# the image is expected to have (W, H, C, ...) shape, where ... means an arbitrary number of suffixed dimensions
function normalize(image::AbstractArray{T, N}, mean::Vector{<: Real}, std::Vector{<: Real}) where {T <: Real, N}
    suffixed_dims_indices = [Colon() for _ in 1:ndims(image)-3]
    for channel in 1:size(image)[3]
        image[:, :, channel, suffixed_dims_indices...] = (image[:, :, channel, suffixed_dims_indices...] .- mean[channel]) ./ std[channel]
    end
    return image
end

# convert_image_eltype equivalent to torchvision's convert_image_dtype
function convert_image_eltype(image::AbstractArray{T, N}, new_eltype::DataType) where {T <: Real, N}
    if eltype(image) == new_eltype
        return image
    end
    if !(new_eltype <: Union{AbstractFloat, Integer})
        error("convert_image_eltype: new_eltype must be a subtype of AbstractFloat or Integer")
    end
    if eltype(image) <: AbstractFloat
        if new_eltype <: AbstractFloat
            return convert(Array{new_eltype}, image)
        end
        if (eltype(image) == Float32 && new_eltype in (Int32, Int64)) || (eltype(image) == Float64 && new_eltype == Int64)
            error("convert_image_eltype: the cast from $(eltype(image)) to $new_eltype cannot be performed safely")
        end
        eps = 1e-3
        max_value = 1.00 # always for floats
        output = image * (max_value + 1 - eps)
        return convert(Array{new_eltype}, output)
    elseif eltype(image) <: Integer
        input_max = typemax(eltype(image))
        if new_eltype <: AbstractFloat
            output = convert(Array{new_eltype}, image)
            output = output / input_max
            return output
        end
        output_max = convert(Float64, typemax(eltype(image)))
        if input_max > output_max
            factor = Int((input_max + 1) ÷ (output_max + 1))
            output = floor.(image / factor)
            return convert(Array{new_eltype}, output)
        else
            factor = Int((output_max + 1) ÷ (input_max + 1))
            output = convert(Array{new_eltype}, output)
            return output * factor
        end
    else
        error("convert_image_eltype: eltype(image) must be a subtype of AbstractFloat or Integer")
    end
end

function _preprocess(image::AbstractArray{T, 3}, resize_size::Integer; dtype::DataType=Float32) where T <: Real
    # convert the image to the element type dtype and scale the values accordingly
    image = convert_image_eltype(image, dtype)
    # resize equivalent to torchvision's resize with one integer given as size argument
    width, height, channels = size(image)
    # print an error if the number of channels is not equal to 3 (rgb-images), important for normalization
    if channels != 3
        error("_preprocess: error while preprocessing, the image is expected to have 3 channels, however, $channels channel(s) was/were found")
    end
    # keeping the aspect ratio
    if height >= width
        new_size = (resize_size, convert(Int, trunc(resize_size * (height/width))), channels)
    elseif width > height
        new_size = (convert(Int, trunc(resize_size * (width/height))), resize_size, channels)
    end
    image = imresize(image, new_size)
    # desired size after cropping 
    crop_size = (224, 224)
    # center crop equivalent to torchvision's center crop 
    image = center_crop(image, crop_size[1], crop_size[2])
    # mean and standard deviation for normalization (separately for each channel)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # normalize equivalent to torchvision's normalize 
    image = normalize(image, mean, std)
    return image
end

"""
    preprocess_for_resnet18_and_34(image::AbstractArray{T, 3}) where T <: Real

Preprocessing of an image for input into the pre-trained ResNet18 and ResNet34.
`image` of shape (W, H, C) is expected to have 3 channels (required for channel-by-channel normalization). 
"""
function preprocess_for_resnet18_and_34(image::AbstractArray{T, 3}; dtype::DataType=Float32) where T <: Real
    # desired size after resizing 
    resize_size = 256
    return _preprocess(image, resize_size, dtype=dtype)
end

"""
    preprocess_for_resnet50_and_101_and_152(image::AbstractArray{T, 3}) where T <: Real

Preprocessing of an image for input into the pre-trained ResNet50, ResNet101 and ResNet152.
`image` of shape (W, H, C) is expected to have 3 channels (required for channel-by-channel normalization). 
"""
function preprocess_for_resnet50_and_101_and_152(image::AbstractArray{T, 3}; dtype::DataType=Float32) where T <: Real
    # desired size after resizing 
    resize_size = 232
    return _preprocess(image, resize_size, dtype=dtype)
end

"""
    read_image_from_file(path::AbstractString)

Read an image file into an UInt8 3d-array of shape (W, H, C).
"""
function read_image_from_file(path::AbstractString)
    image = load(path)
    image = channelview(image)
    image = rawview(image)
    image = PermutedDimsArray(image, (3, 2, 1))
    # image = convert(Array{UInt8, 3}, image)
    return image
end

"""
    add_batch_dim(image::AbstractArray{T, N}) where {T, N}

Add a batch dimension as the new last dimension: effectivly from shape (...) to (..., 1)
"""
function add_batch_dim(image::AbstractArray{T, N}) where {T, N}
    #=
    output = similar(image, size(image)..., 1)
    prefixed_dims_indices = [Colon() for _ in 1:ndims(image)]
    output[prefixed_dims_indices..., 1] = image
    return output
    =#
    return reshape(image, size(image)..., 1)
end