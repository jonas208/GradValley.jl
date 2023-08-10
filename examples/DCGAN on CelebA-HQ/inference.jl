using GradValley
using GradValley.Layers
using Images

num_images = 50
name_prefix = "fake"
format = ".jpeg"
# make sure there is an / at the end of the dist string
dist = "inference/"
!isdir(dist) && mkdir(dist)

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Float32 or Float64
dtype = Float32

# converts a tensor of size (width, height, channels) to a 2d RGB image array
function tensor_to_image(tensor::AbstractArray{T, 3}) where T <: AbstractFloat
    image = PermutedDimsArray(tensor, (3, 2, 1))
    image = colorview(RGB, image)
    return image
end

file_nameG = "generator.jld2"
generator = load(file_nameG, "generator")
# testmode!(generator)
module_to_eltype_device!(generator, element_type=dtype, device="cpu")

noise = randn(dtype, 1, 1, nz, num_images)
fake = generator(noise)
fake = @time generator(noise)

for i in 1:num_images
    image = @view fake[:, :, :, i]

    # normalize
    min = minimum(image)
    max = maximum(image)
    norm(x) = (x - min) / (max - min)
    image = norm.(image)

    image = tensor_to_image(image)

    file_path = dist * name_prefix * string(i) * format
    save(file_path, image)
end