using GradValley
include("preprocessing_for_resnets.jl")
using FileIO

# make sure there is an / at the end of the data_directory string
data_directory = "F:/archive (1)/celeba_hq_256/" # replace the string with the real path to the folder containing the images
files = readdir(data_directory)
dataset_size = length(files) # aka number of files/images

dtype = Float64 # Float64 is heavily recommend here, we can switch to Float32 for training any way
image_size = 64
batch_size = 128

# get function for the data loader that reads and transforms an image
function get_image(index::Integer)
    image = read_image_from_file(data_directory * files[index])
    image_size = 64
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
        new_size = (image_size, convert(Int, trunc(image_size * (height/width))), channels)
    elseif width > height
        new_size = (convert(Int, trunc(image_size * (width/height))), image_size, channels)
    end
    image = imresize(image, new_size)
    #=
    # desired size after cropping 
    crop_size = (image_size, image_size)
    # center crop equivalent to torchvision's center crop 
    image = center_crop(image, crop_size[1], crop_size[2])
    =#
    # mean and standard deviation for normalization (separately for each channel)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # normalize equivalent to torchvision's normalize 
    image = normalize(image, mean, std)

    return (image, )
end

# initialize the data loader for loading the images into batches
dataloader = DataLoader(get_image, dataset_size, batch_size=batch_size, shuffle=true)
num_batches = dataloader.num_batches
file_name = "CelebA-HQ_preprocessed.jld2" # you can change the file name/path here as well
println("Number of batches: $num_batches")

# data is a vector conatining the image batches
data = Vector{Array{dtype, 4}}(undef, num_batches)
# iterate over the data loader and add the batches to the data vector
for (batch_index, (images_batch, )) in enumerate(dataloader)
    println("[$batch_index/$num_batches]")
    data[batch_index] = images_batch
end
# the vector containing the batches is stored in file_name under the "data" key
save(file_name, Dict("data" => data))