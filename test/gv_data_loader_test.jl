# Note that most of this code is from the MNIST example: https://github.com/jonas208/GradValley.jl/tree/main/examples/MNIST_with_LeNet5.jl
# However, to save another dependencie, MNIST is replaced here with random data

# using MLDatasets # a package for downloading datasets

# initialize train- and test-dataset
# mnist_train = MNIST(:train)
# mnist_test = MNIST(:test)
dtype = Float32

# function for getting an image and the corresponding target vector from the train or test partition
function get_element(index, partition)
    # load one image and the corresponding label
    if partition == "train"
        image, label = rand(dtype, 28, 28), rand(0:9) # mnist_train[index]
    else # test partition
        image, label = rand(dtype, 28, 28), rand(0:9) # mnist_test[index]
    end
    # add channel dimension and rescaling the values to their original 8 bit gray scale values
    image = reshape(image, 28, 28, 1) .* 255
    # generate the target vector from the label, one for the correct digit, zeros for the wrong digits
    targets = zeros(10)
    targets[label + 1] = 1.00

    return image, targets
end

# initialize the data loaders (with anonymous function which helps to easily distinguish between test- and train-partition)
train_data_loader = DataLoader(index -> get_element(index, "train"), 60_000, batch_size=32, shuffle=true, drop_last=true) # 60_000 = length(mnist_train)
test_data_loader = DataLoader(index -> get_element(index, "test"), 10_000, batch_size=32) # 10_000 = length(mnist_test)

# in most cases NOT recommended: you can force the data loaders to load all the data at once into memory, depending on the dataset's size, this may take a while
train_data = train_data_loader[begin:end]
test_data = test_data_loader[begin:end]

# now you can write your train- or test-loop like so 
for (batch, (images_batch, targets_batch)) in enumerate(test_data_loader) #=do anything useful here=# end
for (batch, (images_batch, targets_batch)) in enumerate(train_data_loader) #=do anything useful here=# end

# test indexing and size functions
train_element, train_batch = get_element(1, "train")[1], train_data_loader[1][1]
test_element, test_batch = get_element(1, "test")[1], test_data_loader[1][1]
size(train_data_loader); size(test_data_loader)
length(train_data_loader); length(test_data_loader)
# test types  
@test eltype(train_element) == eltype(train_batch)
@test eltype(test_element) == eltype(test_batch)
# test batch size 
@test size(train_batch)[end] == 32
@test size(test_batch)[end] == 32

@info size(train_batch)
@info size(test_batch)

# test reshuffle
reshuffle!(train_data_loader)
reshuffle!(test_data_loader)

# test printing
println(train_data_loader)
println(test_data_loader)