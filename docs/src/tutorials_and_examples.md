# Tutorials and Examples

Here, you can find detailed explanations on how to build and train specific models with GradValley.jl.

## A LeNet-like model for handwritten digit recognition

In this tutorial, we will learn the basics of GradValley.jl while building a model for handwritten digit recognition, reaching approximately 99% accuracy on the MNIST-dataset.
The whole code at once can be found [here](https://github.com/jonas208/GradValley.jl/blob/main/tutorials/MNIST_with_LeNet5.jl).

### Importing modules

```julia
using GradValley # the master module of GradValley.jl
using GradValley.Layers # The "Layers" module provides all the building blocks for creating a model. 
using GradValley.Optimization # The "Optimization" module provides different loss functions and optimizers.
```

### Using the dataset

We will use the MLDatasets package which downloads the MNIST-dataset for us automatically.
If you haven't installed MLDatasets yet, write this for [installation](https://juliaml.github.io/MLDatasets.jl/stable/#Installation):
```julia
import Pkg; Pkg.add("MLDatasets")
```
Then we can import MLDatasets:
```julia
using MLDatasets # a package for downloading datasets
```

#### Splitting up the dataset into a train and a test partition

The MNIST-dataset contains 70,000 images, we will use 60,000 images for training the network and 10,000 images for evaluating accuracy.

```julia
# initialize train- and test-dataset
mnist_train = MNIST(:train) 
mnist_test = MNIST(:test)
```

#### Using GradValley.DataLoader for handling data

A typical workflow when dealing with datasets is to use the GradValley.DataLoader struct. A data loader makes it easy to iterate directly over the batches in a dataset. 
Due to better memory efficiency, the data loader loads the batches *just in time*. When initializing a data loader, we specify a function that returns exactly one element from the dataset at a given index.
We also have to specify the size of the dataset (e.g. the number of images). All parameters that the data loader accepts (see [Reference](@ref) for more information):
```julia
DataLoader(get_function::Function, dataset_size::Integer; batch_size::Integer=1, shuffle::Bool=false, drop_last::Bool=false)
```
Now we write the *get function* for the two data loaders.
```julia
# function for getting an image and the corresponding target vector from the train or test partition
function get_element(index, partition)
    # load one image and the corresponding label
    if partition == "train"
        image, label = mnist_train[index]
    else # test partition
        image, label = mnist_test[index]
    end
    # add channel dimension and rescaling the values to their original 8 bit gray scale values
    image = reshape(image, 1, 28, 28) .* 255
    # generate the target vector from the label, one for the correct digit, zeros for the wrong digits
    targets = zeros(10)
    targets[label + 1] = 1.00

    return convert(Array{Float64, 3}, image), targets
end
```
We can now initialize the data loaders.
```julia
# initialize the data loaders
train_data_loader = DataLoader(index -> get_element(index, "train"), length(mnist_train), batch_size=32, shuffle=true)
test_data_loader = DataLoader(index -> get_element(index, "test"), length(mnist_test), batch_size=32)
```
If you want to force the data loader to load the data all at once, you could do:
```julia
# force the data loaders to load all the data at once into memory, depending on the dataset's size, this may take a while
train_data = train_data_loader[begin:end]
test_data = test_data_loader[begin:end]
```

### Building the neuronal network aka. the model

The most recommend way to build models is to use the GradValley.Layers.SequentialContainer struct. A SequtialContainer can take an array of layers or other SequentialContainers (sub-models).
While forward-pass, the given inputs are *sequentially* propagated through every layer (or sub-model) and the output will be returned. For more details, see [Reference](@ref).
The LeNet5 model is one of the earliest convolutional neuronal networks (CNNs) reaching approximately 99% accuracy on the MNIST-dataset.
The LeNet5 is built of two main parts, the feature extractor and the classifier. So it would be a good idea to clarify that in the code:
```julia
# Definition of a LeNet-like model consisting of a feature extractor and a classifier
feature_extractor = SequentialContainer([ # a convolution layer with 1 in channel, 6 out channels, a 5*5 kernel and a relu activation
                                         Conv(1, 6, (5, 5), activation_function="relu"),
                                         # an average pooling layer with a 2*2 filter (when not specified, stride is automatically set to kernel size)
                                         AvgPool((2, 2)),
                                         Conv(6, 16, (5, 5), activation_function="relu"),
                                         AvgPool((2, 2))])
flatten = Reshape((256, ))
classifier = SequentialContainer([ # a fully connected layer (also known as dense or linear) with 256 in features, 120 out features and a relu activation
                                  Fc(256, 120, activation_function="relu"),
                                  Fc(120, 84, activation_function="relu"),
                                  Fc(84, 10),
                                  # a softmax activation layer, the softmax will be calculated along the second dimension (the features dimension)
                                  Softmax(dim=2)])
# The final model consists of three different submodules, 
# which shows that a SequentialContainer can contain not only layers, but also other SequentialContainers
model = SequentialContainer([feature_extractor, flatten, classifier])
```

#### Printing a nice looking summary of the model

Summarizing a model and counting the number of trainable parameters is easily done with the GradValley.Layers.summarie_model function.
```julia
# printing a nice looking summary of the model
summary, num_params = summarize_model(model)
println(summary)
```

#### Defining hyperparameters

Before we start to train and test the model, we define all necessary hyperparameters.
If we want to change the learning rate or the loss function for example, this is the one place to do this.
```julia
# defining hyperparameters
loss_function = mse_loss # mean squared error
learning_rate = 0.05
optimizer = MSGD(model, learning_rate, momentum=0.5) # momentum stochastic gradient descent with a momentum of 0.5
epochs = 5 # 5 or 10
```

### Train and test the model

The next step is to write a function for training the model using the above defined hyperparameters.
The network is trained 10 times (epochs) with the entire training data set. After each batch, the weights/parameters of the network are adjusted/optimized.
However, we want to test the model after each epoch, so we need to write a function for evaluating the model's accuracy first.

```julia
# evaluate the model's accuracy
function test()
    num_correct_preds = 0
    avg_test_loss = 0
    for (batch, (images_batch, targets_batch)) in enumerate(test_data_loader)
        # computing predictions
        predictions_batch = forward(model, images_batch)
        # checking for each image in the batch individually if the prediction is correct
        for index_batch in 1:size(predictions_batch)[1]
            single_prediction = predictions_batch[index_batch, :]
            single_target = targets_batch[index_batch, :]
            if argmax(single_prediction) == argmax(single_target)
                num_correct_preds += 1
            end
        end
        # adding the loss for measuring the average test loss
        avg_test_loss += loss_function(predictions_batch, targets_batch, return_derivative=false)
    end

    accuracy = num_correct_preds / size(test_data_loader) * 100 # size(data_loader) returns the dataset size
    avg_test_loss /= length(test_data_loader) # length(data_loader) returns the number of batches

    return accuracy, avg_test_loss
end

# train the model with the above defined hyperparameters
function train()
    for epoch in 1:epochs

        @time begin # for measuring time taken by one epoch

            avg_train_loss = 0.00
            # iterating over the whole data set
            for (batch, (images_batch, targets_batch)) in enumerate(train_data_loader)
                # computing predictions
                predictions_batch = forward(model, images_batch)
                # backpropagation
                zero_gradients(model)
                loss, derivative_loss = loss_function(predictions_batch, targets_batch)
                backward(model, derivative_loss)
                # optimize the model's parameters
                step!(optimizer)
                # printing status
                if batch % 100 == 0
                    image_index = batch * train_data_loader.batch_size
                    data_set_size = size(train_data_loader)
                    println("Batch $batch, Image [$image_index/$data_set_size], Loss: $(round(loss, digits=5))")
                end
                # adding the loss for measuring the average train loss
                avg_train_loss += loss
            end

            avg_train_loss /= length(train_data_loader)
            accuracy, avg_test_loss = test()
            print("Results of epoch $epoch: Avg train loss: $(round(avg_train_loss, digits=5)), Avg test loss: $(round(avg_test_loss, digits=5)), Accuracy: $accuracy%, Time taken:")

        end

    end
end
```

#### Run the training and save the trained model afterwards

When the file is run as the main script, we want to actually call the train() function and save the final model afterwards.
We will use the [BSON.jl](https://github.com/JuliaIO/BSON.jl) package for saving the model easily.
```julia
# when this file is run as the main script,
# then train() is run and the final model will be saved using a package called BSON.jl
import Pkg; Pkg.add("BSON")
using BSON: @save
if abspath(PROGRAM_FILE) == @__FILE__
    train()
    file_name = "MNIST_with_LeNet5_model.bson"
    @save file_name model
    println("Saved trained model as $file_name")
end
```

#### Use the trained model

If you want to easily use the trained model, you firstly need to import the necessary modules from GradValley.
Then you can use the @load macro of BSON to load the model object. Now you can let the model make a few individual predictions, for example.
Use this code in an extra file.
```julia
using GradValley
using GradValley.Layers 
using GradValley.Optimization
using MLDatasets
using BSON: @load

# load the trained model
@load "MNIST_with_LeNet5_model.bson" model

# make some individual predictions
mnist_test = MNIST(:test)
for i in 1:5
    random_index = rand(1:length(mnist_test))
    image, label = mnist_test[random_index]
    # remember to add batch and channel dimensions and to rescale the image as was done during training and testing
    image_batch = convert(Array{Float64, 4}, reshape(image, 1, 1, 28, 28)) .* 255
    prediction = forward(model, image_batch)
    predicted_label = argmax(prediction[1, :]) - 1
    println("Predicted label: $predicted_label, Correct Label: $label")
end
```

### Running the file with multiple threads

It is heavily recommended to run this file, and any other files using GradValley, with multiple threads.
Using multiple threads can make training much faster.
To do this, use the ```-t``` option when running a julia script in terminal/PowerShell/command line/etc.
If your CPU has 24 threads, for example, then run:
```
julia -t 24 ./MNIST_with_LeNet5.jl
```
The specified number of threads should match the number of threads your CPU provides.

### Results

These were my results after 5 training epochs:
*Results of epoch 5: Avg train loss: 0.00237, Avg test loss: 0.00283, Accuracy: 98.21%, Time taken: 13.416619 seconds (20.34 M allocations: 30.164 GiB, 5.86% gc time)*
On my Ryzen 9 5900X CPU (using all 24 threads, slightly overclocked), one epoch took around ~15 seconds (no compilation time), so the whole training (5 epochs) took around ~75 seconds (no compilation time).

## Generic ResNet (18/34/50/101/152) implementation

The same code can be also found [here](https://github.com/jonas208/GradValley.jl/blob/main/tutorials/ResNets.jl).

This example shows the ResNet implementation used by the [pre-trained ResNets](https://jonas208.github.io/GradValley.jl/(pre-trained)_models/#ResNet18/34/50/101/152-(Image-Classification)).
The function `ResBlock` generates a standard residual block (with one residual/skipped connection) with optional downsampling. 
On the other hand, the function `ResBottelneckBlock` generates a bottleneck residual block (a variant of the residual block that utilises 1x1 convolutions to create a bottleneck) with optional downsampling.
The residual connections can be easily implemented using the [`GraphContainer`](@ref). [`GraphContainer`](@ref) allows differentiation for any computational graphs (not only sequential graphs for which the [`SequentialContainer`](@ref) is intended).
The function `ResNet` constructs a generic ResNet. The functions `ResNetXX` use this function to create the individual models.

Note that this implementation is inspired by [this article](https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448).

```julia
# import GradValley
using GradValley
using GradValley.Layers

# define a ResBlock (with optional downsampling) 
function ResBlock(in_channels::Int, out_channels::Int, downsample::Bool)
    # define modules
    if downsample
        shortcut = SequentialContainer([
            Conv(in_channels, out_channels, (1, 1), stride=(2, 2), use_bias=false),
            BatchNorm2d(out_channels)
        ])
        conv1 = Conv(in_channels, out_channels, (3, 3), stride=(2, 2), padding=(1, 1), use_bias=false)
    else
        shortcut = Identity()
        conv1 = Conv(in_channels, out_channels, (3, 3), padding=(1, 1), use_bias=false)
    end

    conv2 = Conv(out_channels, out_channels, (3, 3), padding=(1, 1), use_bias=false)
    bn1 = BatchNorm2d(out_channels, activation_function="relu")
    bn2 = BatchNorm2d(out_channels) # , activation_function="relu"

    relu = Identity(activation_function="relu")

    # define the forward pass with the residual/skipped connection
    function forward_pass(modules, input)
        # extract modules from modules vector (not necessary (therefore commented out) because the forward_pass function is defined in the ResBlock function (not somewhere "outside").)
        # shortcut, conv1, conv2, bn1, bn2, relu = modules
        # compute shortcut
        output_shortcut = forward(shortcut, input)
        # compute sequential part
        output = forward(bn1, forward(conv1, input))
        output = forward(bn2, forward(conv2, output))
        # residual/skipped connection
        output = forward(relu, output + output_shortcut) 
        
        return output
    end

    # initialize a container representing the ResBlock
    modules = [shortcut, conv1, conv2, bn1, bn2, relu]
    res_block = GraphContainer(forward_pass, modules)

    return res_block
end

# define a ResBottelneckBlock (with optional downsampling) 
function ResBottelneckBlock(in_channels::Int, out_channels::Int, downsample::Bool)
    # define modules
    shortcut = Identity()   

    if downsample || in_channels != out_channels
        if downsample
            shortcut = SequentialContainer([
                Conv(in_channels, out_channels, (1, 1), stride=(2, 2), use_bias=false),
                BatchNorm2d(out_channels)
            ])
        else
            shortcut = SequentialContainer([
                Conv(in_channels, out_channels, (1, 1), use_bias=false),
                BatchNorm2d(out_channels)
            ])
        end
    end

    conv1 = Conv(in_channels, out_channels ÷ 4, (1, 1), use_bias=false)
    if downsample
        conv2 = Conv(out_channels ÷ 4, out_channels ÷ 4, (3, 3), stride=(2, 2), padding=(1, 1), use_bias=false)
    else
        conv2 = Conv(out_channels ÷ 4, out_channels ÷ 4, (3, 3), padding=(1, 1), use_bias=false)
    end
    conv3 = Conv(out_channels ÷ 4, out_channels, (1, 1), use_bias=false)

    bn1 = BatchNorm2d(out_channels ÷ 4, activation_function="relu")
    bn2 = BatchNorm2d(out_channels ÷ 4, activation_function="relu")
    bn3 = BatchNorm2d(out_channels) # , activation_function="relu"

    relu = Identity(activation_function="relu")

    # define the forward pass with the residual/skipped connection
    function forward_pass(modules, input)
        # extract modules from modules vector (not necessary (therefore commented out) because the forward_pass function is defined in the ResBlock function (not somewhere "outside").)
        # shortcut, conv1, conv2, conv3, bn1, bn2, bn3, relu = modules
        # compute shortcut
        output_shortcut = forward(shortcut, input)
        # compute sequential part
        output = forward(bn1, forward(conv1, input))
        output = forward(bn2, forward(conv2, output))
        output = forward(bn3, forward(conv3, output))
        # residual/skipped connection
        output = forward(relu, output + output_shortcut) 
        
        return output
    end

    # initialize a container representing the ResBlock
    modules = [shortcut, conv1, conv2, conv3, bn1, bn2, bn3, relu]
    res_bottelneck_block = GraphContainer(forward_pass, modules)

    return res_bottelneck_block
end

# define a ResNet 
function ResNet(in_channels::Int, ResBlock::Union{Function, DataType}, repeat::Vector{Int}; use_bottelneck::Bool=false, classes::Int=1000)
    # define layer0
    layer0 = SequentialContainer([
        Conv(in_channels, 64, (7, 7), stride=(2, 2), padding=(3, 3), use_bias=false),
        BatchNorm2d(64, activation_function="relu"),
        MaxPool((3, 3), stride=(2, 2), padding=(1, 1))
    ])

    # define number of filters/channels
    if use_bottelneck
        filters = Int[64, 256, 512, 1024, 2048]
    else
        filters = Int[64, 64, 128, 256, 512]
    end

    # define the following modules
    layer1_modules = [ResBlock(filters[1], filters[2], false)]
    for i in 1:repeat[1] - 1
        push!(layer1_modules, ResBlock(filters[2], filters[2], false))
    end
    layer1 = SequentialContainer(layer1_modules)

    layer2_modules = [ResBlock(filters[2], filters[3], true)]
    for i in 1:repeat[2] - 1
        push!(layer2_modules, ResBlock(filters[3], filters[3], false))
    end
    layer2 = SequentialContainer(layer2_modules)

    layer3_modules = [ResBlock(filters[3], filters[4], true)]
    for i in 1:repeat[3] - 1
        push!(layer3_modules, ResBlock(filters[4], filters[4], false))
    end
    layer3 = SequentialContainer(layer3_modules)

    layer4_modules = [ResBlock(filters[4], filters[5], true)]
    for i in 1:repeat[4] - 1
        push!(layer4_modules, ResBlock(filters[5], filters[5], false))
    end
    layer4 = SequentialContainer(layer4_modules)

    gap = AdaptiveAvgPool((1, 1))
    flatten = Reshape((filters[5], ))
    fc = Fc(filters[5], classes)

    # initialize a container representing the ResNet
    res_net = SequentialContainer([layer0, layer1, layer2, layer3, layer4, gap, flatten, fc])

    return res_net
end

# construct a ResNet18
function ResNet18(in_channels=3, classes=1000)
    return ResNet(in_channels, ResBlock, [2, 2, 2, 2], use_bottelneck=false, classes=classes)
end

# construct a ResNet34
function ResNet34(in_channels=3, classes=1000)
    return ResNet(in_channels, ResBlock, [3, 4, 6, 3], use_bottelneck=false, classes=classes)
end

# construct a ResNet50
function ResNet50(in_channels=3, classes=1000)
    return ResNet(in_channels, ResBottelneckBlock, [3, 4, 6, 3], use_bottelneck=true, classes=classes)
end

# construct a ResNet101
function ResNet101(in_channels=3, classes=1000)
    return ResNet(in_channels, ResBottelneckBlock, [3, 4, 23, 3], use_bottelneck=true, classes=classes)
end

# construct a ResNet152
function ResNet152(in_channels=3, classes=1000)
    return ResNet(in_channels, ResBottelneckBlock, [3, 8, 36, 3], use_bottelneck=true, classes=classes)
end
```

It is heavily recommended to run this file (or the file in which you inlcude ResNets.jl), and any other files using GradValley, with multiple threads.
Using multiple threads can make training and calculating predictions much faster.
To do this, use the ```-t``` option when running a julia script in terminal/PowerShell/command line/etc.
If your CPU has 24 threads, for example, then run:
```
julia -t 24 ./ResNets.jl
```
The specified number of threads should match the number of threads your CPU provides.