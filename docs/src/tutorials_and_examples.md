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
using MLDatasets # # a package for downloading datasets
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
We will be using the [BSON.jl](https://github.com/JuliaIO/BSON.jl) package for saving the model easily.
```julia
# when this file is run as the main script,
# then train() is run and the final model will be saved using a package called BSON.jl
if abspath(PROGRAM_FILE) == @__FILE__
    train()
    file_name = "MNIST_with_LeNet5_model.bson"
    @save file_name model
    println("Saved trained model as $file_name")
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