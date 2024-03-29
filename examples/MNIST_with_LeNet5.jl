#=
This tutorial was written by Jonas S. and is part of the GradValley.jl repository.

About:
In this tutorial, we will learn the basics of GradValley.jl while building a model for handwritten digit recognition, reaching approximately 99% accuracy on the MNIST-dataset.
The more detailed version of this tutorial can be found in this section of the documentation: https://jonas208.github.io/GradValley.jl/tutorials_and_examples/#A-LeNet-like-model-for-handwritten-digit-recognition

Important Note:
It is heavily recommended to run this file, and any other files using GradValley, with multiple threads.
Using multiple threads can make training and calculating predictions much faster.
To do this, use the -t option when running a julia script in terminal/PowerShell/command line/etc.
If your CPU has 24 threads, for example, then run:
julia -t 24 ./MNIST_with_LeNet5.jl
The specified number of threads should match the number of threads your CPU provides.
=#

using GradValley # the master module of GradValley.jl
using GradValley.Layers # The "Layers" module provides all the building blocks for creating a model. 
using GradValley.Optimization # The "Optimization" module provides different loss functions and optimizers.
using MLDatasets # a package for downloading datasets
using BSON: @save # a package for saving and loading julia objects as files

# initialize train- and test-dataset
mnist_train = MNIST(:train)
mnist_test = MNIST(:test)

# function for getting an image and the corresponding target vector from the train or test partition
function get_element(index, partition)
    # load one image and the corresponding label
    if partition == "train"
        image, label = mnist_train[index]
    else # test partition
        image, label = mnist_test[index]
    end
    # add channel dimension and rescaling the values to their original 8 bit gray scale values
    image = reshape(image, 28, 28, 1) .* 255
    # generate the target vector from the label, one for the correct digit, zeros for the wrong digits
    # the element type of the image is Float32, so the target vector should have the same element type
    target = zeros(Float32, 10) 
    target[label + 1] = 1.f0

    return image, target
end

# initialize the data loaders
train_data_loader = DataLoader(index -> get_element(index, "train"), length(mnist_train), batch_size=32, shuffle=true)
test_data_loader = DataLoader(index -> get_element(index, "test"), length(mnist_test), batch_size=32)

# force the data loaders to load all the data at once into memory, depending on the dataset's size, this may take a while
# train_data = train_data_loader[begin:end] # turned off to save time
# test_data = test_data_loader[begin:end] # turned off to save time

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
                                  # a softmax activation layer, the softmax will be calculated along the first dimension (the features dimension)
                                  Softmax(dims=1)])
# The final model consists of three different submodules, 
# which shows that a SequentialContainer can contain not only layers, but also other SequentialContainers
model = SequentialContainer([feature_extractor, flatten, classifier])

# After a model is initialized, its parameters are Float32 arrays by default. The input to the model must always be of the same element type as its parameters!
# You can change the device (CPU/GPU) and element type of the model's parameters with the function module_to_eltype_device!
# The element type of our data (image/target) is already Float32 and because this LeNet is such a small model, using the CPU is just fine.

# printing a nice looking summary of the model
summary, num_params = summarize_model(model)
println(summary)

# defining hyperparameters
loss_function = mse_loss # mean squared error
learning_rate = 0.05
optimizer = MSGD(model, learning_rate, momentum=0.5) # momentum stochastic gradient descent with a momentum of 0.5
epochs = 5 # 5 or 10, for example

# evaluate the model's accuracy
function test()
    num_correct_preds = 0
    avg_test_loss = 0
    for (batch, (images_batch, targets_batch)) in enumerate(test_data_loader)
        # computing predictions
        predictions_batch = model(images_batch) # equivalent to forward(model, images_batch)
        # checking for each image in the batch individually if the prediction is correct
        batch_size = size(predictions_batch)[end] # the batch dimension is always the last dimension
        for index_batch in 1:batch_size
            single_prediction = predictions_batch[:, index_batch]
            single_target = targets_batch[:, index_batch]
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
                predictions_batch = model(images_batch) # equivalent to forward(model, images_batch)
                # backpropagation
                zero_gradients(model) # reset the gradients
                loss, derivative_loss = loss_function(predictions_batch, targets_batch)
                backward(model, derivative_loss) # compute the gradients
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

# when this file is run as the main script, 
# then train() is run and the final model will be saved using a package called BSON.jl
if abspath(PROGRAM_FILE) == @__FILE__
    train()
    file_name = "MNIST_with_LeNet5_model.bson"
    @save file_name model
    println("Saved trained model as $file_name")
end

#= Use this code in another file:
# load the model and make some individual predictions

using GradValley
using GradValley.Layers 
using GradValley.Optimization
using MLDatasets
using BSON: @load

# load the pre-trained model
@load "MNIST_with_LeNet5_model.bson" model

# make some individual predictions
mnist_test = MNIST(:test)
for i in 1:5
    random_index = rand(1:length(mnist_test))
    image, label = mnist_test[random_index]
    # remember to add batch and channel dimensions and to rescale the image as was done during training and testing
    image_batch = reshape(image, 28, 28, 1, 1) .* 255
    prediction = model(image_batch)
    predicted_label = argmax(prediction[:, 1]) - 1
    println("Predicted label: $predicted_label, Correct Label: $label")
end
=#