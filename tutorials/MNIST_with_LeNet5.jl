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
    image = reshape(image, 1, 28, 28) .* 255
    # generate the target vector from the label, one for the correct digit, zeros for the wrong digits
    targets = zeros(10)
    targets[label + 1] = 1.00

    return convert(Array{Float64, 3}, image), targets
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
                                  # a softmax activation layer, the softmax will be calculated along the second dimension (the features dimension)
                                  Softmax(dim=2)])
# The final model consists of three different submodules, 
# which shows that a SequentialContainer can contain not only layers, but also other SequentialContainers
model = SequentialContainer([feature_extractor, flatten, classifier])

# printing a nice looking summary of the model
summary, num_params = summarize_model(model)
println(summary)

# defining hyperparameters
loss_function = mse_loss # mean squared error
learning_rate = 0.05
optimizer = MSGD(model, learning_rate, momentum=0.5) # momentum stochastic gradient descent with a momentum of 0.5
epochs = 5 # 5 or 10

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

# when this file is run as the main script using julia MNIST_with_LeNet5.jl, 
# then train() is run and the final model will be saved using a package called BSON.jl
if abspath(PROGRAM_FILE) == @__FILE__
    train()
    file_name = "MNIST_with_LeNet5_model.bson"
    @save file_name model
    println("Saved trained model as $file_name")
end