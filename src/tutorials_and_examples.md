# Tutorials and Examples

Here you can find detailed explanations on how to build and train specific models with GradValley.jl.

## A LeNet-like model for handwritten digit recognition

In this tutorial, we will learn the basics of GradValley.jl while building a model for handwritten digit recognition reaching approximately 99% accuracy on the MNIST-dataset.
The whole code at once can be found [here]().

### Importing modules

```julia
using GradValley # The master module of GradValley.jl
using GradValley.Layers # The "Layers" module provides all the building blocks for creating a model. 
using GradValley.Optimization # The "Optimization" module provides different loss functions and optimizers.
```

### Using the dataset

We will using the MLDatasets package which downloads the MNIST-dataset for us automatically.
If you haven't installed MLDatasets yet, write this for [installation](https://juliaml.github.io/MLDatasets.jl/stable/#Installation):
```julia
import Pkg; Pkg.add("MLDatasets")
```
Then we can import MLDatasets:
```julia
using MLDatasets # A package for downloading datasets
```

#### Splitting up the dataset in a train- and a test-partition

The MNIST-dataset contains 70,000 images, we will use 60,000 images for training the network and 10,000 images for evaluating accuracy.

```julia
# Initialize train- and test-dataset
mnist_train = MNIST(:train) 
mnist_test = MNIST(:test)
```

#### Using GradValley.DataLoader for handling data

A typical workflow when dealing with datasets is to use the GradValley.DataLoader struct. A data loader makes it easy to iterate directly over the batches in a dataset. 
Due to better memory efficiency, the data loader loads the batches *just in time*. When initializing a data loader, we specify a function that returns exactly one element from the dataset at a given index.
We also have to specify the size of the dataset (e.g. the number of images). All parameters that the dataloader accepts (see [Reference]() for more information):
```julia
DataLoader(get_function::Function, dataset_size::Integer; batch_size::Integer=1, shuffle::Bool=false, drop_last::Bool=false)
```
Now we write the *get function* for the two data loaders.
```julia
# function for getting an image and the corressponding target vector from the train or test partition
function get_element(index, partition)
    # load one image and the corresponding label
    if partition == "train"
        image, label = mnist_train[index]
    else # test partition
        image, label = mnist_test[index]
    end
    # add channel dimension
    image = reshape(image, 1, 28, 28)
    # generate the target vector from the label, one for the correct digit, zeros for the wrong digits
    targets = zeros(10)
    targets[label + 1] = 1.00

    return image, label
end
```
We can now initialitze the data loaders.
```julia
train_data_loader = DataLoader(index -> get_element(index, "train"), length(mnist_train), batch_size=32)
test_data_loader = DataLoader(index -> get_element(index, "test"), length(mnist_test), batch_size=32)
```
If you want to force the data loader to load the data all at once, you could do:
```julia
# depending on the dataset's size, this may take a while
train_data = train_data_loader[begin:end]
test_data = test_data_loader[begin:end]
```

### Building the neuronal network aka. the model

The most recommend way to build models is to use the GradValley.Layers.SequentialContainer struct. A SequtialContainer can take an array of layers or other SequentialContainers (sub-models).
While forward-pass, the given inputs are *sequentially* propagated through every layer (or sub-model) and the output will be returned. For more details, see [Reference]().
The LeNet5 model is one of the earliest convolutional neuronal networks (CNNs) reaching approximately 99% accuracy on the MNIST-dataset.
The LeNet5 is build from two main parts, the feature extractor and the classifier. So it would be a good idea to clarify that in the code:
```julia
# Definition of a LeNet-like model consisting of a feature extractor and a classifier
feature_extractor = SequentialContainer([ # a convolution layer with 1 in channel, 6 out channels, a 5*5 kernel and a relu activation
                                         Conv(1, 6, (5, 5), activation_function="relu"),
                                         # a average pooling layer with a 2*2 filter (when not specified, stride is automatcally set to kernel size)
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
# The final model consists of three different sub-modules, 
# which shows that a SequentialContainer can contain not only layers, but also other SequentialContainers
model = SequentialContainer([feature_extractor, flatten, classifier])
```

#### Defining hyperparameters

Before we start to train and test the model, we define all necessary hyperparamters.
If we want to change the learning rate or the loss function for example, this is the one place to do this.
```julia
# defining hyperparameters
loss_function = mse_loss # mean squared error
learning_rate = 0.05
optimizer = MSGD(model, learning_rate, momentum=0.5) # momentum stochastic gradient decent with a momentum of 0.5
epochs = 20
```

### Train the model