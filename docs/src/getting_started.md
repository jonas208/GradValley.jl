# Getting Started

## First Impressions

This example shows the basic workflow on model building and how to use loss functions and optimizers to train the model:
```julia
using GradValley
using GradValley.Layers # The "Layers" module provides all the building blocks for creating a model.
using GradValley.Optimization # The "Optimization" module provides different loss functions and optimizers.

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
                                  
# feeding the network with some random data
# After a model is initialized, its parameters are Float32 arrays by default. The input to the model must always be of the same element type as its parameters!
# You can change the device (CPU/GPU) and element type of the model's parameters with the function module_to_eltype_device!
input = rand(Float32, 28, 28, 1, 32) # a batch of 32 images with one channel and a size of 28*28 pixels
prediction = model(input) # layers and containers are callable, alternatively, you can call the forward function directly: forward(model, input)

# choosing an optimizer for training
learning_rate = 0.05
optimizer = MSGD(model, learning_rate, momentum=0.5) # momentum stochastic gradient decent with a momentum of 0.5

# generating some random target data for a training step
target = rand(Float32, size(prediction)...) # remember to specify the correct element type here as well
# backpropagation
zero_gradients(model)
loss, derivative_loss = mse_loss(prediction, target) # mean squared error
backward(model, derivative_loss) # computing gradients
step!(optimizer) # making a optimization step with the calculated gradients and the optimizer
```

## First Real Project

Here are some suggestions to implement your first real project with GradValley.jl:
- The "Hello World" of Deep Learning: Try the Tutorial on training [A LeNet-like model for handwritten digit recognition](@ref).
- The [Reference](@ref): In the reference, you can find descriptions of all the layers, loss functions and optimizers.
- Download a pre-trained model: More [(Pre-Trained) Models](@ref) will likely be deployed over time.
- Look at more [Tutorials and Examples](@ref).