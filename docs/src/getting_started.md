# Getting Started

## First Impressions

This example shows the basic workflow on model building and how to use loss functions and optimizers to train the model:
```julia
using GradValley
using GradValley.Layers # The "Layers" module provides all the building blocks for creating a model.
using GradValley.Optimization # The "Optimization" module provides different loss functions and optimizers.

# Definition of a LeNet-like model consisting of a feature extractor and a classifier
feature_extractor = SequentialContainer([Conv(1, 6, (5, 5), activation_function="relu"),
                                         AvgPool((2, 2)),
                                         Conv(6, 16, (5, 5), activation_function="relu"),
                                         AvgPool((2, 2))])
flatten = Reshape((256, ))
classifier = SequentialContainer([Fc(256, 120, activation_function="relu"),
                                  Fc(120, 84, activation_function="relu"),
                                  Fc(84, 10),
                                  Softmax(dim=2)])
# The final model consists of three different sub-modules, which shows that a SequentialContainer can contain not only layers, but also other SequentialContainers
model = SequentialContainer([feature_extractor, flatten, classifier])
                                  
# feeding the network with some random data
input = rand(32, 1, 28, 28) # a batch of 32 images with one channel and a size of 28*28 pixels
prediction = forward(model, input) # the forward function can work with a layer or a SequentialContainer

# choosing an optimizer for training
learning_rate = 0.05
optimizer = MSGD(model, learning_rate, momentum=0.5) # momentum stochastic gradient decent with a momentum of 0.5

# generating some random data for a training step
target = rand(size(prediction)...)
# backpropagation
zero_gradients(model)
loss, derivative_loss = mse_loss(prediction, target) # mean squared error
backward(model, derivative_loss) # computing gradients
step!(optimizer) # making a optimization step with the calculated gradients and the optimizer
```

## First Real Project

Here are some suggestions to implement your first real project with GradValley.jl:
- The "Hello World" of Deep Learning: Try [the Tutorial on training a LeNet-like model](https://jonas208.github.io/GradValley.jl/) for handwritten digit recognition.
- The [Reference]((https://jonas208.github.io/GradValley.jl/)): In the reference, you can find descriptions of all the layers, loss functions and optimizers.
- Download a pre-trained model: More [pre-trained models](https://jonas208.github.io/GradValley.jl/) will likely be deployed over time.
- Look at more [examples and tutorials](https://jonas208.github.io/GradValley.jl/).