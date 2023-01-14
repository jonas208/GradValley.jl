# GradValley.jl
A new lightweight package for deep learning with Julia

![My Image](logo.png)

## ATTENTION, IMPORTANT INFORMATION: THIS REPOSITORY IS CURRENTLY UNDER CONSTRUCTION, IT IS NOT READY FOR USE YET!!

GradValley.jl is a new lightweight module for deep learning written in 100% Julia. It offers a high level interface for model building and training. It is completely independent from other machine learning packages like [Flux](https://github.com/FluxML/Flux.jl), [Knet](https://github.com/denizyuret/Knet.jl), [NNlib](https://github.com/FluxML/NNlib.jl) or [NNPACK](https://github.com/Maratyszcza/NNPACK). It is based on Julia's standard array type and needs no additional tensor type. <br>
GradValley.jl's backend is written "human-friendly". So if you're looking into how exactly such deep learning algorithms work, looking at the [source code](https://github.com/jonas208/GradValley.jl/tree/main/src) could also be a good learning resource. See [this page in documentation](https://jonas208.github.io/GradValley.jl/learning/) for further information. <br>
To get started, see [Installation](https://github.com/jonas208/GradValley.jl/blob/main/README.md#installation) and [Getting Started](https://github.com/jonas208/GradValley.jl/blob/main/README.md#getting-started).

The [documentation](https://jonas208.github.io/GradValley.jl/) can be found on the GitHub Pages site of this repository: https://jonas208.github.io/GradValley.jl/ <br>
Further [tutorials and examples](https://jonas208.github.io/GradValley.jl/tutorials_and_examples/) can be also found there.

#### Note: This software package and its documentation are in an early stage of development and are therefore still a beta version.

# Installation
Use Julias's package manager in the REPL:
```
pkg> add https://github.com/jonas208/GradValley.jl
```
Or install directly in a julia script:
```julia
import Pkg
Pkg.add(url="https://github.com/jonas208/GradValley.jl")
```

# Getting Started
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
                                  # a softmax activation layer, the softmax will be calculated along the second dimension (the features dimension)
                                  Softmax(dim=2)])
# The final model consists of three different submodules, 
# which shows that a SequentialContainer can contain not only layers, but also other SequentialContainers
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

# Documentation, Tutorials and Examples
- The [documentation](https://jonas208.github.io/GradValley.jl/) can be found on the GitHub Pages site of this repository: https://jonas208.github.io/GradValley.jl/ <br>
- Further [tutorials and examples](https://jonas208.github.io/GradValley.jl/tutorials_and_examples/) can be also found there.
- Information about [pre-trained models](https://jonas208.github.io/GradValley.jl/(pre-trained)_models/) can be found there too.

# Questions in Discussions
If you have any questions about this software package, please let me know in the [discussion](https://github.com/jonas208/GradValley.jl/discussions) section of this repository.

# Contributing
Everyone is invited to contribute. To do so:

- Fork the current-most state of the main branch
- Implement features or changes
- Add your name to AUTHORS.md
- Create a pull-request to the upstream repository

# License
The GradValley.jl software package is currently published under the MIT "Expat" license. See LICENSE for further information.
