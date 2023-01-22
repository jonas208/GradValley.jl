# Home

Welcome to the GradValley.jl documentation!

GradValley.jl is a new lightweight module for deep learning written in 100% Julia. It offers a high level interface for model building and training. It is completely independent from other machine learning packages like [Flux](https://github.com/FluxML/Flux.jl), [Knet](https://github.com/denizyuret/Knet.jl), [NNlib](https://github.com/FluxML/NNlib.jl) or [NNPACK](https://github.com/Maratyszcza/NNPACK). It is based on Julia's standard array type and needs no additional tensor type.
GradValley.jl's "backend" is written "human-friendly". So if you're looking into how exactly such deep learning algorithms work, looking at the [source code](https://github.com/jonas208/GradValley.jl/tree/main/src) could also be a good learning resource. See [Learning](@ref) for further information.
To get started, see [Installation](@ref) and [Getting Started](@ref). After that, you could look at the [Tutorials and Examples](@ref) section.

!!! note
    This software package and its documentation are in an early stage of development and are therefore still a beta version. If you are missing certain features, see [Current Limitations](@ref) for planned future features, or directly share your ideas in the [discussion](https://github.com/jonas208/GradValley.jl/discussions) section of the GitHub [repository](https://github.com/jonas208/GradValley.jl).

!!! warning
    **This documentation is currently under construction. It currently does not claim to be complete, it is clear that some entries and information are still missing. It is currently being continuously adapted and improved.**

## About
A while ago I started looking into machine learning. The topic fascinated me from the beginning, so I wanted to gain a deeper understanding of the way such models work. In my opinion, the best way to do this is to write your own small software package for machine learning and not just blindly use one of the big, established frameworks such as PyTorch or TensorFlow. The Julia programming language was my choice because of its popularity in academia and its very good performance compared to pure Python, which is after all very popular in the world of artificial intelligence.
The product of this work is this module called GradValley.jl with which various current neural networks (e.g. CNNs) can be implemented easily and intuitively.

### Array structure convention:
The order used in GradValley for processing images (or similar data) is NCHW, where N is the batch dimension, C is the channel dimension, H is the vertical and W is the horizontal size of the image. In this regard, GradValley differs from some other deep learning packages in Julia and is similar to PyTorch instead. This makes the migration of pre-trained models from PyTorch or from the Python world in general to GradValley much easier.

### Explanation of the name "GradValley":
When optimizing the weights of a machine learning model, an attempt is always made to find the best possible error minimum. The derivatives, i.e. the gradients, of the error function in relation to the weights are required for this. So the goal is to find the "valley" of the error using the gradients ("grad" stands for gradient). That's why it's called GradValley.

### Current Limitations
Due to the relatively early development status of this software, no GPU support is currently offered. GradValley.jl doesn't provide a real automatic differentiation (AD) engine like [PyTorch](https://pytorch.org/) does, for example. However, in the case of this software package, it is not really necessary to have real AD. Model building is mostly done with the [`SequentialContainer`](@ref), this clearly defines the forward pass and thus the backward pass is also known to the software.
The following features are planned and likely to be added in the future:
- ConvTranspose as a upsampling layer
- more predefined activation function, loss functions and optimizers
- an easy way to use skipped connections (for ResNet for example)
- implementation of a GraphContainer as an alternative to SequentialContainer that allows a more flexible signal flows using AD

## GitHub Repository
In the GitHub [repository of GradValley.jl](https://github.com/jonas208/GradValley.jl), you can find e.g. the source code, the source of this documentation and information about continues testing and it's code coverage. The repo is also the place to ask questions and share your thoughts about this project.
[Contributing](@ref) or opening issues is of course also welcome. (This documentation page is also hosted on GitHub using GitHub Pages.)

## Questions and Discussions
If you have any questions about this software package, please let me know in the [discussion](https://github.com/jonas208/GradValley.jl/discussions) section of the GitHub [repository](https://github.com/jonas208/GradValley.jl).

## Contributing
Everyone is invited to contribute. To do so:

- Fork the current-most state of the main branch
- Implement features or changes
- Add your name to AUTHORS.md
- Create a pull-request to the upstream repository

## License
The GradValley.jl software package is currently published under the MIT "Expat" license. See [LICENSE](https://github.com/jonas208/GradValley.jl/blob/main/LICENSE) in the GitHub [repository](https://github.com/jonas208/GradValley.jl) for further information.