# Home

Welcome to the GradValley.jl documentation!

GradValley.jl is a new lightweight package for Deep Learning written in 100% Julia. GradValley offers a high level interface for flexible model building and training. It is independent of other machine learning packages like [Flux](https://github.com/FluxML/Flux.jl), [Knet](https://github.com/denizyuret/Knet.jl), [NNlib](https://github.com/FluxML/NNlib.jl) or [NNPACK](https://github.com/Maratyszcza/NNPACK) (see [dependencies](https://github.com/jonas208/GradValley.jl/blob/main/Project.toml)). It is based on Julia’s standard array type and needs no additional tensor type.
To get started, see [Installation](@ref) and [Getting Started](@ref). After that, you could look at the [Tutorials and Examples](@ref) section. Or directly start using a [pre-trained model](https://jonas208.github.io/GradValley.jl/dev/(pre-trained)_models/), for example a [pre-trained ResNet](https://jonas208.github.io/GradValley.jl/dev/(pre-trained)_models/#(Pre-Trained)-Models).

Because GradValley is just 100% high level Julia code, the implemented backend algorithms powering Deep Learning (e.g. convolution) are pretty nice to read. So if you're looking into how exactly such Deep Learning algorithms work, looking at the [source code](https://github.com/jonas208/GradValley.jl/tree/main/src) (and at it's documentation in [Reference](@ref)) could also be a helpful learning resource. See [Learning](@ref) for further learning resources. 

!!! note
    This software package and its documentation are in an early stage of development and are therefore still a beta version. If you are missing certain features, see [Current Limitations](@ref) for planned future features, or directly share your ideas in the [discussion](https://github.com/jonas208/GradValley.jl/discussions) section of the GitHub [repository](https://github.com/jonas208/GradValley.jl). This software package and its documentation are currently being continuously adapted and improved.

## Why GradValley.jl
- **Intuitive Model Building:** Model building is normally done using [Containers](@ref). With [Containers](@ref), large models can be broken down into smaller components (e.g. ResNets in ResBlocks), which in turn can then be easily combined into one large model. See the [ResNets example](https://jonas208.github.io/GradValley.jl/dev/tutorials_and_examples/#Generic-ResNet-(18/34/50/101/152)-implementation) in the [Tutorials and Examples](@ref) section.
- **Flexible:** [Containers](@ref) behave like layers, so you can use containers in containers in containers... (arbitrary nesting allowed). GraphContainer's automatic differentiation allows defining your own computational graph in a function, which then can be automatically differentiated during backward pass (using reverse mode AD, aka [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)).
- **Switching from Python to Julia:** Model building is very similar to other frameworks and and the behavior of the layers is strongly oriented towards PyTorch, e.g. the algorithm behind adaptive pooling. 
- **100% Julia:** Julia's biggest advantage compared to Python is speed. This allows you to easily extend existing Julia packages yourself. Extending python packages is, at least if they use e.g. C code in the backend, much more difficult. 
- **Julia's environment**: The Julia community developed a lot of awesome packages. Julia packages have the advantage that they can be usually always used very well together. For example, take a look at [Flux.jl](https://github.com/FluxML/Flux.jl), [Plots.jl](https://github.com/JuliaPlots/Plots.jl), [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl), [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) or [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).
- **Well documented**: The documentation aims to provide detailed information about all of GradValley’s functionalities. For example, the documentation of each layer contains e.g. a description, an argument list, a mathematical definition and extensive examples.
- **See for yourself:** To get started, see [Installation](@ref) and [Getting Started](@ref). After that, you could look at the [Tutorials and Examples](@ref) section. Or directly start using a [pre-trained model](https://jonas208.github.io/GradValley.jl/dev/(pre-trained)_models/), for exmaple a [pre-trained ResNet](https://jonas208.github.io/GradValley.jl/dev/(pre-trained)_models/).

## About
A while ago I started looking into machine learning. The topic fascinated me from the beginning, so I wanted to gain a deeper understanding of the way such models work. In my opinion, the best way to do this is to write your own small software package for machine learning and not just blindly use one of the big, established frameworks such as PyTorch or TensorFlow. The Julia programming language was my choice because of its popularity in academia and its very good performance compared to pure Python, which is after all very popular in the world of artificial intelligence.
The product of this work is this package called GradValley.jl with which various current neural networks (e.g. CNNs) can be implemented easily and intuitively.

### Array structure convention
The order used in GradValley for processing images (or similar data) is WHCN, where N is the batch dimension, C is the channel dimension, H is the vertical and W is the horizontal size of the image. The batch dimension is always the last. 

### Explanation of the name "GradValley"
When optimizing the weights of a machine learning model, an attempt is always made to find the best possible error minimum. The derivatives, i.e. the gradients, of the error function in relation to the weights are required for this. So the goal is to find the "valley" of the error using the gradients ("grad" stands for gradient). That's why it's called GradValley.

### Current Limitations
The following features are planned and likely to be added in the future:
- more predefined activation function, loss functions and optimizers
- further performance improvments for the cpu 

## GitHub Repository
In the GitHub [repository of GradValley.jl](https://github.com/jonas208/GradValley.jl), you can find e.g. the source code, the source of this documentation and information about continues testing and it's code coverage. The repo is also a place to ask questions and share your thoughts about this project.
[Contributing](@ref) or opening issues is of course also welcome. (This documentation page is also hosted on GitHub using GitHub Pages.)

## Questions and Discussions
If you have any questions about this software package, please let us know. For example, use the [discussion](https://github.com/jonas208/GradValley.jl/discussions) section of the GitHub [repository](https://github.com/jonas208/GradValley.jl).
Or interact with the community on the [Julia Discourse Forum](https://discourse.julialang.org/) (specific domains > Machine Learning), the [Julia Slack](https://julialang.org/slack/) (channel #machine-learning) or the [Julia Zulip](https://julialang.zulipchat.com/register/) (stream #machine-learning).
The [Julia Discourse Forum](https://discourse.julialang.org/) is usually the preferred place for asking questions. 

## Contributing
Contributors are more than welcome! A proper guide for contributors will be added soon. Normally the rough procedure is as follows:
- Fork the current-most state of the main branch
- Implement features or changes
- Add your name to AUTHORS.md
- Create a pull-request to the repository

## License
The GradValley.jl software package is currently published under the MIT "Expat" license. See [LICENSE](https://github.com/jonas208/GradValley.jl/blob/main/LICENSE) in the GitHub [repository](https://github.com/jonas208/GradValley.jl) for further information.