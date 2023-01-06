# GradValley.jl
A new lightweight package for deep learning in Julia

![My Image](logo.png)

## ATTENTION, IMPORTANT INFORMATION: THIS REPOSITORY IS CURRENTLY UNDER CONSTRUCTION, IT IS NOT READY FOR USE YET!!

GradValley.jl is a new lightweight module for deep learning in 100% Julia. It offers a high level interface for model building and training. It is completely independent from other machine learning packages like [Flux](https://github.com/FluxML/Flux.jl), [Knet](https://github.com/denizyuret/Knet.jl), [NNlib](https://github.com/FluxML/NNlib.jl) or [NNPACK](https://github.com/Maratyszcza/NNPACK). It is based on Julia's standard array type and needs no additional tensor type. <br>
GradValley.jl's backend is written "human-friendly". So if you're looking into how exactly such deep learning algorithms work, looking at the code could also be a good learning resource. See [this page](https://jonas208.github.io/GradValley.jl/) in [documentation](https://jonas208.github.io/GradValley.jl/) for further information. <br>
To get started, see [Installation](https://github.com/jonas208/GradValley.jl/edit/main/README.md#installation) and [First Steps](https://github.com/jonas208/GradValley.jl/edit/main/README.md#first-steps).

The [documentation](https://jonas208.github.io/GradValley.jl/) can be found on the GitHub Pages site of this repository: https://jonas208.github.io/GradValley.jl/ <br>
Further [tutorials and examples](https://jonas208.github.io/GradValley.jl/) can be also found there.

#### Note: This software package is in an early stage of development and is therefore still a beta version.

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

# First Steps

# Documentation, Tutorials and Examples
- The [documentation](https://jonas208.github.io/GradValley.jl/) can be found on the GitHub Pages site of this repository: https://jonas208.github.io/GradValley.jl/ <br>
- Further [tutorials and examples](https://jonas208.github.io/GradValley.jl/) can be also found there.
- Information about [pre-trained models](https://jonas208.github.io/GradValley.jl/) can be found there too.

# Contributing
Everyone is invited to contribute. To do so:

- Fork the current-most state of the master branch
- Implement features or changes
- Add your name to AUTHORS.md
- Create a pull-request to the upstream repository

# License
The GradValley.jl software package is currently published under the MIT "Expat" license. See LICENSE for further information.
