# GradValley.jl
A new lightweight module for deep learning in Julia

![My Image](logo.png)

### ATTENTION, IMPORTANT INFORMATION: THIS REPOSITORY IS CURRENTLY UNDER CONSTRUCTION, IT IS NOT READY FOR USE YET!

GradValley.jl is a new lightweight module for deep learning in 100% Julia. It offers a high level interface for model building and training. It is completely independent from other machine learning packages like [Flux](https://github.com/FluxML/Flux.jl), [Knet](https://github.com/denizyuret/Knet.jl), [NNlib](https://github.com/FluxML/NNlib.jl) or [NNPACK](https://github.com/Maratyszcza/NNPACK). It is based on Julia's standard array type and needs no additional tensor type. <br>
GradValley.jl's backend is written "human-friendly". So if you're looking into how exactly such deep learning algorithms work, looking at the code could also be a good learning resource. See [documentation](https://github.com/jonas208/GradValley.jl/wiki) for further information. <br>
To get started, see Installation and First steps. See the [models folders](https://github.com/jonas208/GradValley.jl/tree/main/models) or the Tutorials page in the Wiki for examples.

The Wiki page of this repository provides a documentation for the GradValley.jl software package. Further Tutorials can be also found there.

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

# Contributing
Everyone is invited to contribute. To do so:

- Fork the current-most state of the master branch
- Implement features or changes
- Add your name to AUTHORS.md
- Create a pull-request to the upstream repository

# License
The GradValley.jl software package is currently published under the MIT "Expat" license. See LICENSE for further information.
