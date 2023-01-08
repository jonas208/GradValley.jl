# Installation

The package can be installed with the Julia package manager. From the Julia REPL, type ```]``` to enter the Pkg REPL mode and run:
```
pkg> add https://github.com/jonas208/GradValley.jl
```
Or, equivalently, via the Pkg API:
```julia
julia> import Pkg; Pkg.add(url="https://github.com/jonas208/GradValley.jl")
```

## Used Dependencies
GradValley.jl uses two packages which are inbuilt in the Julia programming language:
- [Random Numbers](https://docs.julialang.org/en/v1/stdlib/Random/) (no specific version)
- [Linear Algebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) (no specific version)
Besides that, GradValley.jl uses one external package:
- [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl) (at least v0.12.146)
You can also look at the [Project.toml](https://github.com/jonas208/GradValley.jl/blob/main/Project.toml) file to find information about used dependencies and compatibility.
### All used dependencies will be automatically installed due installation of GradValley.jl.