# Saving and Loading

There doesn't exist *the one* right way to save and load models. However, at the moment, the [JLD2](https://github.com/JuliaIO/JLD2.jl) package is recommended.
In the MNIST-Tutorial however, the [BSON](https://github.com/JuliaIO/BSON.jl) package was used. But this package has problems with very large files, for example with large [ResNets](https://arxiv.org/pdf/1512.03385.pdf) (e.g. the [pre-trained ResNets](https://jonas208.github.io/GradValley.jl/(pre-trained)_models/) in the [(Pre-Trained) Models](@ref) section).

Because GradValley saves some information for the backward pass (e.g. gradients) directly in the layers, it is highly recommended to run the [`clean_module_from_backward_information!`](@ref) on the model first. Otherwise, the files may get larger than they would have to.
Then, you can save the model in the JLD2 file format with the [FileIO](https://github.com/JuliaIO/FileIO.jl) package:

```julia
# import all packages 
using GradValley
using GradValley.Layer
using FileIO # the recommended package for saving/loading models
# define a model as an example
model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# recommended: run clean_model_from_backward_information! on the model (doesn't necessary in this specific case because no forward/backward pass was run before)
clean_module_from_backward_information!(model)
# save the model to a file 
file_name = "my_example_model.jld2"
save(file_name, Dict("model" => model))
```

Loading the model is then normally done in another file. Note that all used packages that were used in connection with the saved model must be also imported in the script where the file is loaded again (in this case GradValley/GradValley.Layers).

```julia
# import all used packages 
using GradValley
using GradValley.Layer
using FileIO 
# load the model from a file
file_name = "my_example_model.jld2"
model = load(file_name, "model")
# test if the model works correctly
forward(model, rand(32, 1000))
```

It is heavily recommended to run this file (or the file in which you inlcude ResNets.jl), and any other files using GradValley, with multiple threads.
Using multiple threads can make training much faster.
To do this, use the -t option when running a julia script in terminal/PowerShell/command line/etc.
If your CPU has 24 threads, for example, then run:
julia -t 24 ./ResNets.jl
The specified number of threads should match the number of threads your CPU provides.