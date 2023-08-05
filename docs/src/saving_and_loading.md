# Saving and Loading

There doesn't exist *the one* right way to save and load models. However, at the moment, the [JLD2](https://github.com/JuliaIO/JLD2.jl) package is recommended.
In the MNIST-Tutorial however, the [BSON](https://github.com/JuliaIO/BSON.jl) package was used. But this package has problems with very large files, for example with large [ResNets](https://arxiv.org/pdf/1512.03385.pdf) (e.g. the [pre-trained ResNets](https://jonas208.github.io/GradValley.jl/(pre-trained)_models/) in the [(Pre-Trained) Models](@ref) section).

Because GradValley saves some information for the backward pass (e.g. the during forward pass recorded computational graph) directly in the containers, it is highly recommended to run [`clean_model_from_backward_information!`](@ref) on the model first. Otherwise, the files may get larger than they would have to.
If the model was moved to the GPU, it's also heavily recommend to move the model to the CPU before saving it to a file. To move the model back to the CPU, use [`module_to_eltype_device!`](@ref).
Then, you can save the model in the JLD2 file format with the [FileIO](https://github.com/JuliaIO/FileIO.jl) package:

```julia
# import all packages 
using GradValley
using GradValley.Layer
using FileIO # the recommended package for saving/loading models
# define a model as an example
model = SequentialContainer([Fc(1000, 500), Fc(500, 250), Fc(250, 125)])
# recommended: run clean_model_from_backward_information! on the model (doesn't necessary in this specific case because no forward/backward pass was run before)
clean_model_from_backward_information!(model)
# recommended: run module_to_eltype_device! with kw-arg device="cpu" on the model (doesn't necessary in this specific case because the model wasn't moved to the GPU before)
module_to_eltype_device!(model, device="cpu")
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
# you can run module_to_eltype_device! to make sure the parameter's element type is correct
module_to_eltype_device!(model, element_type=Float32, device="cpu")
# test if the model works correctly
input = rand(Float32, 1000, 32)
output = model(input)
```