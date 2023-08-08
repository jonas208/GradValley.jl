# GPU Support

GradValley.jl supports [CUDA](https://github.com/JuliaGPU/CUDA.jl) capable Nvidia GPUs. GPU support is made possible by [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and [cuDNN.jl](https://github.com/JuliaGPU/CUDA.jl/tree/master/lib/cudnn).
Layers, containers, loss functions and optimizers work for GPUs exactly the same as they do for the CPU. 
To move a model (single layer or container) to the GPU, use [`module_to_eltype_device!`](@ref) and set the keyword argument `device` to `"gpu"`. 
Because on the GPU, `Float32` is usually much faster than `Float64`, always use `Float32` if you don't need more precision. If your model has been moved to the GPU, the input to your model and the target values must be of type [`CuArray`](https://cuda.juliagpu.org/stable/usage/overview/#The-CuArray-type).
To learn more about how to use GPUs in Julia, see the [website of JuliaGPU](https://juliagpu.org/) and the [documentation of the CUDA.jl package](https://cuda.juliagpu.org/stable/).

A common workflow to enable training and inference on both the GPU and CPU is to use [CUDA.functional()](https://cuda.juliagpu.org/stable/api/essentials/#Initialization) to check if a working GPU environment has been found.
If [CUDA.functional()](https://cuda.juliagpu.org/stable/api/essentials/#Initialization) is `true`, then the model and the input/target data can be moved to the GPU. This way, the code works not just for one type of device.

The following example describes how GPU and CPU ready code can look like. 
A real example that uses the GPU (if available) for training is the [DCGAN Tutorial](). With a RTX 3090, for example, it's possible to train the DCGAN on Celeb-A HQ (30,000 images) for 25 epochs in just 5 to 10 minutes (much faster than on the CPU)!

```julia
using GradValley
using GradValley.Layers
using GradValley.Optimization
using CUDA

# AlexNet-like model (without grouped convolution and dropout and with AvgPool instead of MaxPool)
feature_extractor = SequentialContainer([
    Conv(3, 64, (11, 11), stride=(4, 4), activation_function="relu"),
    # MaxPool((3, 3), stride=(2, 2)),
    AvgPool((3, 3), stride=(2, 2)),
    Conv(64, 192, (5, 5), padding=(2, 2), activation_function="relu"),
    # MaxPool((3, 3), stride=(2, 2)),
    AvgPool((3, 3), stride=(2, 2)),
    Conv(192, 384, (3, 3), padding=(1, 1), activation_function="relu"),
    Conv(384, 256, (3, 3), padding=(1, 1), activation_function="relu"),
    Conv(256, 256, (3, 3), padding=(1, 1), activation_function="relu"),
    # MaxPool((3, 3), stride=(2, 2))
    AvgPool((3, 3), stride=(2, 2))
])
classifier = SequentialContainer([
    AdaptiveAvgPool((6, 6)),
    Reshape((6 * 6 * 256, )), # 6 * 6 * 256 = 9.216
    Fc(9216, 4096, activation_function="relu"),
    Fc(4096, 4096, activation_function="relu"),
    Fc(4096, 1000),
    # Softmax(dims=1)
])
model = SequentialContainer([feature_extractor, classifier])

# define the element type you want to use 
dtype = Float32
# check if CUDA is available
use_cuda = CUDA.functional()
# move the model to the correct device and convert its parameters to the specified dtype
if use_cuda
    module_to_eltype_device!(model, element_type=dtype, device="gpu")
else    
    module_to_eltype_device!(model, element_type=dtype, device="cpu")
end

# create some random data for testing
input = rand(dtype, 224, 224, 3, 32)
target = rand(dtype, 1000, 32)
# move the data to the GPU if necessary
if use_cuda
    input, target = CuArray.((input, target))
end

# define some hyperparameters for testing
loss_fn = mse_loss
optim = Adam(model)

# forward pass
prediction = model(input)
# backward pass
zero_gradients(model)
loss, derivative_loss = loss_fn(prediction, target)
backward(model, derivative_loss)
# optimization step
step!(optim)
```