# Reference

!!! note
    Note that for all keyword arguments of type `NTuple{2, Int}`, the order of dimensions is (y/height-dimension, x/width-dimension).

```@contents
Pages = ["reference.md"]
Depth = 4
```

## GradValley

### DataLoader
```@docs
DataLoader
reshuffle!
```

## GradValley.Layers

### Containers
```@docs
SequentialContainer
GraphContainer
summarize_model
module_to_eltype_device!
clean_model_from_backward_information!
```

### Forward- and Backward-Pass
```@docs
forward
backward
```

### Reset/zero gradients
```@docs
zero_gradients
```

### Training mode/test mode
```@docs
trainmode!
testmode!
```

### Convolution
```@docs
Conv
ConvTranspose
```

### Pooling
```@docs
MaxPool
AvgPool
AdaptiveMaxPool
AdaptiveAvgPool
```

### Fully connected
```@docs
Fc
```

### Identity
```@docs
Identity
```

### Normalization
```@docs
BatchNorm
```

### Reshape / Flatten
```@docs
Reshape
```

### Activation functions
Almost every layer constructor has the keyword argument `activation_function` specifying the element-wise activation function. `activation_function` can be `nothing` or a string. `nothing` means no activation function, a string gives the name of the activation. Because softmax isn’t a simple element-wise activation function like the most activations, [`Softmax`](@ref) has it’s own layer. The following element-wise activation functions are currently implemented:  
- `"relu"`: applies the element-wise relu activation (recified linear unit): ``f(x) = ``
- `"sigmoid"`: applies the element-wise sigmoid acivation (logistic curve): ``f(x) = ``
- `"tanh"`: applies the element-wise tanh activation (tangens hyperbolicus): ``f(x) = ``
- `"leaky_relu"`: applies a element-wise leaky relu activation with a negative slope of 0.01: ``f(x) = ``
- `"leaky_relu:$(negative_slope)"`: applies a element-wise leaky relu activation with a negative slope of `negative_slope` (e.g. `"leaky_relu:0.2"` means a leaky relu activation with a negative slope of 0.2): ``f(x) = ``

### Special activation functions
```@docs
Softmax
```

## GradValley.Optimization

### Optimizers
```@docs
Adam
SGD
MSGD
Nesterov
```
### Optimization step function
```@docs
step!
```

### Loss functions
```@docs
mae_loss
mse_loss
bce_loss
```

## GradValley.Functional
GradValley.Functional contains many primitives common for various neuronal networks. Not all functions are documented because they are mostly used only internally (not by the user). 

```@docs
GradValley.Functional.zero_pad_nd
GradValley.Functional.zero_pad_2d
```