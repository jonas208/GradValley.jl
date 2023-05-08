# Reference

!!! warning
    For some (mostly internal) functions, the documentation is still missing because this documentation is still under construction!

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
clean_module_from_backward_information!
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
DepthwiseConv
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
BatchNorm2d
```

### Reshape / Flatten
```@docs
Reshape
```

### Special activation functions
```@docs
Softmax
```

## GradValley.Optimization

### Optimizers
```@docs
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
```

## GradValley.Functional
GradValley.Functional contains many primitives common for various neuronal networks. Not all functions are documented because they are mostly used only internally (not by the user). 

```@docs
GradValley.Functional.zero_pad_nd
GradValley.Functional.zero_pad_2d
GradValley.Functional.convolution2d!
GradValley.Functional.convolution2d
GradValley.Functional.convolution2d_data_backward!
GradValley.Functional.convolution2d_data_backward
GradValley.Functional.deconvolution2d!
GradValley.Functional.deconvolution2d
```