# Reference

!!! warning
    For some layers and functions, the documentation is still missing because this documentation is still under construction!

```@contents
Pages = ["reference.md"]
Depth = 4
```

## GradValley

## GradValley.Layers

### Containers
```@docs
SequentialContainer
GraphContainer
summarize_model
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

## GradValley.Functional