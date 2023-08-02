# using source for testing
include("C:/Users/joerg/Documents/pythonscripts/Neuronale Netze/NNJL/GradValley 5.1/GradValley/src/GradValley.jl")
using .GradValley, .GradValley.Functional, .GradValley.Layers, .GradValley.Optimization
using .GradValley.Functional: convolution2d, convolution2d_data_backward, convolution2d_filter_backward, convolution2d_bias_backward
using .GradValley.Functional: deconvolution2d, deconvolution2d_data_backward, deconvolution2d_filter_backward, deconvolution2d_bias_backward
using .GradValley.Functional: maximum_pooling2d, maximum_pooling2d_backward, average_pooling2d, average_pooling2d_backward
using .GradValley.Functional: adaptive_average_pooling2d, adaptive_average_pooling2d_backward
using .GradValley.Functional: adaptive_maximum_pooling2d, adaptive_maximum_pooling2d_backward
using .GradValley.Functional: batch_norm2d_forward, batch_norm2d_data_backward, batch_norm2d_weight_backward, batch_norm2d_backward
using .GradValley.Functional: reshape_forward, reshape_backward
using .GradValley.Functional: fc_forward, fc_backward
using .GradValley.Functional: softmax_forward, softmax_backward
# using installed package for testing
#=
using GradValley, GradValley.Functional, GradValley.Layers, GradValley.Optimization
using GradValley.Functional: convolution2d, convolution2d_data_backward, convolution2d_filter_backward, convolution2d_bias_backward
using GradValley.Functional: deconvolution2d, deconvolution2d_data_backward, deconvolution2d_filter_backward, deconvolution2d_bias_backward
using GradValley.Functional: maximum_pooling2d, maximum_pooling2d_backward, average_pooling2d, average_pooling2d_backward
using GradValley.Functional: adaptive_average_pooling2d, adaptive_average_pooling2d_backward
using GradValley.Functional: adaptive_maximum_pooling2d, adaptive_maximum_pooling2d_backward
using GradValley.Functional: batch_norm2d_forward, batch_norm2d_data_backward, batch_norm2d_weight_backward, batch_norm2d_backward
using GradValley.Functional: reshape_forward, reshape_backward
using GradValley.Functional: fc_forward, fc_backward
using GradValley.Functional: softmax_forward, softmax_backward
=#
using Test
using CUDA

# check if cuda is available
use_cuda = CUDA.functional()

#=
# the backend cpu functions in src/functional are tested against cuda to ensure mutual correctness
if use_cuda
    println("Test Convolution"); include("gv_convolution_cuda_test.jl")
    println("Test Deconvolution"); include("gv_deconvolution_cuda_test.jl")
    println("Test Pooling"); include("gv_pooling_cuda_test.jl")
    println("Test Adaptive Pooling"); include("gv_adaptive_pooling_cuda_test.jl")
    println("Test Batch Normalization"); include("gv_batch_normalization_cuda_test.jl")
    println("Test Reshape/Flatten"); include("gv_reshape_flatten_cuda_test.jl")
    println("Test Fully Connected"); include("gv_fully_connected_cuda_test.jl")
    println("Test Activation Functions"); include("gv_activation_functions_cuda_test.jl")
end
=#

# test the DataLoader (only works on the cpu currently)
include("gv_data_loader_test.jl")

# test weight and layer initialization, computational graph creation and automatic differentiation tools (e.g. backward pass) in a combined example on the cpu
println("Combined Test on the CPU"); include("combined_cpu_test.jl")
# do the same on the gpu
if use_cuda
    println("Combined Test on the GPU"); include("combined_gpu_test.jl")
end