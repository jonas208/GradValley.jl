kernel_size = (3, 3)
stride = (2, 1)
padding = (1, 0)
dilation = (1, 1) # for pooling, only dilation = (1, 1) is supported in cuDNN
dtype = Float64

input_cpu = rand(dtype, 150, 150, 4, 64)
input_gpu = CuArray(input_cpu)

#=
Average Pooling
=#

output_cpu = average_pooling2d(input_cpu, kernel_size, stride=stride, padding=padding, dilation=dilation)
output_gpu = average_pooling2d(input_gpu, kernel_size, stride=stride, padding=padding, dilation=dilation)

output_gradient_cpu = rand(dtype, size(output_cpu)...)
output_gradient_gpu = CuArray(output_gradient_cpu)

input_gradient_cpu = average_pooling2d_backward(output_gradient_cpu, input_cpu, kernel_size, stride=stride, padding=padding, dilation=dilation)
input_gradient_gpu = average_pooling2d_backward(output_gradient_gpu, output_gpu, input_gpu, kernel_size, stride=stride, padding=padding, dilation=dilation)

@test isapprox(CuArray(output_cpu), output_gpu)
@test isapprox(CuArray(input_gradient_cpu), input_gradient_gpu)

#=
Maximum Pooling
=#

output_cpu, positions = maximum_pooling2d(input_cpu, kernel_size, stride=stride, padding=padding, dilation=dilation, return_data_for_backprop=true)
output_gpu = maximum_pooling2d(input_gpu, kernel_size, stride=stride, padding=padding, dilation=dilation)

output_gradient_cpu = rand(dtype, size(output_cpu)...)
output_gradient_gpu = CuArray(output_gradient_cpu)

input_gradient_cpu = maximum_pooling2d_backward(output_gradient_cpu, input_cpu, positions, padding=padding)
input_gradient_gpu = maximum_pooling2d_backward(output_gradient_gpu, output_gpu, input_gpu, kernel_size, stride=stride, padding=padding, dilation=dilation)

@test isapprox(CuArray(output_cpu), output_gpu)
@test isapprox(CuArray(input_gradient_cpu), input_gradient_gpu)