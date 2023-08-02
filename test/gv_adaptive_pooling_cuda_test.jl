input_cpu = rand(dtype, 150, 150, 3, 64)
input_gpu = CuArray(input_cpu)
output_size = (50, 50)
dtype = Float64

#=
Average Pooling
=#

output_cpu = adaptive_average_pooling2d(input_cpu, output_size)
output_gpu = adaptive_average_pooling2d(input_gpu, output_size)

output_gradient_cpu = rand(dtype, size(output_cpu)...)
output_gradient_gpu = CuArray(output_gradient_cpu)

input_gradient_cpu = adaptive_average_pooling2d_backward(output_gradient_cpu, input_cpu)
input_gradient_gpu = adaptive_average_pooling2d_backward(output_gradient_gpu, input_gpu)

@test isapprox(CuArray(output_cpu), output_gpu)
@test isapprox(CuArray(input_gradient_cpu), input_gradient_gpu)

#=
Maximum Pooling
=#

output_cpu, positions = adaptive_maximum_pooling2d(input_cpu, output_size, return_data_for_backprop=true)
output_gpu = adaptive_maximum_pooling2d(input_gpu, output_size)

output_gradient_cpu = rand(dtype, size(output_cpu)...)
output_gradient_gpu = CuArray(output_gradient_cpu)

input_gradient_cpu = adaptive_maximum_pooling2d_backward(output_gradient_cpu, input_cpu, positions)
input_gradient_gpu = adaptive_maximum_pooling2d_backward(output_gradient_gpu, input_gpu)

@test isapprox(CuArray(output_cpu), output_gpu)
@test isapprox(CuArray(input_gradient_cpu), input_gradient_gpu)