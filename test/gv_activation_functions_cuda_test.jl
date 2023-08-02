dims = 1
input_cpu = rand(dtype, 1000, 64) * 10
input_gpu = CuArray(input_cpu)
dtype = Float64

output_cpu = softmax_forward(input_cpu, dims=dims)
output_gpu = softmax_forward(input_gpu, dims=dims)

output_gradient_cpu = rand(dtype, size(output_cpu)...) * 100
output_gradient_gpu = CuArray(output_gradient_cpu)

input_gradient_cpu = softmax_backward(output_gradient_cpu, output_cpu, dims=dims)
input_gradient_gpu = softmax_backward(output_gradient_gpu, output_gpu, dims=dims)

@test isapprox(CuArray(output_cpu), output_gpu)
@test isapprox(CuArray(input_gradient_cpu), input_gradient_gpu)